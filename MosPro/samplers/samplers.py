import sys
sys.path.append('.')
import torch
import numpy as np
import logging
import time
import torch.distributions as dists
import torch.nn.functional as F
import pandas as pd
from MosPro.models.predictors import BaseCNN
from omegaconf import OmegaConf
import os
import torch.nn as nn
from MosPro.utils.pareto_moosvgd import compose_two_gradients_moosvgd
from MosPro.utils.pareto_pref_vec import get_d_paretomtl, circle_points, get_d_paretomtl_batch
from typing import List
from MosPro.utils.min_norm_solvers import MinNormSolver

to_np = lambda x: x.cpu().detach().numpy()
to_list = lambda x: to_np(x).tolist()
alpha_list = []
# Wrapper class to reverse the sign of the output
class InversePredictor(nn.Module):
    def __init__(self, model):
        super(InversePredictor, self).__init__()
        print(f'InversePredictor: {model.__class__.__name__}')
        self.model = model

    def forward(self, x):
        return -self.model(x)

class Encoder(object):
    """convert between strings and their one-hot representations"""
    def __init__(self, alphabet: str = 'ARNDCQEGHILKMFPSTWYV'):
        self.alphabet = alphabet
        self.a_to_t = {a: i for i, a in enumerate(self.alphabet)}
        self.t_to_a = {i: a for i, a in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)
    
    @property
    def vocab(self) -> np.ndarray:
        return np.array(list(self.alphabet))
    
    @property
    def tokenized_vocab(self) -> np.ndarray:
        return np.array([self.a_to_t[a] for a in self.alphabet])

    def onehotize(self, batch):
        #create a tensor, and then onehotize using scatter_
        onehot = torch.zeros(len(batch), self.vocab_size)
        onehot.scatter_(1, batch.unsqueeze(1), 1)
        return onehot
    
    def encode(self, seq_or_batch: str or list, return_tensor = True) -> np.ndarray or torch.Tensor:
        if isinstance(seq_or_batch, str):
            encoded_list = [self.a_to_t[a] for a in seq_or_batch]
        else:
            encoded_list = [[self.a_to_t[a] for a in seq] for seq in seq_or_batch]
        return torch.tensor(encoded_list) if return_tensor else encoded_list
    
    def decode(self, x: np.ndarray or list or torch.Tensor) -> str or list:
        if isinstance(x, np.ndarray):
            x = x.tolist()
        elif isinstance(x, torch.Tensor):
            x = x.tolist()

        if isinstance(x[0], list):
            return [''.join([self.t_to_a[t] for t in xi]) for xi in x]
        else:
            return ''.join([self.t_to_a[t] for t in x])

def _mutagenesis_tensor(base_seq):
    base_seq = torch.squeeze(base_seq)  # Remove batch dimension.
    seq_len, vocab_len = base_seq.shape
    # Create mutagenesis tensor
    all_seqs = []
    for i in range(seq_len):
        for j in range(vocab_len):
            new_seq = base_seq.clone()
            new_seq[i][j] = 1
            all_seqs.append(new_seq)
    all_seqs = torch.stack(all_seqs)
    return all_seqs

class MosProSampler(torch.nn.Module):
    
    def __init__(
            self,
            config,
            predictor_1_dir: str,
            predictor_2_dir: str,
            temperature: float,
            ckpt_name: str,
            verbose: bool = False,
            gibbs_samples: int = 500,
            device: str = "cuda",
            inverse_sign_1: bool = False,
            inverse_sign_2: bool = False,
            gradient_compose_method: str = 'average',
            balance_weight_1 = 1,
            balance_weight_2 = 6,
            lambda_ = 1.0,
            mutation_sites = None,
            lambda_method = None,
            weight_1 = 1.0,
            weight_2 = 1.0,
            pref_index = 0,
        ):
        super().__init__()
        self.config = config
        self._ckpt_name = ckpt_name
        self._log = logging.getLogger(__name__)
        self.device = torch.device(device)
        self.inverse_sign_1 = inverse_sign_1
        self.inverse_sign_2 = inverse_sign_2
        self.gradient_compose_method = gradient_compose_method
        print(f'gradient_compose_method: {self.gradient_compose_method}')
        print(f'Using device: {self.device}')
        self.predictor_tokenizer = Encoder()
        self.predictor_1 = self._setup_predictor(predictor_1_dir, self.inverse_sign_1)
        self.predictor_2 = self._setup_predictor(predictor_2_dir, self.inverse_sign_2)
        print(f'predictor_1 dir: {predictor_1_dir}')
        print(f'predictor_2 dir: {predictor_2_dir}')
        self.num_tokens = len(self.predictor_tokenizer.alphabet)
        self.temp = temperature
        self.total_pairs = 0
        self.num_current_src_seqs = 0
        self.gibbs_samples = gibbs_samples
        self._verbose = verbose
        self.balance_weight_1 = balance_weight_1
        self.balance_weight_2 = balance_weight_2
        self.lambda_ = lambda_
        self.mutation_sites = mutation_sites
        self.lambda_method = lambda_method
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        if self.gradient_compose_method == 'pareto_pref_vec':
            self.pref_vec = torch.tensor(circle_points([1], [config.num_pref_vecs])[0]).float().to(self.device)
        self.pref_index = pref_index
        self.normalize_grad = config.normalize_grad
        print(f'normalize_grad: {self.normalize_grad}')
        print(f'using temperature: {self.temp}')
        print(f'balance_weight_1: {self.balance_weight_1}, balance_weight_2: {self.balance_weight_2}, lambda_: {self.lambda_}')
        print(f'mutation_sites: {self.mutation_sites}')
        print(f'lambda_method: {self.lambda_method}')
        if self.lambda_method == '2_weight':
            print(f'weight_1: {weight_1}, weight_2: {weight_2}')
        if self.gradient_compose_method == 'pareto_pref_vec':
            print(f'pref_index: {self.pref_index}')
            print(f'pref_vec: {self.pref_vec}')
        elif self.gradient_compose_method == 'linear':
            print(f'Linear weights: [{config.linear_weight_1}, {1 - config.linear_weight_1}]')
        
    def _setup_predictor(self, predictor_dir: str, inverse_sign: bool):
        # Load model weights.
        ckpt_path = os.path.join(predictor_dir, 'checkpoints', self._ckpt_name)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        for root, dirs, files in os.walk(predictor_dir):
            for file in files:
                if file.endswith('.yml'):
                    cfg_path = os.path.join(root, file)
        # cfg_path = os.path.join(predictor_dir, 'config.yaml')
        with open(cfg_path, 'r') as fp:
            ckpt_cfg = OmegaConf.load(fp.name)
        ckpt_cfg.model.make_one_hot = False
        # predictor = BaseCNN(ckpt_cfg.model)
        predictor = globals()[ckpt_cfg.model.model_type](ckpt_cfg.model)
        predictor.load_state_dict(ckpt)
        predictor.eval()
        predictor.to(self.device)
        self._log.info(predictor)
        if inverse_sign:
            predictor = InversePredictor(predictor)
        return predictor

    def tokenize_seqs(self, seqs):
        return self.gen_tokenizer.encode(seqs)

    # def _calc_local_diff(self, seq_one_hot):
    #     # Construct local difference
    #     gx = torch.autograd.grad(self.predictor(seq_one_hot).sum(), seq_one_hot)[0]
    #     gx_cur = (gx * seq_one_hot).sum(-1)[:, :, None]
    #     delta_ij = gx - gx_cur
    #     return delta_ij

    def _compute_alpha(self, M, N):
        # Calculate Trace(N^T * N)
        # trace_NtN = torch.trace(torch.mm(N.t(), N))
        trace_NtN = (N * N).sum()

        # Calculate Trace(M^T * N)
        # trace_MtN = torch.trace(torch.mm(M.t(), N))
        trace_MtN = (M * N).sum()

        # Calculate || M - N ||_2^2
        frobenius_norm_squared = torch.norm(M - N, p='fro')**2

        # Compute a
        a = (trace_NtN - trace_MtN) / frobenius_norm_squared
        a = a
        a = torch.max(torch.min(a, torch.tensor(1.0)), torch.tensor(0.0))
        # print(a.item())
        return a
    
    def _compose_two_gradients(self, grad1, grad2, gradient_compose_method, inputs, score_1, score_2):
        if self.normalize_grad:
            grad1 = grad1 / torch.norm(grad1, dim=[1,2], keepdim=True)
            grad2 = grad2 / torch.norm(grad2, dim=[1,2], keepdim=True)
        # print(f'grad1: {grad1.shape}, grad2: {grad2.shape}')
        # print(f'grad1 norm: {torch.norm(grad1, dim=[1,2])}')
        # print(f'grad2 norm: {torch.norm(grad2, dim=[1,2])}')
        # input()
        if gradient_compose_method == 'linear':
            # return (grad1 * self.balance_weight_1 + grad2 * self.balance_weight_2) / 2
            return grad1 * self.config.linear_weight_1 + grad2 * (1 - self.config.linear_weight_1)
        elif gradient_compose_method == 'pareto':
            # print(grad1.shape, grad2.shape)
            alpha = self._compute_alpha(grad1 * self.balance_weight_1, grad2 * self.balance_weight_2)
            # print(f'grad1: {grad1.shape}, grad2: {grad2.shape}, alpha: {alpha.shape}')
            # print(f'alpha: {alpha}')
            # input()
            if self.lambda_method == 'alpha_lambda':
                return alpha * self.lambda_ * grad1 + (1 - alpha * self.lambda_) * grad2
            elif self.lambda_method == 'lambda':
                return alpha * grad1 + self.lambda_ * (1 - alpha) * grad2
            elif self.lambda_method == '2_weight':
                return alpha * self.weight_1 * grad1 + (1 - alpha) * self.weight_2 * grad2
            else:
                raise NotImplementedError
        elif gradient_compose_method == 'pareto_moosvgd':
            return compose_two_gradients_moosvgd(grad1, grad2, inputs, score_1, score_2)
        elif gradient_compose_method == 'pareto_pref_vec':
            grads = [grad1, grad2]
            scores = [score_1, score_2]
            weights = get_d_paretomtl_batch(grads, scores, self.pref_vec, self.pref_index)
            weights = weights.to(self.device)
            # print(f'weights: {weights.shape}, {weights[0]}, {weights[1]}')
            # print(f'grad1: {grad1.shape}, grad2: {grad2.shape}')
            # print(f'weights: {weights.shape}, {weights}, {weights[:,0].shape}, {weights[:,0]}')
            # input()
            return weights[:, 0].reshape((-1, 1, 1)) * grad1 + weights[:, 1].reshape((-1, 1, 1)) * grad2
        elif gradient_compose_method == 'mgda':
            grad1 = self.balance_weight_1 * grad1
            grad2 = self.balance_weight_2 * grad2
            b, l, h = grad1.shape
            l2_normalize = lambda x: x / torch.norm(x, dim=-1, p=2, keepdim=True)
            if self.normalize_grad:
                grads = [l2_normalize(grad1.reshape(b, -1)), l2_normalize(grad2.reshape(b, -1))]
            else:
                grads = [grad1.reshape(b, -1), grad2.reshape(b, -1)]
            weights = MinNormSolver.find_min_norm_element(grads)[0]
            grads = [grads[0].reshape(b, l, h), grads[1].reshape(b, l, h)]
            composed_grad = weights[0] * grads[0] + weights[1] * grads[1]
            # print(f'grad1: {grad1.shape}, grad2: {grad2.shape}')
            # print(f'grads: {grads[0].shape}, {grads[1].shape}')
            # print(f'grad1 norm: {torch.norm(grad1, keepdim=True)}')
            # print(f'grad2 norm: {torch.norm(grad2, keepdim=True)}')
            # print(f'grad2 norm / grad1 norm: {torch.norm(grad2, keepdim=True) / torch.norm(grad1, keepdim=True)}')
            # print(f'score_1 norm: {torch.norm(score_1, keepdim=True)}')
            # print(f'score_2 norm: {torch.norm(score_2, keepdim=True)}')
            # print(f'score_2 norm / score_1 norm: {torch.norm(score_2, keepdim=True) / torch.norm(score_1, keepdim=True)}')
            # print(f'weights: {weights}, {weights.shape}')
            # print(f'composed_grad: {composed_grad.shape}')
            # input()
            return composed_grad
        else:
            raise NotImplementedError
    
    def _calc_local_diff_2(self, seq_one_hot):
        # Construct local difference for two predictor case
        score_1 = self.predictor_1(seq_one_hot)
        score_2 = self.predictor_2(seq_one_hot)
        gx1 = torch.autograd.grad(score_1.sum(), seq_one_hot, retain_graph=True)[0]
        gx2 = torch.autograd.grad(score_2.sum(), seq_one_hot, retain_graph=True)[0]
        gx = self._compose_two_gradients(gx1, gx2, self.gradient_compose_method, seq_one_hot, score_1, score_2)
        # print(f'gx1: {gx1.shape}, gx2: {gx2.shape}, gx: {gx.shape}')
        # input()
        gx_cur = (gx * seq_one_hot).sum(-1)[:, :, None]
        delta_ij = gx - gx_cur
        return delta_ij
    
    def set_rows_to_neg_inf(self, tensor, indices):
        n = tensor.shape[0]
        for i in range(n):
            if i not in indices:
                tensor[i] = torch.full_like(tensor[i], float(-1e9), device=tensor.device)
        return tensor
    
    def _gibbs_sampler(self, seq_one_hot):
        delta_ij = self._calc_local_diff_2(seq_one_hot)
        delta_ij = delta_ij[0]
        if self.mutation_sites is not None:
            delta_ij = self.set_rows_to_neg_inf(delta_ij, self.mutation_sites)
        # One step of GWG sampling.
        def _gwg_sample():
            seq_len, num_tokens = delta_ij.shape
            # Construct proposal distributions
            gwg_proposal = dists.OneHotCategorical(logits = delta_ij.flatten() / self.temp)
            r_ij = gwg_proposal.sample((self.gibbs_samples,)).reshape(
                self.gibbs_samples, seq_len, num_tokens)

            # [num_samples, L, 20]
            seq_token = torch.argmax(seq_one_hot, dim=-1)
            mutated_seqs = seq_token.repeat(self.gibbs_samples, 1)
            seq_idx, res_idx, aa_idx = torch.where(r_ij)
            mutated_seqs[(seq_idx, res_idx)] = aa_idx
            return mutated_seqs
        
        return _gwg_sample 

    def _make_one_hot(self, seq, differentiable=False):
        seq_one_hot = F.one_hot(seq, num_classes=self.num_tokens)
        if differentiable:
            seq_one_hot = seq_one_hot.float().requires_grad_()
        return seq_one_hot

    # def _evaluate_one_hot(self, seq):
    #     input_one_hot = self._make_one_hot(seq)
    #     model_out = self.predictor(input_one_hot)
    #     return model_out

    def _evaluate_one_hot_2(self, seq):
        input_one_hot = self._make_one_hot(seq)
        model_1_out = self.predictor_1(input_one_hot.float())
        model_2_out = self.predictor_2(input_one_hot.float())
        
        return model_1_out, model_2_out
    
    def _decode(self, one_hot_seq):
        return self.predictor_tokenizer.decode(one_hot_seq)
    
    def _metropolis_hasting(
            self, mutants, source_one_hot, delta_score, score_1, score_2, mutated_scores_1, mutated_scores_2):
       
        source = torch.argmax(source_one_hot, dim=-1)
    
        # [num_seq, L]
        mutated_indices = mutants != source[None]
        # [num_seq, L, 20]
        mutant_one_hot = self._make_one_hot(mutants, differentiable=True)
        mutated_one_hot = mutant_one_hot * mutated_indices[..., None]
        # print(f'MH: {source_one_hot[None].shape}')
        # print(f'source_delta_ij:')
        source_delta_ij = self._calc_local_diff_2(source_one_hot[None])
        # print(f'MH: {mutant_one_hot.shape}')
        # print(f'mutant_delta_ij:')
        mutant_delta_ij = self._calc_local_diff_2(mutant_one_hot)

        orig_source_shape = source_delta_ij.shape
        orig_mutant_shape = mutant_delta_ij.shape

        # Flatten starting from the second to last dimension and apply softmax
        q_source = source_delta_ij.flatten(start_dim=-2)
        q_source = F.softmax(q_source / self.temp, dim=-1)

        q_mutant = mutant_delta_ij.flatten(start_dim=-2)
        q_mutant = F.softmax(q_mutant / self.temp, dim=-1)

        # Reshape back to the original shape
        q_source = q_source.view(orig_source_shape).squeeze(0)
        q_mutant = q_mutant.view(orig_mutant_shape)
        
        mutation_tuple = torch.nonzero(mutated_one_hot, as_tuple=True)
        q_ij_source = q_source[mutation_tuple[1], mutation_tuple[2]]
        q_ij_mutant = q_mutant[torch.arange(q_mutant.shape[0]).to(self.device), mutation_tuple[1], mutation_tuple[2]] 
        q_ij_ratio = q_ij_mutant / q_ij_source
        accept_prob = torch.exp(delta_score)*q_ij_ratio.to(self.device)
        
        mh_step = accept_prob < torch.rand(accept_prob.shape).to(self.device)
        return mh_step

    def _evaluate_mutants(
            self,
            *,
            mutants,
            score_1,
            score_2,
            source_one_hot,
        ):
        all_mutated_scores_1, all_mutated_scores_2 = self._evaluate_one_hot_2(mutants)
        delta_score = (all_mutated_scores_1 - score_1 + all_mutated_scores_2 - score_2) / 2

        accept_mask = self._metropolis_hasting(
            mutants, source_one_hot, delta_score, score_1, score_2, all_mutated_scores_1, all_mutated_scores_2)
        accepted_x = to_list(mutants[accept_mask])
        accepted_seq = [self._decode(x) for x in accepted_x]
        accepted_score_1 = to_list(all_mutated_scores_1[accept_mask])
        accepted_score_2 = to_list(all_mutated_scores_2[accept_mask])
        return pd.DataFrame({
            'mutant_sequences': accepted_seq,
            'mutant_scores_1': accepted_score_1,
            'mutant_scores_2': accepted_score_2
        }), mutants[accept_mask]

    def compute_mutant_stats(self, source_seq, mutant_seqs):
        num_mutated_res = torch.sum(
            ~(mutant_seqs == source_seq[None]), dim=-1)
        return num_mutated_res

    def forward(self, batch):
        seqs = batch['sequences']

        #Tokenize
        tokenized_seqs = self.predictor_tokenizer.encode(seqs).to(self.device)
        total_num_seqs = len(tokenized_seqs)

        # Sweep over hyperparameters
        all_mutant_pairs = []
        for i, (real_seq, token_seq) in enumerate(zip(seqs, tokenized_seqs)):
            start_time = time.time()

            # Cast as float to take gradients through
            seq_one_hot = self._make_one_hot(token_seq, differentiable=True)

            # Compute base score
            pred_score_1, pred_score_2 = self._evaluate_one_hot_2(token_seq[None])
            pred_score_1, pred_score_2 = pred_score_1.item(), pred_score_2.item()
            
            # Construct Gibbs sampler
            sampler = self._gibbs_sampler(seq_one_hot[None])
            seq_pairs = []
            total_num_proposals = 0
            all_proposed_mutants = []
            all_accepted_mutants = []

            # Sample mutants
            proposed_mutants = sampler()
            num_proposals = proposed_mutants.shape[0]
            total_num_proposals += num_proposals
            proposed_num_edits = self.compute_mutant_stats(
                token_seq, proposed_mutants)
            proposed_mutants = proposed_mutants[proposed_num_edits > 0]
            all_proposed_mutants.append(to_np(proposed_mutants))

            # Run Gibbs generation of pairs
            sample_outputs, accepted_mutants = self._evaluate_mutants(
                mutants=proposed_mutants,
                score_1=pred_score_1,
                score_2=pred_score_2,
                source_one_hot=seq_one_hot
            )

            all_accepted_mutants.append(to_np(accepted_mutants))
            sample_outputs['source_sequences'] = real_seq
            sample_outputs['source_scores_1'] = pred_score_1
            sample_outputs['source_scores_2'] = pred_score_2

            seq_pairs.append(sample_outputs)
            if self._verbose:
                num_pairs = len(sample_outputs)
                print(
                    f'Temp: {self.temp:.3f}'
                    f'Accepted: {num_pairs}/{num_proposals} ({num_pairs/num_proposals:.2f})'
                )

            if len(seq_pairs) > 0:
                seq_pairs = pd.concat(seq_pairs).drop_duplicates(
                    subset=['source_sequences', 'mutant_sequences'],
                    ignore_index=True
                )
                all_mutant_pairs.append(seq_pairs)
            if self._verbose:
                elapsed_time = time.time() - start_time
                num_new_pairs = len(seq_pairs)
                all_proposed_mutants = np.concatenate(all_proposed_mutants, axis=0)
                proposed_res_freq = np.mean(
                    all_proposed_mutants != to_np(token_seq)[None], axis=0
                ).round(decimals=2)
                n_proposed = all_proposed_mutants.shape[0]

                all_accepted_mutants = np.concatenate(all_accepted_mutants, axis=0)
                accepted_res_freq = np.mean(
                    all_accepted_mutants != to_np(token_seq)[None], axis=0).round(decimals=2)
                n_accepted = all_accepted_mutants.shape[0]

                print(
                    f'Done with sequence {i+1}/{total_num_seqs} in {elapsed_time:.1f}s. '
                    f'Accepted {num_new_pairs}/{total_num_proposals} ({num_new_pairs/total_num_proposals:.2f}) sequences. \n'
                    f'Proposed sites (n={n_proposed}): {proposed_res_freq}. \n'
                    f'Accepted sites (n={n_accepted}): {accepted_res_freq}.'
                )


        if len(all_mutant_pairs) == 0:
            return None
        return pd.concat(all_mutant_pairs).drop_duplicates(
            subset=['source_sequences', 'mutant_sequences'],
            ignore_index=True
        )
