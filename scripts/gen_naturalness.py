import sys
sys.path.append('.')
import numpy as np
import torch
import esm
import pandas as pd
import os, json, time, copy, argparse, datetime, shutil
from tqdm import tqdm
from utils import common
from Bio import SeqIO

class NaturalnessGenerator:
    def __init__(self, model, device, logger):
        if 'esm' in model:
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model)
            self.layers = int(model.split('_')[1][1:])
        else:
            raise NotImplementedError(f'Unknown model {model}')
        self.device = device
        self.logger = logger
        self.model.eval()
        self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.num_tokens = len(self.alphabet.all_toks)
        
    def compute_pll(self, probs, sequence):
        '''compute pseudo-likelihood'''
        pll = 0
        for i, char in enumerate(sequence):
            idx = self.alphabet.get_idx(char)
            p_mlm = probs[i, i, idx]
            pll += p_mlm
        return pll
    
    def get_naturalness_one_seq(self, sequence):
        masked_strings = [sequence[:i] + "<mask>" + sequence[i+1:] for i in range(len(sequence))]
        batch = [(f'mask_seq_{i}', seq) for i, seq in enumerate(masked_strings)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(batch)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = self.model(batch_tokens.to(self.device), repr_layers=[self.layers])
        logits = results['logits'][:, 1: batch_lens[0]-1, :]
        probs = torch.softmax(logits, dim=-1)
        pll = self.compute_pll(probs, sequence)
        naturalness = torch.exp(pll / len(sequence))
        
        return naturalness.item()
    
    def get_naturalness(self, sequences):
        naturalness = []
        for seq in tqdm(sequences):
            score = self.get_naturalness_one_seq(seq)
            naturalness.append(score)
        assert len(naturalness) == len(sequences), f'Length mismatch: {len(naturalness)} vs {len(sequences)}'
        
        return naturalness
    
    def get_probs(self, seq, sites):
        masked_strings = [seq[:i] + "<mask>" + seq[i+1:] for i in sites]
        batch = [(f'mask_seq_{i}', seq) for i, seq in enumerate(masked_strings)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(batch)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = self.model(batch_tokens.to(self.device), repr_layers=[self.layers])
        logits = results['logits'][:, 1: batch_lens[0]-1, :]
        probs = torch.softmax(logits, dim=-1)
        prob_list = []
        for i, char in enumerate(seq):
            idx = self.alphabet.get_idx(char)
            p_mlm = probs[i, i, idx]
            prob_list.append(p_mlm.item())
        
        return torch.tensor(prob_list)
    
    def get_probs_cache(self, seq, sites):
        masked_strings = [seq[:i] + "<mask>" + seq[i+1:] for i in sites]
        probs = [self.seq2prob[seq].unsqueeze(0) for seq in masked_strings]
        probs = torch.vstack(probs)
        prob_list = []
        for i in range(len(sites)):
            idx = self.alphabet.get_idx(seq[sites[i]])
            p_mlm = probs[i, sites[i], idx]
            prob_list.append(p_mlm.item())
            
        return torch.tensor(prob_list)
    
    def get_naturalness_wt_one_seq(self, sequence, wt_seq):
        assert len(sequence) == len(wt_seq), f'Length mismatch: {len(sequence)} vs {len(wt_seq)}'
        mutation_sites = []
        for i, (char, wt_char) in enumerate(zip(sequence, wt_seq)):
            if char != wt_char:
                mutation_sites.append(i)
        if len(mutation_sites) == 0:
            return 0.0
        probs = self.get_probs_cache(sequence, mutation_sites)
        part_wt_probs = self.wt_probs[mutation_sites]
        pll = torch.sum(torch.log(probs / part_wt_probs))
        
        return pll.item()
    
    def precompute(self, sequences, wt_seq, batch_size=50):
        masked_strings = []
        for seq in sequences:
            mutation_sites = []
            for i, (char, wt_char) in enumerate(zip(seq, wt_seq)):
                if char != wt_char:
                    mutation_sites.append(i)
            masked_strings.extend([seq[:i] + "<mask>" + seq[i+1:] for i in mutation_sites])
        all_masked_strings = [(f'mask_seq_{i}', seq) for i, seq in enumerate(masked_strings)]
        self.logger.info(f'Precomputing {len(all_masked_strings)} masked sequences.')
        num_batches = int(np.ceil(len(all_masked_strings) / batch_size))
        seq2prob = {}
        for i in tqdm(range(num_batches), desc='precompute'):
            batch = all_masked_strings[i*batch_size: (i+1)*batch_size]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(batch)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
            with torch.no_grad():
                results = self.model(batch_tokens.to(self.device), repr_layers=[self.layers])
            logits = results['logits'][:, 1: batch_lens[0]-1, :]
            probs = torch.softmax(logits, dim=-1)
            for j, seq in enumerate(batch_strs):
                seq2prob[seq] = probs[j]
                
        return seq2prob
    
    def get_naturalness_wt(self, sequences, wt_seq):
        self.wt_probs = self.get_probs(wt_seq, list(range(len(wt_seq))))
        self.seq2prob = self.precompute(sequences, wt_seq)
        naturalness = []
        for seq in tqdm(sequences):
            score = self.get_naturalness_wt_one_seq(seq, wt_seq)
            naturalness.append(score)
        assert len(naturalness) == len(sequences), f'Length mismatch: {len(naturalness)} vs {len(sequences)}'
        
        return naturalness
    
    def get_naturalness_local_one_seq(self, sequence):
        pass
    
    def get_naturalness_local(self, sequences, batch_size=200):
        num_batches = int(np.ceil(len(sequences) / batch_size))
        batch_all = [(f'seq_{i}', seq) for i, seq in enumerate(sequences)]
        scores_all = []
        for i in tqdm(range(num_batches), desc='naturalness_local'):
            batch = batch_all[i*batch_size: (i+1)*batch_size]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(batch)
            batch_tokens_onehot = torch.nn.functional.one_hot(batch_tokens, num_classes=self.num_tokens).float().to(self.device)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
            with torch.no_grad():
                results = self.model(batch_tokens.to(self.device), repr_layers=[self.layers])
            logits = results['logits']
            score = (batch_tokens_onehot * torch.nn.functional.log_softmax(logits, -1)).sum(dim=[1,2])
            scores_all.extend(score.tolist())
        
        return scores_all
        
    
def load_single_seq_fasta(fasta_file):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    assert len(records) == 1, f'Expecting 1 sequence, got {len(records)}'
    return str(records[0].seq)

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='esm2_t33_650M_UR50D')
    parser.add_argument('--data', '-i', type=str)
    parser.add_argument('--output', '-o', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--logdir', type=str, default='./logs_naturalness')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--use_wt', action='store_true')
    parser.add_argument('--wt_fasta', type=str, default=None)
    parser.add_argument('--local', action='store_true')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    start_overall = time.time()
    args = get_args()
    common.seed_all(args.seed)
    
    # Logging
    log_dir = common.get_new_log_dir(args.logdir, prefix='gen_naturalness', tag=args.tag)
    logger = common.get_logger('naturalness', log_dir)
    logger.info(args)
    
    df = pd.read_csv(args.data)
    if args.start is not None and args.end is not None:
        df = df.iloc[args.start:args.end]
    sequences = df['sequence'].tolist()
    logger.info(f'Loaded {len(sequences)} sequences.')
    
    generator = NaturalnessGenerator(args.model, args.device, logger)
    if args.use_wt and args.wt_fasta is not None:
        logger.info(f'Using WT sequence from {args.wt_fasta}.')
        wt_seq = load_single_seq_fasta(args.wt_fasta)
        naturalness = generator.get_naturalness_wt(sequences, wt_seq)
    elif args.local:
        naturalness = generator.get_naturalness_local(sequences)
    else:
        naturalness = generator.get_naturalness(sequences)
    df['naturalness'] = naturalness
    df.to_csv(args.output, index=False)
    logger.info(f'Naturalness written to {args.output}.')
    end_overall = time.time()
    logger.info(f'Elapsed time: {common.sec2min_sec(end_overall - start_overall)}')


