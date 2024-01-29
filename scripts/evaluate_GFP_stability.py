import sys
sys.path.append('.')
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from Bio import SeqIO
import argparse, datetime, json, time
import pandas as pd
import numpy as np
from MosPro.utils import common
from MosPro.utils import eval
from MosPro.utils.common import sec2min_sec
from MosPro.utils.eval import calc_hypervolume, diversity, novelty, greedy_selection
from MosPro.models.predictors import BaseCNN
import torch
import random
from MosPro.datasets.fitness_dataset import seq2indices


now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logger = common.get_logger(__name__)

def prepare_foldx_data(sampled_seqs, tmp_dir, ref_seq_fasta):
    records = list(SeqIO.parse(ref_seq_fasta, 'fasta'))
    ref_seq = str(records[0].seq)
    for s in sampled_seqs:
        assert len(s) == len(ref_seq), f'{len(s)}!= {len(ref_seq)}'
    individual_list_dir = os.path.join(tmp_dir, 'individual_list')
    os.makedirs(individual_list_dir, exist_ok=True)
    mut_str_list = []
    for i in range(len(sampled_seqs)):
        assert len(sampled_seqs[i]) == len(ref_seq)
        mut_str = []
        for k in range(len(sampled_seqs[i])):
            if sampled_seqs[i][k] != ref_seq[k]:
                mut_str.append(f'{ref_seq[k]}A{k+1}{sampled_seqs[i][k]}')
        mut_str = ','.join(mut_str)
        mut_str_list.append(mut_str)
    logger.info(f'mut_str: {mut_str_list[:10]}')
    assert len(mut_str_list) == len(sampled_seqs) == len(set(mut_str_list))
    with open(os.path.join(tmp_dir, 'mut_seqs.json'), 'w') as f:
        json.dump(mut_str_list, f)
    batch_size = 2
    num_batches = int(np.ceil(len(mut_str_list) / batch_size))
    logger.info(f'num_batches: {num_batches}')
    for i in range(num_batches):
        start = i * batch_size
        end = min((i+1) * batch_size, len(mut_str_list))
        with open(os.path.join(individual_list_dir, f'individual_list_{i}.txt'), 'w') as f:
            for k in range(start, end):
                f.write(f'{mut_str_list[k]};\n')
    out_dir = os.path.join(tmp_dir, 'output')
    # input()
    return individual_list_dir, out_dir, num_batches

def foldx_runner(batch_idx, pdb_dir, pdb_file, mut_file, out_dir, num_runs=5):
    cmd = f'./FoldX --command=BuildModel --pdb-dir={pdb_dir} --pdb={pdb_file} --output-dir={out_dir} --mutant-file={mut_file} --numberOfRuns={num_runs} --out-pdb=false --output-file=batch_{batch_idx} > logs_foldx/foldx_batch_{batch_idx}.log'
    os.system(cmd)

def run_foldx(pdb_dir, repaired_ref_pdb, individual_list_dir, out_dir, num_batches, num_threads):
    start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    mut_file_list = [os.path.join(individual_list_dir, f'individual_list_{i}.txt') for i in range(num_batches)]
    batches = [i for i in range(num_batches)]
    Parallel(n_jobs=num_threads, verbose=10)(delayed(foldx_runner)(i, pdb_dir, repaired_ref_pdb, mut_file_list[i], out_dir) for i in batches)
    end = time.time()
    logger.info(f'FoldX running time: {sec2min_sec(end - start)}')
    
def collect_foldx_results(out_dir, num_batches, sampled_seqs, uncached_seqs, save_dir, config, args):
    uncached_ddg_list = []
    for i in range(num_batches):
        with open(os.path.join(out_dir, f'Average_batch_{i}_ref_seq_af2_Repair.fxout')) as f:
            lines = f.readlines()[9:]
            uncached_ddg_list.extend([float(l.split('\t')[2]) for l in lines])
    logger.info(f'ddg: {len(uncached_ddg_list)}')
    assert len(uncached_ddg_list) == len(uncached_seqs)
    uncached_seq2ddg = {uncached_seqs[i]: uncached_ddg_list[i] for i in range(len(uncached_seqs))}
    ddg_list = []
    with open(config.foldx_cache) as f:
        foldx_cache = json.load(f)
    for seq in sampled_seqs:
        if seq in foldx_cache:
            ddg_list.append(foldx_cache[seq])
        else:
            ddg_list.append(uncached_seq2ddg[seq])
            foldx_cache[seq] = uncached_seq2ddg[seq]
    with open(config.foldx_cache, 'w') as f:
        json.dump(foldx_cache, f)
    logger.info(f'foldx_cache: {len(foldx_cache)}')
    # df = pd.DataFrame({
    #         'sequence': sampled_seqs,
    #         'ddg': ddg_list
    #     })
    # df.to_csv(os.path.join(save_dir, 'foldx_results.csv'), index=False)
    
    gt = pd.read_csv(config.ground_truth_stability)
    max_seen_ddg = np.max(gt.target)
    min_seen_ddg = np.min(gt.target)
    normalize = lambda x: (x - min_seen_ddg) / (max_seen_ddg - min_seen_ddg)
    normalized_ddg_list = [normalize(x) for x in ddg_list]
    
    # logger.info(f'task\tmean\tmedian\tstd\tmax\tmin')
    # logger.info(f'stability\t{np.mean(ddg_list):.3f}\t{np.median(ddg_list):.3f}\t{np.std(ddg_list):.3f}\t{np.max(ddg_list):.3f}\t{np.min(ddg_list):.3f}')
    # logger.info(f'stability_normalized\t{np.mean(normalized_ddg_list):.3f}\t{np.median(normalized_ddg_list):.3f}\t{np.std(normalized_ddg_list):.3f}\t{np.max(normalized_ddg_list):.3f}\t{np.min(normalized_ddg_list):.3f}')
    
    results = {'scores_origin': ddg_list,
               'scores_normalized': normalized_ddg_list,}
    
    return results
    # with open(os.path.join(save_dir, 'stability_metrics.txt'), 'w') as f:
    #     f.write('num_unique,mean_fitness,median_fitness,std_fitness,max_fitness,min_fitness,source_path\n')
    #     f.write(f'{len(set(sampled_seqs))},{np.mean(ddg_list)},{np.median(ddg_list)},{np.std(ddg_list)},{np.max(ddg_list)},{np.min(ddg_list)},{args.sample}\n')
    #     f.write(f'{len(set(sampled_seqs))},{np.mean(normalized_ddg_list)},{np.median(normalized_ddg_list)},{np.std(normalized_ddg_list)},{np.max(normalized_ddg_list)},{np.min(normalized_ddg_list)},{args.sample}\n')

def evaluate_stability(sampled_seqs, config, args):
    with open(config.foldx_cache) as f:
        foldx_cache = json.load(f)
    uncached_seqs = list(set(sampled_seqs) - set(foldx_cache.keys()))
    logger.info(f'uncached_seqs: {len(uncached_seqs)}')
    tmp_dir = config.tmp_dir + now
    os.makedirs(tmp_dir, exist_ok=True)
    individual_list_dir, out_dir, num_batches = prepare_foldx_data(uncached_seqs, tmp_dir, config.ref_seq_fasta)
    run_foldx(os.path.dirname(config.repaired_ref_pdb), os.path.basename(config.repaired_ref_pdb), individual_list_dir, out_dir, num_batches, args.num_threads)
    results = collect_foldx_results(out_dir, num_batches, sampled_seqs, uncached_seqs, os.path.dirname(config.sample_path), config, args)
    
    return results

def evaluate_GFP(sampled_seqs, config, args):
    for root, dirs, files in os.walk(config.GFP_oracle_dir):
        for file in files:
            if file.endswith('.yml'):
                oracle_config = common.load_config(os.path.join(root, file))
                break
    oracle = globals()[oracle_config.model.model_type](oracle_config.model)
    ckpt = torch.load(os.path.join(config.GFP_oracle_dir, 'checkpoints/best_checkpoints.pt'))
    oracle.load_state_dict(ckpt)
    oracle.eval()
    oracle.to(args.device)
    logger.info(f'Loaded oracle from {config.GFP_oracle_dir}')
    outputs = []
    for seq in tqdm(sampled_seqs, desc='running oracle'):
        seq = seq2indices(seq).unsqueeze(0).to(args.device)
        with torch.no_grad():
            output = oracle(seq).item()
        outputs.append(output)
        
    gt = pd.read_csv(config.ground_truth_GFP)
    max_seen = np.max(gt.target)
    min_seen = np.min(gt.target)
    normalize = lambda x: (x - min_seen) / (max_seen - min_seen)
    outputs_normalized = [normalize(x) for x in outputs]
    logger.info(f'task\tmean\tmedian\tstd\tmax\tmin')
    logger.info(f'GFP\t{np.mean(outputs):.3f}\t{np.median(outputs):.3f}\t{np.std(outputs):.3f}\t{np.max(outputs):.3f}\t{np.min(outputs):.3f}')
    logger.info(f'GFP_normalized\t{np.mean(outputs_normalized):.3f}\t{np.median(outputs_normalized):.3f}\t{np.std(outputs_normalized):.3f}\t{np.max(outputs_normalized):.3f}\t{np.min(outputs_normalized):.3f}')
    results = {'scores_origin': outputs, 
               'scores_normalized': outputs_normalized,}
    
    return results

def get_hypervolume(stability_results, GFP_results, config):
    gt_GFP = pd.read_csv(config.ground_truth_GFP)
    gt_stability = pd.read_csv(config.ground_truth_stability)
    min_seen_GFP = np.min(gt_GFP.target)
    max_seen_stability = np.max(gt_stability.target)
    scores_stability = stability_results['scores_origin']
    scores_GFP = GFP_results['scores_origin']
    # inverse_scores_stability = [-x for x in scores_stability]
    inverse_scores_GFP = [-x for x in scores_GFP]
    # inverse_min_seen_stability = -min_seen_stability
    inverse_min_seen_GFP = -min_seen_GFP
    hv = calc_hypervolume(scores_stability, inverse_scores_GFP, max_seen_stability, inverse_min_seen_GFP)
    
    scores_GFP_normalized = GFP_results['scores_normalized']
    inverse_scores_GFP_normalized = [-x for x in scores_GFP_normalized]
    hv_normalized = calc_hypervolume(stability_results['scores_normalized'], inverse_scores_GFP_normalized, 1, 0)
    # calculate reference hv
    worst_scores_stability = [np.max(gt_stability.target)] * len(scores_stability)
    worst_scores_GFP = [np.min(gt_GFP.target)] * len(scores_GFP)
    inverse_worst_scores_GFP = [-x for x in worst_scores_GFP]
    ref_hv = calc_hypervolume(worst_scores_stability, inverse_worst_scores_GFP, max_seen_stability, inverse_min_seen_GFP)
    logger.info(f'ref_hv_worst: {ref_hv}')
    
    best_scores_stability = [np.min(gt_stability.target)] * len(scores_stability)
    best_scores_GFP = [np.max(gt_GFP.target)] * len(scores_GFP)
    inverse_best_scores_GFP = [-x for x in best_scores_GFP]
    ref_hv = calc_hypervolume(best_scores_stability, inverse_best_scores_GFP, max_seen_stability, inverse_min_seen_GFP)
    logger.info(f'ref_hv_best: {ref_hv}')
    
    ref_hv = calc_hypervolume([0] * len(scores_stability), [-1] * len(scores_GFP), 1, 0)
    logger.info(f'ref_hv_best_norm: {ref_hv}')
    # input()
    return hv, hv_normalized

def evaluate_all(stability_results, GFP_results, sampled_seqs, config, args):
    hv, hv_normalized = get_hypervolume(stability_results, GFP_results, config)
    logger.info(f'Hypervolume: {hv:.3f}')
    base_pool_df = pd.read_csv(config.base_pool_path)
    base_pool_seqs = base_pool_df.sequence.tolist()
    all_novelty = novelty(sampled_seqs, base_pool_seqs)
    scores_save_path = os.path.join(os.path.dirname(config.sample_path), 'evaluation_results.csv' if args.tag is None else f'evaluation_results_{args.tag}.csv')
    metrics_save_path = os.path.join(os.path.dirname(config.sample_path), 'evaluation_metrics.csv' if args.tag is None else f'evaluation_metrics_{args.tag}.csv')
    scores_df = pd.DataFrame({
        'sequence': sampled_seqs,
        'stability': stability_results['scores_origin'],
        'GFP': GFP_results['scores_origin'],
        'novelty': all_novelty,
        'stability_normalized': stability_results['scores_normalized'],
        'GFP_normalized': GFP_results['scores_normalized'],
    })
    scores_df.to_csv(scores_save_path, index=False)
    metrics_df = pd.DataFrame({
        'num_seqs': [len(sampled_seqs), len(sampled_seqs), len(sampled_seqs), len(sampled_seqs)],
        'task': ['stability', 'GFP', 'stability', 'GFP'],
        'normalized': ['True', 'True', 'False', 'False'],
        'hv': [hv_normalized, hv_normalized, hv, hv],
        'mean_fitness': [np.mean(stability_results['scores_normalized']), np.mean(GFP_results['scores_normalized']), np.mean(stability_results['scores_origin']), np.mean(GFP_results['scores_origin'])],
        'median_fitness': [np.median(stability_results['scores_normalized']), np.median(GFP_results['scores_normalized']), np.median(stability_results['scores_origin']), np.median(GFP_results['scores_origin'])],
        'std_fitness': [np.std(stability_results['scores_normalized']), np.std(GFP_results['scores_normalized']), np.std(stability_results['scores_origin']), np.std(GFP_results['scores_origin'])],
        'max_fitness': [np.max(stability_results['scores_normalized']), np.max(GFP_results['scores_normalized']), np.max(stability_results['scores_origin']), np.max(GFP_results['scores_origin'])],
        'min_fitness': [np.min(stability_results['scores_normalized']), np.min(GFP_results['scores_normalized']), np.min(stability_results['scores_origin']), np.min(GFP_results['scores_origin'])],
        'diversity': [diversity(sampled_seqs), diversity(sampled_seqs), diversity(sampled_seqs), diversity(sampled_seqs)],
        'mean_novelty': [np.mean(all_novelty), np.mean(all_novelty), np.mean(all_novelty), np.mean(all_novelty)],
        'median_novelty': [np.median(all_novelty), np.median(all_novelty), np.median(all_novelty), np.median(all_novelty)],
    })
    print(metrics_df)
    metrics_df.to_csv(metrics_save_path, index=False)
    logger.info(f'Saved scores to {scores_save_path}')
    logger.info(f'Saved metrics to {metrics_save_path}')

def topk_selection(sample_path, topk1, topk2):
    df = pd.read_csv(sample_path)
    df = df.drop_duplicates(subset='mutant_sequences', ignore_index=True)
    if 'mutant_scores_1' not in df.columns:
        df['mutant_scores_1'] = df['mutant_scores']
    if 'mutant_scores_2' not in df.columns:
        df['mutant_scores_2'] = df['mutant_scores']
    df = df.sort_values('mutant_scores_1', ascending=False)
    sampled_seqs_1 = df.mutant_sequences.tolist()[:topk1]
    df = df.sort_values('mutant_scores_2', ascending=False)
    sampled_seqs_2 = df.mutant_sequences.tolist()[:topk2]
    sampled_seqs = list(set(sampled_seqs_1 + sampled_seqs_2))
    logger.info(f'topk1: {len(sampled_seqs_1)}, topk2: {len(sampled_seqs_2)}')
    logger.info(f'Intersections: {len(set(sampled_seqs_1) & set(sampled_seqs_2))}')
    logger.info(f'Sampled {len(sampled_seqs)} unique sequences for evaluation.')

    return sampled_seqs

def random_selection(config):
    logger.warning('Randomly selecting sequences for evaluation.')
    num_select = config.num_select
    df = pd.read_csv(config.sample_path)
    # df = df.drop_duplicates(subset='mutant_sequences', ignore_index=True)
    if 'mutant_sequences' in df.columns:
        sampled_seqs = df.mutant_sequences.tolist()
    else:
        sampled_seqs = df.sequence.tolist()
    # sampled_seqs = np.random.choice(sampled_seqs, num_select, replace=False)
    sampled_seqs = random.choices(sampled_seqs, k=num_select)
    # sampled_seqs = list(set(sampled_seqs))
    logger.info(f'Sampled {len(sampled_seqs)} sequences for evaluation.')
    
    return sampled_seqs

def nested_selection(config):
    logger.warning('Nested selecting sequences for evaluation.')
    sample_path = config.sample_path
    num_select = config.num_select
    df = pd.read_csv(sample_path)
    df = df.drop_duplicates(subset='mutant_sequences', ignore_index=True)
    df = df.sort_values('mutant_scores_1', ascending=False).head(num_select * 10)
    df = df.sort_values('mutant_scores_2', ascending=False).head(num_select)
    sampled_seqs = df.mutant_sequences.tolist()
    sampled_seqs = list(set(sampled_seqs))
    logger.info(f'Sampled {len(sampled_seqs)} unique sequences for evaluation.')
    
    return sampled_seqs

def single_task_selection(sample_path, topk1, topk2):
    logger.warning('Single task selecting sequences for evaluation.')
    num_select = topk1 + topk2
    df = pd.read_csv(sample_path)
    # print(len(df))
    df = df.drop_duplicates(subset='mutant_sequences', ignore_index=True)
    # print(len(df))
    # input()
    df = df.sort_values('mutant_scores', ascending=False).head(num_select)
    sampled_seqs = df.mutant_sequences.tolist()
    sampled_seqs = list(set(sampled_seqs))
    logger.info(f'Sampled {len(sampled_seqs)} unique sequences for evaluation.')
    
    return sampled_seqs

def max_hv_selection(sample_path, topk1, topk2):
    logger.warning('Max hv selecting sequences for evaluation.')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs/evaluate.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--sample_path', type=str, default=None)
    parser.add_argument('--num_threads', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--select_path', type=str, default=None)

    return parser.parse_args()

def main():
    args = get_args()
    config = common.load_config(args.config)
    common.seed_all(args.seed)
    if args.sample_path is not None:
        config.sample_path = args.sample_path
    logger.info(f'Evaluating {config.sample_path}...')
    # sampled_seqs = globals()[config.selection_method](config.sample_path, config.topk1, config.topk2)
    if args.select_path is not None:
        sampled_seqs = pd.read_csv(args.select_path).sequence.tolist()
    else:
        sampled_seqs = globals()[config.selection_method](config)
    logger.info(f'Selected {len(sampled_seqs)} sequences for evaluation.')
    # sampled_seqs = greedy_selection(config.sample_path, )
    GFP_results = evaluate_GFP(sampled_seqs, config, args)
    logger.info('Finished evaluating GFP.')
    stability_results = evaluate_stability(sampled_seqs, config, args)
    logger.info('Finished evaluating stability.')
    evaluate_all(stability_results, GFP_results, sampled_seqs, config, args)

if __name__ == '__main__':
    main()
    
    

