import sys
sys.path.append('.')
import pandas as pd
import torch
from tqdm import tqdm
# from joblib import Parallel, delayed
# from Bio import SeqIO
import argparse, datetime, json, time, random, os
import numpy as np
from utils import common
from utils import eval
from utils.common import sec2min_sec
from utils.eval import calc_hypervolume, diversity, novelty, greedy_selection
from models import BaseCNN
from gen_naturalness import NaturalnessGenerator, load_single_seq_fasta
from scripts.evaluate_GFP_stability import random_selection
from datasets.fitness_dataset import seq2indices

now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logger = common.get_logger(__name__)

def evaluate_wt_naturalness(sampled_seqs, config, args):
    generator = NaturalnessGenerator(config.naturalness_model, args.device, logger)
    wt_seq = load_single_seq_fasta(config.ref_seq_fasta)
    naturalness = generator.get_naturalness_wt(sampled_seqs, wt_seq)
    gt = pd.read_csv(config.ground_truth_csv)
    max_seen = np.max(gt.naturalness)
    min_seen = np.min(gt.naturalness)
    normalize = lambda x: (x - min_seen) / (max_seen - min_seen)
    naturalness_normalized = [normalize(x) for x in naturalness]
    logger.info(f'task\tmean\tmedian\tstd\tmax\tmin')
    logger.info(f'naturalness\t{np.mean(naturalness):.3f}\t{np.median(naturalness):.3f}\t{np.std(naturalness):.3f}\t{np.max(naturalness):.3f}\t{np.min(naturalness):.3f}')
    logger.info(f'naturalness_normalized\t{np.mean(naturalness_normalized):.3f}\t{np.median(naturalness_normalized):.3f}\t{np.std(naturalness_normalized):.3f}\t{np.max(naturalness_normalized):.3f}\t{np.min(naturalness_normalized):.3f}')
    results = {'scores_origin': naturalness, 
               'scores_normalized': naturalness_normalized,}
    
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
        
    gt = pd.read_csv(config.ground_truth_csv)
    max_seen = np.max(gt.GFP).item()
    min_seen = np.min(gt.GFP).item()
    normalize = lambda x: (x - min_seen) / (max_seen - min_seen)
    outputs_normalized = [normalize(x) for x in outputs]
    logger.info(f'task\tmean\tmedian\tstd\tmax\tmin')
    logger.info(f'GFP\t{np.mean(outputs):.3f}\t{np.median(outputs):.3f}\t{np.std(outputs):.3f}\t{np.max(outputs):.3f}\t{np.min(outputs):.3f}')
    logger.info(f'GFP_normalized\t{np.mean(outputs_normalized):.3f}\t{np.median(outputs_normalized):.3f}\t{np.std(outputs_normalized):.3f}\t{np.max(outputs_normalized):.3f}\t{np.min(outputs_normalized):.3f}')
    results = {'scores_origin': outputs, 
               'scores_normalized': outputs_normalized,}
    
    return results


def get_hypervolume(naturalness_results, GFP_results, config):
    gt = pd.read_csv(config.ground_truth_csv)
    min_seen_GFP = np.min(gt.GFP).item()
    min_seen_naturalness = np.min(gt.naturalness).item()
    scores_naturalness = naturalness_results['scores_origin']
    scores_GFP = GFP_results['scores_origin']
    inverse_scores_GFP = [-x for x in scores_GFP]
    inverse_min_seen_GFP = -min_seen_GFP
    inverse_scores_naturalness = [-x for x in scores_naturalness]
    inverse_min_seen_naturalness = -min_seen_naturalness
    hv = calc_hypervolume(inverse_scores_naturalness, inverse_scores_GFP, inverse_min_seen_naturalness, inverse_min_seen_GFP)
    
    scores_GFP_normalized = GFP_results['scores_normalized']
    inverse_scores_GFP_normalized = [-x for x in scores_GFP_normalized]
    scores_naturalness_normalized = naturalness_results['scores_normalized']
    inverse_scores_naturalness_normalized = [-x for x in scores_naturalness_normalized]
    hv_normalized = calc_hypervolume(inverse_scores_naturalness_normalized, inverse_scores_GFP_normalized, 1, 0)
    # calculate reference hv
    worst_scores_naturalness = [np.min(gt.naturalness).item()] * len(scores_naturalness)
    worst_scores_GFP = [np.min(gt.GFP).item()] * len(scores_GFP)
    inverse_worst_scores_GFP = [-x for x in worst_scores_GFP]
    inverse_worst_scores_naturalness = [-x for x in worst_scores_naturalness]
    ref_hv = calc_hypervolume(inverse_worst_scores_naturalness, inverse_worst_scores_GFP, inverse_min_seen_naturalness, inverse_min_seen_GFP)
    logger.info(f'ref_hv_worst: {ref_hv}')
    
    best_scores_naturalness = [np.max(gt.naturalness).item()] * len(scores_naturalness)
    best_scores_GFP = [np.max(gt.GFP).item()] * len(scores_GFP)
    inverse_best_scores_GFP = [-x for x in best_scores_GFP]
    inverse_best_scores_naturalness = [-x for x in best_scores_naturalness]
    ref_hv = calc_hypervolume(inverse_best_scores_naturalness, inverse_best_scores_GFP, inverse_min_seen_naturalness, inverse_min_seen_GFP)
    logger.info(f'ref_hv_best: {ref_hv}')
    
    ref_hv = calc_hypervolume([0] * len(scores_naturalness), [-1] * len(scores_GFP), 1, 0)
    logger.info(f'ref_hv_best_norm: {ref_hv}')
    # input()
    return hv, hv_normalized

def evaluate_all(naturalness_results, GFP_results, sampled_seqs, config, args):
    hv, hv_normalized = get_hypervolume(naturalness_results, GFP_results, config)
    logger.info(f'Hypervolume: {hv:.3f}')
    base_pool_df = pd.read_csv(config.base_pool_path)
    base_pool_seqs = base_pool_df.sequence.tolist()
    all_novelty = novelty(sampled_seqs, base_pool_seqs)
    scores_save_path = os.path.join(os.path.dirname(config.sample_path), 'evaluation_results.csv' if args.tag is None else f'evaluation_results_{args.tag}.csv')
    metrics_save_path = os.path.join(os.path.dirname(config.sample_path), 'evaluation_metrics.csv' if args.tag is None else f'evaluation_metrics_{args.tag}.csv')
    scores_df = pd.DataFrame({
        'sequence': sampled_seqs,
        'naturalness': naturalness_results['scores_origin'],
        'GFP': GFP_results['scores_origin'],
        'novelty': all_novelty,
        'naturalness_normalized': naturalness_results['scores_normalized'],
        'GFP_normalized': GFP_results['scores_normalized'],
    })
    scores_df.to_csv(scores_save_path, index=False)
    metrics_df = pd.DataFrame({
        'num_seqs': [len(sampled_seqs), len(sampled_seqs), len(sampled_seqs), len(sampled_seqs)],
        'task': ['naturalness', 'GFP', 'naturalness', 'GFP'],
        'normalized': ['True', 'True', 'False', 'False'],
        'hv': [hv_normalized, hv_normalized, hv, hv],
        'mean_fitness': [np.mean(naturalness_results['scores_normalized']), np.mean(GFP_results['scores_normalized']), np.mean(naturalness_results['scores_origin']), np.mean(GFP_results['scores_origin'])],
        'median_fitness': [np.median(naturalness_results['scores_normalized']), np.median(GFP_results['scores_normalized']), np.median(naturalness_results['scores_origin']), np.median(GFP_results['scores_origin'])],
        'std_fitness': [np.std(naturalness_results['scores_normalized']), np.std(GFP_results['scores_normalized']), np.std(naturalness_results['scores_origin']), np.std(GFP_results['scores_origin'])],
        'max_fitness': [np.max(naturalness_results['scores_normalized']), np.max(GFP_results['scores_normalized']), np.max(naturalness_results['scores_origin']), np.max(GFP_results['scores_origin'])],
        'min_fitness': [np.min(naturalness_results['scores_normalized']), np.min(GFP_results['scores_normalized']), np.min(naturalness_results['scores_origin']), np.min(GFP_results['scores_origin'])],
        'diversity': [diversity(sampled_seqs), diversity(sampled_seqs), diversity(sampled_seqs), diversity(sampled_seqs)],
        'mean_novelty': [np.mean(all_novelty), np.mean(all_novelty), np.mean(all_novelty), np.mean(all_novelty)],
        'median_novelty': [np.median(all_novelty), np.median(all_novelty), np.median(all_novelty), np.median(all_novelty)],
    })
    print(metrics_df)
    metrics_df.to_csv(metrics_save_path, index=False)
    logger.info(f'Saved scores to {scores_save_path}')
    logger.info(f'Saved metrics to {metrics_save_path}')

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
    if args.select_path is not None:
        sampled_seqs = pd.read_csv(args.select_path).sequence.tolist()
    else:
        sampled_seqs = globals()[config.selection_method](config)
    logger.info(f'Selected {len(sampled_seqs)} sequences for evaluation.')
    GFP_results = evaluate_GFP(sampled_seqs, config, args)
    logger.info('Finished evaluating GFP.')
    naturalness_results = evaluate_wt_naturalness(sampled_seqs, config, args)
    logger.info('Finished evaluating naturalness.')
    evaluate_all(naturalness_results, GFP_results, sampled_seqs, config, args)

if __name__ == '__main__':
    main()

