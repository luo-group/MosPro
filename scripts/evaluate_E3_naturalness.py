import sys
sys.path.append('.')
import os, argparse, time, datetime, json, random
from utils import common
import pandas as pd
from utils.eval import calc_hypervolume, diversity, novelty, greedy_selection
import numpy as np

now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logger = common.get_logger(__name__)


def load_all_seqs(csv_path):
    df = pd.read_csv(csv_path)
    seq_column = 'mutant_sequences' if 'mutant_sequences' in df.columns else 'sequence'
    df = df.drop_duplicates(subset=[seq_column])
    
    return df[seq_column].tolist()
    

def load_ground_truth(gt_csv_path):
    df = pd.read_csv(gt_csv_path)
    seqs = df['sequence'].tolist()
    fit_E3 = df['fit_E3'].tolist()
    naturalness = df['naturalness'].tolist()
    assert len(seqs) == len(fit_E3) == len(naturalness), f'Lengths of sequences, fit_E3, and naturalness are not equal: {len(seqs)}, {len(fit_E3)}, {len(naturalness)}'
    n = len(seqs)
    seq2label = {seq: {'fit_E3': fit_E3[i], 'naturalness': naturalness[i]} for i, seq in enumerate(seqs)}
    logger.info(f'Loaded {len(seq2label)} ground truth sequences with labels')
    
    return seq2label

def get_metrics(data_list):
    mean_, median_, std_, min_, max_ = np.mean(data_list), np.median(data_list), np.std(data_list), np.min(data_list), np.max(data_list)
    
    return mean_, median_, std_, min_, max_

def evaluate(sampled_seqs, gt_seq2label, config, save_dir=None, tag=None):
    seqs, fit_E3, naturalness = [], [], []
    for seq in sampled_seqs:
        if seq in gt_seq2label:
            seqs.append(seq)
            fit_E3.append(gt_seq2label[seq]['fit_E3'])
            # if fit_E3[-1] > 0.9:
            #     print(seq, gt_seq2label[seq]['fit_E3'])
            #     input()
            naturalness.append(gt_seq2label[seq]['naturalness'])
    assert len(seqs) == len(fit_E3) == len(naturalness), f'Lengths of sequences, fit_E3, and naturalness are not equal: {len(seqs)}, {len(fit_E3)}, {len(naturalness)}'
    results_df = pd.DataFrame({'sequence': seqs, 'fit_E3': fit_E3, 'naturalness': naturalness})
    tag = '' if tag is None else f'_{tag}'
    results_df.to_csv(os.path.join(save_dir, f'evaluation_results{tag}.csv'), index=False)
    gt_fit_E3 = [gt_seq2label[seq]['fit_E3'] for seq in seqs]
    gt_naturalness = [gt_seq2label[seq]['naturalness'] for seq in seqs]
    min_fit_E3, max_fit_E3 = np.min(gt_fit_E3), np.max(gt_fit_E3)
    normalize_fit_E3 = lambda x: (x - min_fit_E3) / (max_fit_E3 - min_fit_E3)
    min_naturalness, max_naturalness = np.min(gt_naturalness), np.max(gt_naturalness)
    normalize_naturalness = lambda x: (x - min_naturalness) / (max_naturalness - min_naturalness)
    fit_E3 = np.array(fit_E3)
    naturalness = np.array(naturalness)
    fit_E3_normalized = np.array([normalize_fit_E3(fit_E3[i]) for i in range(len(fit_E3))])
    naturalness_normalized = np.array([normalize_naturalness(naturalness[i]) for i in range(len(naturalness))])
    fit_E3_mean, fit_E3_median, fit_E3_std, fit_E3_min, fit_E3_max = get_metrics(fit_E3)
    naturalness_mean, naturalness_median, naturalness_std, naturalness_min, naturalness_max = get_metrics(naturalness)
    hv = calc_hypervolume(-fit_E3, -naturalness, -config.ref_score_1, -config.ref_score_2)
    hv_normalized = calc_hypervolume(-fit_E3_normalized, -naturalness_normalized, 0, 0)
    fit_E3_normalized_mean, fit_E3_normalized_median, fit_E3_normalized_std, fit_E3_normalized_min, fit_E3_normalized_max = get_metrics(fit_E3_normalized)
    naturalness_normalized_mean, naturalness_normalized_median, naturalness_normalized_std, naturalness_normalized_min, naturalness_normalized_max = get_metrics(naturalness_normalized)
    logger.info(f'hv: {hv:.4f}, hv_normalized: {hv_normalized:.4f}')
    logger.info(f'fit_E3: mean={fit_E3_mean:.4f}, median={fit_E3_median:.4f}, std={fit_E3_std:.4f}, min={fit_E3_min:.4f}, max={fit_E3_max:.4f}')
    logger.info(f'naturalness: mean={naturalness_mean:.4f}, median={naturalness_median:.4f}, std={naturalness_std:.4f}, min={naturalness_min:.4f}, max={naturalness_max:.4f}')
    logger.info(f'fit_E3_normalized: mean={fit_E3_normalized_mean:.4f}, median={fit_E3_normalized_median:.4f}, std={fit_E3_normalized_std:.4f}, min={fit_E3_normalized_min:.4f}, max={fit_E3_normalized_max:.4f}')
    logger.info(f'naturalness_normalized: mean={naturalness_normalized_mean:.4f}, median={naturalness_normalized_median:.4f}, std={naturalness_normalized_std:.4f}, min={naturalness_normalized_min:.4f}, max={naturalness_normalized_max:.4f}')
    metrics_df = pd.DataFrame({'num_seqs': [len(seqs), len(seqs), len(seqs), len(seqs)], # 'num_seqs' is a dummy column for 'metric
                               'metric': ['fit_E3', 'naturalness', 'fit_E3_normalized', 'naturalness_normalized'],
                               'hv': [hv, hv, hv_normalized, hv_normalized],
                               'mean': [fit_E3_mean, naturalness_mean, fit_E3_normalized_mean, naturalness_normalized_mean],
                               'median': [fit_E3_median, naturalness_median, fit_E3_normalized_median, naturalness_normalized_median],
                               'std': [fit_E3_std, naturalness_std, fit_E3_normalized_std, naturalness_normalized_std],
                               'min': [fit_E3_min, naturalness_min, fit_E3_normalized_min, naturalness_normalized_min],
                               'max': [fit_E3_max, naturalness_max, fit_E3_normalized_max, naturalness_normalized_max]})
    
    metrics_df.to_csv(os.path.join(save_dir, f'evaluation_metrics{tag}.csv'), index=False)

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs/E3_E2/evaluate.yml')
    # parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--sample_path', type=str, default=None)
    parser.add_argument('--select_path', type=str, default=None)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    config = common.load_config(args.config)
    if args.sample_path is not None:
        config.sample_path = args.sample_path
    logger.info(f'Evaluating {config.sample_path}...')
    if args.select_path is not None:
        sampled_seqs = pd.read_csv(args.select_path).sequence.tolist()
    else:
        sampled_seqs = globals()[config.selection_method](config)
    logger.info(f'Selected {len(sampled_seqs)} sequences for evaluation.')
    logger.info(f'Loaded {len(sampled_seqs)} sequences')
    gt_seq2label = load_ground_truth(config.gt_csv_path)
    evaluate(sampled_seqs, gt_seq2label, config, save_dir=os.path.dirname(config.sample_path), tag=args.tag)
    
    
    