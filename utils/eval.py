import sys
sys.path.append('./utils')
import torch
import numpy as np
from pymoo.indicators.hv import HV
from polyleven import levenshtein
import pandas as pd
import time
import logging
from common import get_logger

logger = get_logger('eval')

# Function to calculate Root Mean Square Error (RMSE)
def calculate_rmse(prediction, ground_truth):
    """
    Calculate the Root Mean Square Error between two 1D tensors.
    
    Parameters:
        prediction (torch.Tensor): 1D tensor containing the predicted values.
        ground_truth (torch.Tensor): 1D tensor containing the ground truth values.
        
    Returns:
        float: Root Mean Square Error (RMSE)
    """
    squared_diffs = (prediction - ground_truth) ** 2
    rmse = torch.sqrt(torch.mean(squared_diffs))
    return rmse

# Function to calculate Mean Absolute Error (MAE)
def calculate_mae(prediction, ground_truth):
    """
    Calculate the Mean Absolute Error between two 1D tensors.
    
    Parameters:
        prediction (torch.Tensor): 1D tensor containing the predicted values.
        ground_truth (torch.Tensor): 1D tensor containing the ground truth values.
        
    Returns:
        float: Mean Absolute Error (MAE)
    """
    mae = torch.mean(torch.abs(prediction - ground_truth))
    return mae

def calc_hypervolume(scores1, scores2, ref_score1, ref_score2):
    '''calculate hypervolume'''
    indicator = HV(ref_point=np.array([ref_score1, ref_score2]))
    hv = indicator(np.array([scores1, scores2]).T)
    
    return hv

def diversity(seqs):
    num_seqs = len(seqs)
    if num_seqs <= 1:
        return 0
    total_dist = 0
    for i in range(num_seqs):
        for j in range(num_seqs):
            x = seqs[i]
            y = seqs[j]
            if x == y:
                continue
            total_dist += levenshtein(x, y)
    return total_dist / (num_seqs*(num_seqs-1))

def novelty(sampled_seqs, base_pool_seqs):
    # sampled_seqs: top k
    # existing_seqs: range dataset
    all_novelty = []
    for src in sampled_seqs:  
        min_dist = 1e9
        for known in base_pool_seqs:
            dist = levenshtein(src, known)
            if dist < min_dist:
                min_dist = dist
        all_novelty.append(min_dist)
        
    return all_novelty

def greedy_selection(csv_path, num_select=100, ref_score_1=None, ref_score_2=None, inverse_sign_1=False, inverse_sign_2=False):
    start = time.time()
    df = pd.read_csv(csv_path)
    seq_column = 'mutant_sequences' if 'mutant_sequences' in df.columns else 'sequence'
    df = df.drop_duplicates(subset=[seq_column])
    all_seqs = df[seq_column].tolist()
    scores_1 = df['mutant_scores_1'].values if not inverse_sign_1 else -df['mutant_scores_1'].values
    scores_2 = df['mutant_scores_2'].values if not inverse_sign_2 else -df['mutant_scores_2'].values
    selected_idxs = []
    n = len(df)
    if ref_score_1 is None:
        ref_score_1 = np.max(scores_1)
    logger.info(f'ref_score_1: {ref_score_1}')
    if ref_score_2 is None:
        ref_score_2 = np.max(scores_2)
    logger.info(f'ref_score_2: {ref_score_2}')
    while len(selected_idxs) < num_select:
        print(f'{len(selected_idxs)}/{num_select}', end='\r', flush=True)
        max_hv = 0
        max_idx = -1
        for i in range(n):
            if i in selected_idxs:
                continue
            selected_scores_1 = scores_1[selected_idxs + [i]]
            selected_scores_2 = scores_2[selected_idxs + [i]]
            hv = calc_hypervolume(selected_scores_1, selected_scores_2, ref_score_1, ref_score_2)
            if hv > max_hv:
                max_hv = hv
                max_idx = i
        if 0 <= max_idx < n:
            selected_idxs.append(max_idx)
    end = time.time()
    logger.info(f'max_hv: {max_hv}')
    logger.info(f'reference_hv: {calc_hypervolume([scores_1.min()] * num_select, [scores_2.min()] * num_select, ref_score_1, ref_score_2)}')
    logger.info(f'greedy selection done! Selected {len(selected_idxs)} sequences. Time elapsed: {end-start:.2f} seconds.')
    selected_sequences = [all_seqs[i] for i in selected_idxs]
    
    return selected_sequences
    
def pareto_selection(csv_path, num_select=100, inverse_sign_1=False, inverse_sign_2=False):
    '''by default, the objectives should be maximized'''
    start = time.time()
    df = pd.read_csv(csv_path)
    seq_column = 'mutant_sequences' if 'mutant_sequences' in df.columns else 'sequence'
    df = df.drop_duplicates(subset=[seq_column])
    all_seqs = df[seq_column].tolist()
    scores_1 = df['mutant_scores_1'].values if not inverse_sign_1 else -df['mutant_scores_1'].values
    scores_2 = df['mutant_scores_2'].values if not inverse_sign_2 else -df['mutant_scores_2'].values
    selected_idxs = []
    n = len(df)
    for i in range(n):
        if i in selected_idxs:
            continue
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if scores_1[i] <= scores_1[j] and scores_2[i] <= scores_2[j] and (scores_1[i] < scores_1[j] or scores_2[i] < scores_2[j]):
                dominated = True
                break
        if not dominated:
            selected_idxs.append(i)
    selected_seqs = [all_seqs[i] for i in selected_idxs]
    end = time.time()
    logger.info(f'pareto selection done! Selected {len(selected_seqs)} sequences. Time elapsed: {end-start:.2f} seconds.')
    
    return selected_seqs

def get_pareto_front(csv_path, label_1, label_2, inverse_sign_1=False, inverse_sign_2=False):
    '''by default, the objectives should be maximized'''
    start = time.time()
    df = pd.read_csv(csv_path)
    seq_column = 'mutant_sequences' if 'mutant_sequences' in df.columns else 'sequence'
    df = df.drop_duplicates(subset=[seq_column])
    all_seqs = df[seq_column].tolist()
    scores_1 = df[label_1].values if not inverse_sign_1 else -df[label_1].values
    scores_2 = df[label_2].values if not inverse_sign_2 else -df[label_2].values
    selected_idxs = []
    n = len(df)
    for i in range(n):
        if i in selected_idxs:
            continue
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if scores_1[i] <= scores_1[j] and scores_2[i] <= scores_2[j] and (scores_1[i] < scores_1[j] or scores_2[i] < scores_2[j]):
                dominated = True
                break
        if not dominated:
            selected_idxs.append(i)
    selected_seqs = [all_seqs[i] for i in selected_idxs]
    selected_scores_1 = scores_1[selected_idxs]
    selected_scores_2 = scores_2[selected_idxs]
    end = time.time()
    logger.info(f'Got pareto front! Selected {len(selected_seqs)} sequences. Time elapsed: {end-start:.2f} seconds.')
    
    return selected_seqs, selected_scores_1, selected_scores_2

def non_dominant_sorting(seqs, scores_1, scores_2):
    '''the optimization direction is maximization'''
    n = len(seqs)
    

def non_dominant_selection(csv_path, obj_1, obj_2, inverse_sign_1, inverse_sign_2, num_select):
    df = pd.read_csv(csv_path)
    seq_column = 'mutant_sequences' if 'mutant_sequences' in df.columns else 'sequence'
    seqs = df[seq_column].tolist()
    scores_1 = df[obj_1].values if not inverse_sign_1 else -df[obj_1].values
    scores_1 = scores_1.tolist()
    scores_2 = df[obj_2].values if not inverse_sign_2 else -df[obj_2].values
    scores_2 = scores_2.tolist()

if __name__ == '__main__':
    # a = torch.arange(10, dtype=torch.float32).reshape(2,5)
    # b = torch.arange(10, dtype=torch.float32).reshape(2,5)
    # a = torch.randn((512,2))
    # b = torch.randn((512,2))
    # acc = calculate_rmse(a, b)
    # logger.info(acc)
    selected_seqs = greedy_selection('../logs_new/GWG_2_gb1_ddg_pref_vec_2023_10_11__11_15_06_pref_index_0/samples_20231011-111506/seed_1.csv', 10, ref_score_1=None, ref_score_2=None, inverse_sign_1=False, inverse_sign_2=True)
    print(len(selected_seqs))
    print(selected_seqs[0])