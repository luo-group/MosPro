import sys
sys.path.append('.')
import torch
import os
from utils import common
import pandas as pd
from datasets.fitness_dataset import seq2indices
from models import BaseCNN

def load_oracle(config, args, logger):
    for root, dirs, files in os.walk(config.oracle_dir):
        for file in files:
            if file.endswith('.yml'):
                oracle_config = common.load_config(os.path.join(root, file))
                break
    oracle = globals()[oracle_config.model.model_type](oracle_config.model)
    ckpt = torch.load(os.path.join(config.oracle_dir, 'checkpoints/best_checkpoints.pt'))
    oracle.load_state_dict(ckpt)
    oracle.eval()
    oracle.to(args.device)
    logger.info(f'Loaded oracle from {config.oracle_dir}')
    
    return oracle

def run_predictor(predictor, x, args, no_grad=False):
    batch_size = 128

    x_batches = torch.split(x, batch_size, 0)
    outputs = []
    if no_grad:
        with torch.no_grad():
            for x_batch in x_batches:
                x_batch = x_batch.to(args.device)
                outputs.append(predictor(x_batch))
    else:
        for x_batch in x_batches:
            x_batch = x_batch.to(args.device)
            outputs.append(predictor(x_batch))
    outputs = torch.cat(outputs)
    
    return outputs

def load_seqs_from_csv(csv_path, seq_column='sequence', topk=None, label_column=None):
    df = pd.read_csv(csv_path)
    if topk is None or label_column is None:
        seqs = df[seq_column].tolist()
        seqs = [seq2indices(seq) for seq in seqs]
        seqs = torch.stack(seqs)
        print(f'Loaded {len(seqs)} sequences from {csv_path}')
    else:
        df = df.sort_values(by=label_column, ascending=False)
        df = df.head(topk)
        seqs = df[seq_column].tolist()
        seqs = [seq2indices(seq) for seq in seqs]
        seqs = torch.stack(seqs)
        print(f'Loaded top {len(seqs)} sequences by label {label_column} from {csv_path}')
    
    return seqs

def n_hops(population, wt):
    K, L, V = population.shape
    nhops = []
    for k in range(K):
        prot = population[k]
        diff = ((prot - wt) > 0).float()
        nhops += [diff.sum()]
    return torch.mean(torch.stack(nhops)), torch.std(torch.stack(nhops))