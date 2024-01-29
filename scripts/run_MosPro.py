import sys
sys.path.append('.')
from typing import List, Optional, Tuple

import copy
import time
import os
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pandas as pd
import torch
# from models.GWG_module import GwgPairSampler
from MosPro.samplers import MosProSampler
from MosPro.sequence_dataset import PreScoredSequenceDataset
import argparse, shutil, yaml
from utils import common
import datetime

to_list = lambda x: x.cpu().detach().numpy().tolist()

def _worker_fn(args, logger):
    """Worker function for multiprocessing.

    Args:
        args (tuple): (worker_i, exp_cfg, inputs)
            worker_i: worker id. Used for setting CUDA device.
            exp_cfg: model config.
            inputs: list of inputs to process.

    Returns:
        all_outputs: results of GWG.
    """
    worker_i, config, inputs = args
    model = MosProSampler(config=config,
            predictor_1_dir=config.predictor_1_dir,
            predictor_2_dir=config.predictor_2_dir,
            temperature=config.temperature,
            ckpt_name=config.ckpt_name,
            verbose=config.verbose,
            gibbs_samples=config.gibbs_samples,
            device=config.device,
            inverse_sign_1=config.inverse_sign_1,
            inverse_sign_2=config.inverse_sign_2,
            gradient_compose_method=config.gradient_compose_method,
            balance_weight_1=config.balance_weight_1,
            balance_weight_2=config.balance_weight_2,
            lambda_=config.lambd,
            mutation_sites=config.mutation_sites,
            lambda_method=config.lambda_method,
            weight_1=config.weight_1,
            weight_2=config.weight_2,
            pref_index=config.pref_index
            )
    all_outputs = []
    for batch in inputs:
        all_outputs.append(model(batch))
    logger.info(f'Done with worker: {worker_i}')
    return all_outputs

def _setup_dataset(cfg):
    # if cfg.data.csv_path is not None:
    #     raise ValueError(f'cfg.data.csv_file must be None.')
    # data_dir = os.path.dirname(cfg.experiment.predictor_dir).replace('ckpt', 'data')
    # if '_custom' in data_dir:
    #     data_dir = data_dir.replace('_custom', '')
    # for i, subdir in enumerate(data_dir.split('/')):
    #     if 'percentile' in subdir:
    #         break
    # data_dir = '/'.join(data_dir.split('/')[:i+1])
    # cfg.data.csv_path = os.path.join(data_dir, 'base_seqs.csv')
    # if not os.path.exists(cfg.data.csv_path):
    #     raise ValueError(f'Could not find dataset at {cfg.data.csv_path}.')

    return PreScoredSequenceDataset(cfg.csv_path, cfg.cluster_cutoff, cfg.max_visits, cfg.task, cfg)

def generate_pairs(cfg: DictConfig, sample_write_path: str, logger) -> Tuple[dict, dict]:
    """Generate pairs using GWG."""
    cfg = copy.deepcopy(cfg)
    # run_cfg = cfg.run

    # set seed for random number generators in pytorch, numpy and python.random
    common.seed_all(cfg.seed)

    dataset = _setup_dataset(cfg)
    logger.info('Set up dataset.')

    # Special settings for debugging.
    exp_cfg = cfg
    epoch = 0
    start_time = time.time()
    while epoch < cfg.max_epochs and len(dataset):
        epoch += 1
        epoch_start_time = time.time()
        if cfg.debug:
            batch_size = 2
        else:
            batch_size = len(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # Run sampling with workers
        batches_per_worker = [[]]
        for i, batch in enumerate(dataloader):
            batches_per_worker[i].append(batch)
            if cfg.debug:
                break
        logger.info(f"using GPU: {torch.device(cfg.device)}" )
        all_worker_outputs = [
            _worker_fn((0, exp_cfg, batches_per_worker[0]), logger)
        ]

        # Process results.
        epoch_pair_count = 0
        candidate_seqs = []
        for worker_results in all_worker_outputs:
            for new_pairs in worker_results:
                if new_pairs is None:
                    continue
                candidate_seqs.append(
                    new_pairs[['mutant_sequences', 'mutant_scores_1', 'mutant_scores_2']].rename(
                        columns={'mutant_sequences': 'sequences', 'mutant_scores_1': 'scores', 'mutant_scores_2': 'scores_2'}
                    )
                )
                epoch_pair_count += dataset.add_pairs(new_pairs, epoch)
        if len(candidate_seqs) > 0:
            candidate_seqs = pd.concat(candidate_seqs)
            candidate_seqs.drop_duplicates(subset='sequences', inplace=True)
        epoch_elapsed_time = time.time() - epoch_start_time

        logger.info(f"Epoch {epoch} finished in {epoch_elapsed_time:.2f} seconds")
        logger.info("------------------------------------")
        logger.info(f"Generated {epoch_pair_count} pairs in this epoch")
        dataset.reset()
        # print(candidate_seqs, len(candidate_seqs))
        if epoch < cfg.max_epochs and len(candidate_seqs) > 1:
            dataset.add(candidate_seqs)
            dataset.cluster()
        logger.info(f"Next dataset = {len(dataset)} sequences")
    dataset.pairs.to_csv(sample_write_path, index=False)
    elapsed_time = time.time() - start_time
    logger.info(f'Finished generation in {elapsed_time:.2f} seconds.')
    logger.info(f'Samples written to {sample_write_path}.')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='./configs/train.yml')
    parser.add_argument('--logdir', type=str, default='./logs_mgda_linear_1024')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gen', action='store_true')
    parser.add_argument('--lambd', type=float, default=None)
    parser.add_argument('--pref_index', type=int, default=None)
    parser.add_argument('--linear_weight_1', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--normalize_grad', action='store_true')
    args = parser.parse_args()
    
    return args

def main():
    start_overall = time.time()
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args = get_args()
    # Load configs
    config = common.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    common.seed_all(config.seed if args.seed is None else args.seed)
    config.lambd = args.lambd if args.lambd is not None else config.lambd
    config.pref_index = args.pref_index if args.pref_index is not None else config.pref_index
    config.device = args.device
    config.linear_weight_1 = args.linear_weight_1 if args.linear_weight_1 is not None else config.linear_weight_1
    config.temperature = config.temperature if args.temperature is None else args.temperature
    config.normalize_grad = args.normalize_grad
    
    # Logging
    log_dir = common.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    logger = common.get_logger('GWG', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    
    logger.info(f'predictor_1_dir: {config.predictor_1_dir}')
    logger.info(f'predictor_2_dir: {config.predictor_2_dir}')
    
    # Set-up output path
    output_dir = os.path.join(log_dir, 'samples_' + now)
    cfg_write_path = os.path.join(output_dir, 'config.yaml')
    os.makedirs(os.path.dirname(cfg_write_path), exist_ok=True)
    with open(cfg_write_path, 'w') as f:
        yaml.dump(dict(config), f)
    logger.info(f'Config saved to {cfg_write_path}')
    
    # Generate samples for multiple seeds.
    seed = config.seed if args.seed is None else args.seed
    sample_write_path = os.path.join(output_dir, f'seed_{seed}.csv')
    logger.info(f'On seed {seed}. Saving results to {sample_write_path}')
    logger.info(f'Inverse sign 1: {config.inverse_sign_1}')
    logger.info(f'Inverse sign 2: {config.inverse_sign_2}')
    generate_pairs(config, sample_write_path, logger)

if __name__ == "__main__":
    main()
