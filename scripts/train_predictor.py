import sys
sys.path.append('.')
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Subset
# from torch.utils.tensorboard import SummaryWriter

from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef
import os, argparse, time, shutil
from tqdm.auto import tqdm

from MosPro.utils import common
from MosPro.utils.eval import calculate_rmse, calculate_mae

from MosPro.datasets.fitness_dataset import FitnessDataset
from MosPro.models.predictors import BaseCNN

torch.set_num_threads(1)

def evaluate(model, val_loader, criterion, device, config, logger):
    model.eval()
    all_loss = []
    all_output = []
    all_pearson_metric, all_spearman_metric = [], []
    all_label = []
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            # print(criterion)
            loss = criterion(output, label).squeeze(-1)
            all_loss.append(common.toCPU(loss).item())
            all_output.append(common.toCPU(output))
            all_label.append(common.toCPU(label))
            # all_pearson_metric.append(pearson_corrcoef(common.toCPU(output), common.toCPU(label)).item())
            # all_spearman_metric.append(spearman_corrcoef(common.toCPU(output), common.toCPU(label)).item())
        # print(all_loss, all_loss[0])
        all_loss = torch.tensor(all_loss)
    all_output = torch.cat(all_output, dim=0)
    all_label = torch.cat(all_label, dim=0)
    pearson = pearson_corrcoef(all_output, all_label).item()
    spearman = spearman_corrcoef(all_output, all_label).item()
    rmse = calculate_rmse(all_output, all_label).item()
    mae = calculate_mae(all_output, all_label).item()
    
    # pearson = np.mean(all_pearson_metric)
    # spearman = np.mean(all_spearman_metric)
    
    model.train()
    
    return all_loss.mean().item(), pearson, spearman, rmse, mae

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config, logger):
    model.train()
    n_bad = 0
    best_val_loss = 1.e10
    best_pearson = 0
    best_spearman = 0
    all_loss = []
    all_val_loss = []
    epsilon = 1e-4
    for epoch in range(config.train.num_epochs):
        # input()
        start = time.time()
        val_loss, val_pearson, val_spearman, val_rmse, val_mae = evaluate(model, val_loader, criterion, device, config, logger)
        all_val_loss.append(val_loss)
        end_test = time.time()
        # if val_pearson < best_pearson + epsilon and val_spearman < best_spearman + epsilon:
        if val_loss > best_val_loss - epsilon:
            n_bad += 1
            if n_bad >= config.train.early_stop:
                logger.info(f'No performance improvement for {config.train.early_stop} epochs. Early stop training!')
                break
        else:
            logger.info(f'New best performance found! val_loss={val_loss:.4f}; pearson={val_pearson:.4f}; spearman={val_spearman:.4f}')
            n_bad = 0
            best_val_loss = val_loss
            best_pearson = val_pearson if val_pearson > best_pearson + epsilon else best_pearson
            best_spearman = val_spearman if val_spearman > best_spearman + epsilon else best_spearman
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(config.train.ckpt_dir, 'best_checkpoints.pt'))
        losses = []
        for data, label in tqdm(train_loader, dynamic_ncols=True, desc='training'):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            losses.append(common.toCPU(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(sum(losses) / len(losses))
        all_loss.append(sum(losses) / len(losses))
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(config.train.ckpt_dir, 'last_checkpoints.pt'))
        end_epoch = time.time()
        logger.info(f'Epoch [{epoch + 1}/{config.train.num_epochs}]: loss: {sum(losses) / len(losses):.4f}; val_loss: {val_loss:.4f}; pearson: {val_pearson:.4f}; spearman: {val_spearman:.4f}; rmse: {val_rmse:.4f}; mae: {val_mae:.4f}; train time: {common.sec2min_sec(end_epoch - end_test)}')
        # writer.add_scalar('train/loss', sum(losses) / len(losses), epoch)
        # writer.add_scalar('val/loss', val_loss, epoch)
        # writer.add_scalar('val/pearson', val_pearson, epoch)
        # writer.add_scalar('val/spearman', val_spearman, epoch)
        # writer.add_scalar('val/rmse', val_rmse, epoch)
        # writer.add_scalar('val/mae', val_mae, epoch)
    
    return all_loss, all_val_loss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='./configs/train.yml')
    parser.add_argument('--logdir', type=str, default='./logs_predictor')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()
    
    return args

def main():
    start_overall = time.time()
    args = get_args()
    # Load configs
    config = common.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    common.seed_all(config.train.seed if args.seed is None else args.seed)

    # Logging
    log_dir = common.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    config.train.ckpt_dir = ckpt_dir
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = common.get_logger('train', log_dir)
    # writer = SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./MosPro', os.path.join(log_dir, 'models'))
    
    # datasets
    all_data = FitnessDataset(config.data.data_file, config.data.label_column, config.data)
    if config.data.augment_negative:
        all_data.augment_with_negative_sample(task=config.task)
    
    if config.data.split_val_by_mutation_sites:
        pass
    else:
        all_indices = common.get_random_indices(len(all_data), config.train.seed)
        train_indices = all_indices[:int(0.9 * len(all_indices))]
        val_indices = all_indices[int(0.9 * len(all_indices)):]
    train_data = Subset(all_data, train_indices)
    val_data = Subset(all_data, val_indices)
    train_loader = DataLoader(train_data, batch_size=config.train.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=config.train.batch_size)
    logger.info(f'All data: {len(all_indices)}; Train data: {len(train_data)}; Val data: {len(val_data)}')
    
    # Load model and set hyperparameters
    config.model.device = args.device
    model = globals()[config.model.model_type](config.model)
    model.to(args.device)
    if 'ESM' in config.model.model_type:
        model.freeze_esm()
    logger.info(model)
    logger.info(f'Model trainable parameters: {common.count_parameters(model)}')
    
    # train
    criterion = globals()[config.train.loss]()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr, weight_decay=config.train.wd, betas=[0.9, 0.99])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, verbose=True)
    
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, args.device, config, logger)
    
    end_overall = time.time()
    logger.info(f'Total training time: {common.sec2min_sec(end_overall - start_overall)}')
    
if __name__ == '__main__':
    main()