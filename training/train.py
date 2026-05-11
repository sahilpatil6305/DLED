"""
Training entry point for DLED (Dual-Level Evidence Detector).

This script trains the `dled` detector with the `Trainer` class.
Other detectors / trainers from the original research codebase have been
removed for this release.
"""
import os
import argparse
import datetime
import random
import yaml
import numpy as np
from datetime import timedelta

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import DeepfakeFrequencyDataset
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter


def get_args():
    parser = argparse.ArgumentParser(description='DLED training script.')
    parser.add_argument('--detector_path', type=str,
                        default='./training/config/detector/dled.yaml',
                        help='Path to detector YAML config.')
    parser.add_argument('--train_config', type=str,
                        default='./training/config/train_config.yaml',
                        help='Path to the training YAML config.')
    parser.add_argument('--train_dataset', nargs='+', required=True,
                        help='Training dataset name(s), e.g. "DF40_FF_train_NOEFS".')
    parser.add_argument('--test_dataset', nargs='+', required=True,
                        help='Testing dataset name(s).')
    parser.add_argument('--max_samples_per_class', type=int, default=None,
                        help='Maximum number of samples per class.')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of training classes.')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs.')
    parser.add_argument('--test_fre', type=int, default=50,
                        help='Test frequency (in epochs).')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for single-GPU training.')
    parser.add_argument('--device_choice', choices=['old', 'new'], default='old',
                        help='`old`: use local_rank arg; `new`: use $LOCAL_RANK env.')
    parser.add_argument('--ddp', action='store_true', default=False,
                        help='Enable distributed training.')
    parser.add_argument('--training_mode', type=str, default='multi',
                        choices=['single', 'multi'])
    parser.add_argument('--training_data_ratio', type=float, default=1.0)
    parser.add_argument('--task_target', type=str, default='test')
    parser.add_argument('--save_path', type=str, default='./saved_models')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--show_training_process', dest='track_his',
                        action='store_true', default=False)
    parser.add_argument('--avu_reg', action='store_true', default=False,
                        help='Enable AvU regularization (required for DLED).')
    parser.add_argument('--kl_reg', action='store_true', default=False)
    parser.add_argument('--detect_novel', action='store_true', default=False)
    return parser.parse_args()


def init_seed(config):
    seed = config.get('manualSeed') or 123456
    config['manualSeed'] = seed
    random.seed(seed)
    np.random.seed(seed)
    if config['cuda']:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_training_data(config):
    train_set = DeepfakeFrequencyDataset(config=config, mode='train')
    if config['ddp']:
        sampler = DistributedSampler(train_set)
        return torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            num_workers=int(config['workers']),
            collate_fn=train_set.collate_fn,
            sampler=sampler,
        )
    return torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config['train_batchSize'],
        shuffle=True,
        num_workers=int(config['workers']),
        collate_fn=train_set.collate_fn,
    )


def prepare_testing_data(config):
    test_data_loaders = {}
    for test_name in config['test_dataset']:
        cfg = config.copy()
        cfg['test_dataset'] = test_name
        test_set = DeepfakeFrequencyDataset(config=cfg, mode='test')
        test_data_loaders[test_name] = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config['test_batchSize'],
            shuffle=False,
            num_workers=int(config['workers']),
            collate_fn=test_set.collate_fn,
            drop_last=(test_name == 'DeepFakeDetection'),
        )
    return test_data_loaders


def choose_metric(config):
    scoring = config['metric_scoring']
    if scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError(f'metric {scoring} is not implemented')
    return scoring


def main():
    args = get_args()

    # Load detector + training configs
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.train_config, 'r') as f:
        train_cfg = yaml.safe_load(f)

    # detector-level label_dict overrides training config
    if 'label_dict' in config:
        train_cfg['label_dict'] = config['label_dict']
    config.update(train_cfg)

    # Device setup
    if args.device_choice == 'old':
        torch.cuda.set_device(args.local_rank)
        config['local_rank'] = args.local_rank
    else:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        config['local_rank'] = local_rank

    # Merge CLI args into config
    config['track_his'] = args.track_his
    config['kl_reg'] = args.kl_reg
    config['avu_reg'] = args.avu_reg
    config['save_model'] = args.save_model
    config['device_choice'] = args.device_choice
    config['save_path'] = args.save_path
    config['train_dataset'] = args.train_dataset
    config['test_dataset'] = args.test_dataset
    config['training_data_ratio'] = args.training_data_ratio
    config['max_samples_per_class'] = args.max_samples_per_class
    config['detect_novel'] = args.detect_novel
    config['num_classes'] = args.num_classes
    config.setdefault('backbone_config', {})
    config['backbone_config']['num_classes'] = args.num_classes
    config['training_mode'] = args.training_mode
    config['ddp'] = args.ddp
    config['test_fre'] = args.test_fre
    config['nEpochs'] = args.num_epochs
    config['task_target'] = args.task_target
    if config.get('dry_run'):
        config['nEpochs'] = 0

    # Logger
    timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = f'_{args.task_target}' if args.task_target else ''
    logger_path = os.path.join(config['log_dir'],
                               config['model_name'] + task_str + '_' + timenow)
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info(f'Save log to {logger_path}')

    logger.info('--------------- Configuration ---------------')
    for key, value in config.items():
        logger.info(f'{key}: {value}')

    if not args.avu_reg:
        raise ValueError('DLED requires --avu_reg.')
    logger.info('AvU regularization: enabled')
    logger.info('KL regularization: ' + ('enabled' if args.kl_reg else 'disabled'))
    logger.info('Save model: ' + ('yes' if args.save_model else 'no'))

    # Reproducibility / cuDNN
    init_seed(config)
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=120))
        logger.addFilter(RankFilter(0))

    # Data
    train_data_loader = prepare_training_data(config)
    test_data_loaders = prepare_testing_data(config)

    # Model
    model_class = DETECTOR[config['model_name']]
    if config['model_name'] != 'dled':
        raise ValueError(
            f"This release only supports model_name=dled, got '{config['model_name']}'.")
    model = model_class(config)
    metric_scoring = choose_metric(config)

    # Trainer
    trainer = Trainer(config, model, optimizer=None, scheduler=None,
                      logger=logger, metric_scoring=metric_scoring)

    best_metric = None
    for epoch in range(config['start_epoch'], config['nEpochs']):
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
            epoch=epoch,
            train_data_loader=train_data_loader,
            test_data_loaders=test_data_loaders,
            test_epoch_freq=args.test_fre,
        )
        if best_metric is not None:
            logger.info(
                f"===> Epoch[{epoch}] end with testing {metric_scoring}: "
                f"{parse_metric_for_print(best_metric)}!"
            )

    if best_metric is not None:
        logger.info(
            f'Stop Training on best Testing metric {parse_metric_for_print(best_metric)}'
        )

    if args.track_his:
        logger.info('Loss for training: %s', trainer.train_loss_dict)
        for dataset_name in config['test_dataset']:
            logger.info('Dataset %s', dataset_name)
            logger.info('Loss for testing: %s', trainer.test_loss_dict[dataset_name])

    logger.info('Uncertainty threshold: %s', trainer.uncertainty_dict)
    for dataset_name in trainer.spatial_acc_dict.keys():
        logger.info('Dataset %s', dataset_name)
        logger.info('spatial_Accuracy: %s', trainer.spatial_acc_dict[dataset_name])
        logger.info('spatial_Binary Accuracy: %s', trainer.spatial_binary_acc_dict[dataset_name])
        logger.info('frequency_Accuracy: %s', trainer.frequency_acc_dict[dataset_name])
        logger.info('frequency_Binary Accuracy: %s', trainer.frequency_binary_acc_dict[dataset_name])
        logger.info('fused_Accuracy: %s', trainer.fused_acc_dict[dataset_name])
        logger.info('fused_Binary Accuracy: %s', trainer.fused_binary_acc_dict[dataset_name])

    if config['save_model']:
        logger.info('===> Save Model')
        save_dir = config['save_path']
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = (
            f"{config['model_name']}_tf{config['test_fre']}_ep{config['nEpochs']}_"
            f"{timestamp}"
        )
        file_path = os.path.join(
            save_dir,
            f"{base_name}_c{config['num_classes']}_{config['train_dataset'][0]}.pth",
        )
        print('file_path: ', file_path)
        torch.save(model.state_dict(), file_path)

    for writer in trainer.writers.values():
        writer.close()


if __name__ == '__main__':
    main()
