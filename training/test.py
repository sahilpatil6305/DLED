"""
Evaluation script for DLED (Dual-Level Evidence Detector).

Loads a trained `dled` checkpoint and reports spatial / frequency /
fused accuracy on the requested test datasets.
"""
import os
import argparse
import datetime
import random
import yaml
from tqdm import tqdm
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from detectors import DETECTOR
from dataset import DeepfakeFrequencyDataset
from logger import create_logger


def get_args():
    parser = argparse.ArgumentParser(description='DLED evaluation script.')
    parser.add_argument('--detector_path', type=str,
                        default='./training/config/detector/dled.yaml',
                        help='Path to detector YAML config.')
    parser.add_argument('--train_config', type=str,
                        default='./training/config/train_config.yaml',
                        help='Path to the training YAML config.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained checkpoint (.pth) to evaluate.')
    parser.add_argument('--train_dataset', nargs='+', required=True,
                        help='Train dataset name (used only to recompute the uncertainty threshold).')
    parser.add_argument('--test_dataset', nargs='+', required=True)
    parser.add_argument('--max_samples_per_class', type=int, default=None)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device_choice', choices=['old', 'new'], default='old')
    parser.add_argument('--training_mode', type=str, default='multi',
                        choices=['single', 'multi'])
    parser.add_argument('--training_data_ratio', type=float, default=1.0)
    parser.add_argument('--task_target', type=str, default='test')
    parser.add_argument('--detect_novel', action='store_true', default=False)
    parser.add_argument('--avu_reg', action='store_true', default=True)
    parser.add_argument('--kl_reg', action='store_true', default=False)
    parser.add_argument('--ddp', action='store_true', default=False)
    return parser.parse_args()


def init_seed(config):
    seed = config.get('manualSeed') or 123456
    random.seed(seed)
    np.random.seed(seed)
    if config['cuda']:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_train_loader(config):
    train_set = DeepfakeFrequencyDataset(config=config, mode='train')
    return torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config['train_batchSize'],
        shuffle=True,
        num_workers=int(config['workers']),
        collate_fn=train_set.collate_fn,
    )


def get_test_loaders(config):
    loaders = {}
    for test_name in config['test_dataset']:
        cfg = config.copy()
        cfg['test_dataset'] = test_name
        test_set = DeepfakeFrequencyDataset(config=cfg, mode='test')
        loaders[test_name] = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config['test_batchSize'],
            shuffle=False,
            num_workers=int(config['workers']),
            collate_fn=test_set.collate_fn,
            drop_last=(test_name == 'DeepFakeDetection'),
        )
    return loaders


def calculate_uthre(train_data_loader, num_classes, model, threshold_ratio=0.05):
    un_dict = {i: {'spatial': [], 'frequency': [], 'fused': []} for i in range(num_classes)}
    thre_dict = {'spatial': {}, 'frequency': {}, 'fused': {}}

    for data_dict in tqdm(train_data_loader, total=len(train_data_loader), desc='Calibrating'):
        for key in data_dict.keys():
            if data_dict[key] is not None and key != 'name':
                data_dict[key] = data_dict[key].cuda()
        predictions = model.model_full_fp(data_dict)
        for label_spe in range(num_classes):
            label_mask = data_dict['label'] == label_spe
            un_dict[label_spe]['spatial'].extend(predictions['spatial_vacuity'][label_mask])
            un_dict[label_spe]['frequency'].extend(predictions['frequency_vacuity'][label_mask])
            un_dict[label_spe]['fused'].extend(predictions['vacuity'][label_mask])

    for uncer_key in un_dict.keys():
        for branch in thre_dict.keys():
            un_dict[uncer_key][branch].sort(reverse=True)
            thre_index = int(len(un_dict[uncer_key][branch]) * threshold_ratio)
            thre_dict[branch][uncer_key] = un_dict[uncer_key][branch][thre_index].cpu().item()
    print('Retrieved uncertainty dict {} !'.format(thre_dict))
    return thre_dict


def main():
    args = get_args()
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.train_config, 'r') as f:
        train_cfg = yaml.safe_load(f)
    if 'label_dict' in config:
        train_cfg['label_dict'] = config['label_dict']
    config.update(train_cfg)

    if args.device_choice == 'old':
        torch.cuda.set_device(args.local_rank)
        config['local_rank'] = args.local_rank
    else:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        config['local_rank'] = local_rank

    config.update({
        'train_dataset': args.train_dataset,
        'test_dataset': args.test_dataset,
        'max_samples_per_class': args.max_samples_per_class,
        'training_data_ratio': args.training_data_ratio,
        'detect_novel': args.detect_novel,
        'num_classes': args.num_classes,
        'training_mode': args.training_mode,
        'device_choice': args.device_choice,
        'ddp': args.ddp,
        'avu_reg': args.avu_reg,
        'kl_reg': args.kl_reg,
    })
    config.setdefault('backbone_config', {})
    config['backbone_config']['num_classes'] = args.num_classes

    timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = f'_{args.task_target}' if args.task_target else ''
    logger_path = os.path.join(config['log_dir'],
                               config['model_name'] + task_str + '_' + timenow)
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'evaluation.log'))
    logger.info(f'Save log to {logger_path}')

    init_seed(config)
    if config['cudnn']:
        cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data_loader = get_train_loader(config)
    test_data_loaders = get_test_loaders(config)

    model_class = DETECTOR['dled']
    model = model_class(config)
    model.load_model(args.model_path)
    model.eval()

    num_classes = config['num_classes']
    u_threshold = calculate_uthre(train_data_loader, num_classes, model)

    for test_name, loader in test_data_loaders.items():
        image_total = 0
        spatial_correct = frequency_correct = fused_correct = 0
        spatial_bin_correct = frequency_bin_correct = fused_bin_correct = 0
        cat_correct = {b: {x: 0 for x in range(num_classes + 1)}
                       for b in ('spatial', 'frequency', 'fused')}
        cat_total = {b: {x: 0 for x in range(num_classes + 1)}
                     for b in ('spatial', 'frequency', 'fused')}

        for data_dict in tqdm(loader, desc=f'Testing[{test_name}]'):
            image_total += len(data_dict['label'])
            label_np = data_dict['label'].numpy()
            data_dict_cuda = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                              for k, v in data_dict.items()}
            with torch.no_grad():
                predictions = model.model_inference(data_dict_cuda, u_threshold)

            preds = {b: predictions['prob'][b].cpu().numpy() for b in ('spatial', 'frequency', 'fused')}
            spatial_correct += np.sum(preds['spatial'] == label_np)
            frequency_correct += np.sum(preds['frequency'] == label_np)
            fused_correct += np.sum(preds['fused'] == label_np)

            binary_label = (label_np != 0).astype(int)
            spatial_bin_correct += np.sum((preds['spatial'] != 0).astype(int) == binary_label)
            frequency_bin_correct += np.sum((preds['frequency'] != 0).astype(int) == binary_label)
            fused_bin_correct += np.sum((preds['fused'] != 0).astype(int) == binary_label)

            for class_idx in range(num_classes + 1):
                class_mask = label_np == class_idx
                if not np.any(class_mask):
                    continue
                for b in ('spatial', 'frequency', 'fused'):
                    cat_correct[b][class_idx] += np.sum(preds[b][class_mask] == class_idx)
                    cat_total[b][class_idx] += int(class_mask.sum())

        logger.info(f'==== {test_name} ====')
        for b in ('spatial', 'frequency', 'fused'):
            for class_idx in range(num_classes + 1):
                if cat_total[b][class_idx]:
                    acc = cat_correct[b][class_idx] / cat_total[b][class_idx]
                    logger.info(f'Class {class_idx} {b} accuracy: {acc:.5f}')
        logger.info(f'Spatial   binary: {spatial_bin_correct / image_total:.5f}, '
                    f'multi: {spatial_correct / image_total:.5f}')
        logger.info(f'Frequency binary: {frequency_bin_correct / image_total:.5f}, '
                    f'multi: {frequency_correct / image_total:.5f}')
        logger.info(f'Fused     binary: {fused_bin_correct / image_total:.5f}, '
                    f'multi: {fused_correct / image_total:.5f}')


if __name__ == '__main__':
    main()
