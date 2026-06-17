#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloudclear.data.dataset import SatelliteDataset
from cloudclear.training.trainer import CloudRemovalTrainer
from torch.utils.data import DataLoader, random_split


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description='CloudClear LISS-IV Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu or cuda)')
    args = parser.parse_args()

    import yaml
    from types import SimpleNamespace

    def load_config(path):
        with open(path) as f:
            d = yaml.safe_load(f)
        return d

    config = load_config(args.config)

    if args.data_dir:
        config['data']['clear_dir'] = args.data_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    config['training']['device'] = device

    # Convert dict to dot-accessible object
    class AttrDict:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, AttrDict(v))
                else:
                    setattr(self, k, v)

    cfg = AttrDict(config)

    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.data.clear_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Config: img_size={cfg.data.img_size}, "
          f"batch_size={cfg.data.batch_size}, "
          f"epochs={cfg.training.epochs}")

    set_seed(42)

    full_dataset = SatelliteDataset(cfg, split='train')

    if len(full_dataset) >= 10:
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        train_dataset = full_dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
        )

    trainer = CloudRemovalTrainer(cfg, device=device)
    trainer.train(train_loader, val_loader)

    print("\nTraining complete!")
    print(f"Results in: {cfg.paths.output_dir}")
    print(f"Checkpoints in: {cfg.paths.checkpoint_dir}")


if __name__ == '__main__':
    main()
