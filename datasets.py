"""
datasets.py — DataLoader factory for MNIST, SVHN, CelebA.

Key differences between datasets that affect preprocessing:
  MNIST  : 28×28 grayscale,  pixel range [0, 1],  G uses Sigmoid
  SVHN   : 32×32 RGB,        pixel range [-1, 1], G uses Tanh
  CelebA : 64×64 RGB,        pixel range [-1, 1], G uses Tanh

Data must be downloaded in advance (cluster has no internet access).
Expected directory layout under data_dir:

  MNIST:
    data/MNIST/raw/train-images-idx3-ubyte
    data/MNIST/raw/train-labels-idx3-ubyte
    data/MNIST/raw/t10k-images-idx3-ubyte
    data/MNIST/raw/t10k-labels-idx3-ubyte

  SVHN:
    data/train_32x32.mat
    data/test_32x32.mat

  CelebA:
    data/celeba/img_align_celeba/*.jpg
    data/celeba/list_attr_celeba.txt
    data/celeba/list_eval_partition.txt

Download on a machine with internet access, then scp to cluster:
    python -c "
    from torchvision import datasets
    datasets.MNIST('./data', download=True)
    datasets.SVHN('./data', download=True)
    datasets.CelebA('./data', download=True)
    "
"""

import os
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Per-dataset metadata
# ---------------------------------------------------------------------------
@dataclass
class DatasetMeta:
    image_size  : int
    channels    : int
    pixel_range : str
    train_size  : int
    noise_dim   : int
    cat_dims    : Tuple[int, ...]
    cont_dim    : int
    lambda_disc : float
    lambda_cont : float

    @property
    def latent_dim(self) -> int:
        return self.noise_dim + sum(self.cat_dims) + self.cont_dim

    @property
    def q_out_dim(self) -> int:
        return sum(self.cat_dims) + self.cont_dim * 2


DATASET_CFG = {
    'mnist': DatasetMeta(
        image_size=28, channels=1, pixel_range='01', train_size=60_000,
        noise_dim=62, cat_dims=(10,), cont_dim=2,
        lambda_disc=1.0, lambda_cont=0.1,
    ),
    'svhn': DatasetMeta(
        image_size=32, channels=3, pixel_range='11', train_size=73_257,
        noise_dim=124, cat_dims=(10, 10, 10, 10), cont_dim=4,
        lambda_disc=1.0, lambda_cont=0.1,
    ),
    'celeba': DatasetMeta(
        image_size=64, channels=3, pixel_range='11', train_size=162_770,
        noise_dim=128, cat_dims=(10,) * 10, cont_dim=0,
        lambda_disc=1.0, lambda_cont=0.0,
    ),
}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def _svhn_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

def _celeba_transform(image_size: int = 64):
    return transforms.Compose([
        transforms.CenterCrop(140),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


# ---------------------------------------------------------------------------
# Helper: clear error when data is missing
# ---------------------------------------------------------------------------

def _check_data(data_dir: str, expected_path: str, name: str):
    full = os.path.join(data_dir, expected_path)
    if not os.path.exists(full):
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"  {name} data not found at:\n  {full}\n\n"
            f"  Cluster has no internet. Download locally first:\n\n"
            f"    python -c \"from torchvision import datasets; "
            f"datasets.{name}('./data', download=True)\"\n\n"
            f"  Then copy to cluster:\n\n"
            f"    scp -r ./data/{name} <user>@<cluster>:~/EE142/data/\n"
            f"{'='*60}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_loader(
    dataset    : str,
    data_dir   : str = './data',
    batch_size : int = 128,
    num_workers: int = 4,
    split      : str = 'train',
) -> DataLoader:

    os.makedirs(data_dir, exist_ok=True)

    if dataset == 'mnist':
        _check_data(data_dir, 'MNIST/raw/train-images-idx3-ubyte', 'MNIST')
        ds = datasets.MNIST(
            root=data_dir, train=(split == 'train'),
            download=False,
            transform=_mnist_transform(),
        )

    elif dataset == 'svhn':
        _check_data(data_dir, 'train_32x32.mat', 'SVHN')
        svhn_split = split if split in ('train', 'test', 'extra') else 'train'
        ds = datasets.SVHN(
            root=data_dir, split=svhn_split,
            download=False,
            transform=_svhn_transform(),
        )

    elif dataset == 'celeba':
        _check_data(data_dir, 'celeba/img_align_celeba', 'CelebA')
        celeba_split = split if split in ('train', 'valid', 'test', 'all') else 'train'
        ds = datasets.CelebA(
            root=data_dir, split=celeba_split,
            target_type='attr',
            download=False,
            transform=_celeba_transform(DATASET_CFG['celeba'].image_size),
        )

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. "
                         f"Choose from: {list(DATASET_CFG.keys())}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def denorm(images: torch.Tensor, pixel_range: str) -> torch.Tensor:
    if pixel_range == '11':
        return (images * 0.5 + 0.5).clamp(0, 1)
    return images.clamp(0, 1)