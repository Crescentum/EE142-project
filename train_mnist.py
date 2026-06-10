"""
Train vanilla InfoGAN on MNIST.

Examples:
    # InfoGAN
    python train_mnist.py --epochs 50

    # GAN baseline (no MI regularization, for Figure 1 comparison)
    python train_mnist.py --lambda_disc 0.0 --lambda_cont 0.0 --ckpt_dir ./checkpoints_gan

    # resume
    python train_mnist.py --resume checkpoints/mnist_vanilla_epoch049.pt
"""

import argparse
from trainer_mnist import InfoGANTrainer, TrainerConfig


def parse_args():
    p = argparse.ArgumentParser(description='Vanilla InfoGAN on MNIST')
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--batch_size',  type=int,   default=128)
    p.add_argument('--data_dir',    type=str,   default='./data')
    p.add_argument('--log_dir',     type=str,   default='./logs')
    p.add_argument('--ckpt_dir',    type=str,   default='./checkpoints')
    p.add_argument('--lr_d',        type=float, default=2e-4)
    p.add_argument('--lr_g',        type=float, default=1e-3)
    p.add_argument('--lambda_disc', type=float, default=1.0)
    p.add_argument('--lambda_cont', type=float, default=0.1)
    p.add_argument('--resume',      type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = TrainerConfig(
        data_dir        = args.data_dir,
        max_epochs      = args.epochs,
        batch_size      = args.batch_size,
        log_dir         = args.log_dir,
        checkpoint_dir  = args.ckpt_dir,
        lr_d            = args.lr_d,
        lr_g            = args.lr_g,
        lambda_disc     = args.lambda_disc,
        lambda_cont     = args.lambda_cont,
    )

    trainer = InfoGANTrainer(cfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == '__main__':
    main()