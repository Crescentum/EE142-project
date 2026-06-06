"""
Entry point. Switch modes via --mode argument.

Examples:
    # baseline
    python train.py --mode vanilla --dataset mnist --epochs 50

    # improvement 1: WGAN-GP only
    python train.py --mode wgan_gp --dataset mnist --epochs 50

    # improvement 2: InfoNCE only
    python train.py --mode infonce --dataset mnist --epochs 50

    # both improvements combined
    python train.py --mode wgan_gp+infonce --dataset mnist --epochs 50

    # other datasets (model.py needs to be extended first)
    python train.py --mode vanilla --dataset svhn   --epochs 50
    python train.py --mode vanilla --dataset celeba --epochs 50

    # resume from checkpoint
    python train.py --mode vanilla --resume checkpoints/mnist_vanilla_epoch049.pt
"""

import argparse
from trainer import InfoGANTrainer, TrainerConfig, VALID_MODES
import torch



def parse_args():
    p = argparse.ArgumentParser(description='InfoGAN training')
    p.add_argument('--mode',       type=str, default='vanilla',
                   choices=list(VALID_MODES))
    p.add_argument('--dataset',    type=str, default='mnist',
                   choices=['mnist', 'svhn', 'celeba'])
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--data_dir',   type=str, default='./data')
    p.add_argument('--log_dir',    type=str, default='./logs')
    p.add_argument('--ckpt_dir',   type=str, default='./checkpoints')
    p.add_argument('--lr_d',       type=float, default=2e-4)
    p.add_argument('--lr_g',       type=float, default=1e-3)
    p.add_argument('--lambda_gp',  type=float, default=10.0,
                   help='gradient penalty weight (wgan_gp modes only)')
    p.add_argument('--lambda_disc',type=float, default=1.0,
                   help='MI loss weight for discrete code')
    p.add_argument('--lambda_cont',type=float, default=0.1,
                   help='MI loss weight for continuous codes')
    p.add_argument('--infonce_temp',type=float, default=0.1,
                   help='InfoNCE temperature τ (infonce modes only)')
    p.add_argument('--resume',     type=str, default=None,
                   help='path to .pt checkpoint to resume from')
    return p.parse_args()


def main():
    args = parse_args()

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        resumed_dataset = ckpt.get('dataset', args.dataset)
        resumed_mode    = ckpt.get('mode', args.mode)
        resumed_epoch   = ckpt.get('epoch', -1)
        
        if args.dataset != 'mnist' and args.dataset != resumed_dataset:
            print(f"[Warning] Command line --dataset {args.dataset} conflicts with "
                  f"checkpoint dataset '{resumed_dataset}'. Using checkpoint value.")
        if args.mode != 'vanilla' and args.mode != resumed_mode:
            print(f"[Warning] Command line --mode {args.mode} conflicts with "
                  f"checkpoint mode '{resumed_mode}'. Using checkpoint value.")
        
        args.dataset = resumed_dataset
        args.mode    = resumed_mode
        print(f"[Resume] Loaded config from checkpoint: dataset={resumed_dataset}, "
              f"mode={resumed_mode}, epoch={resumed_epoch}")

    cfg = TrainerConfig(
        mode            = args.mode,
        dataset         = args.dataset,
        max_epochs      = args.epochs,
        batch_size      = args.batch_size,
        data_dir        = args.data_dir,
        log_dir         = args.log_dir,
        checkpoint_dir  = args.ckpt_dir,
        lr_d            = args.lr_d,
        lr_g            = args.lr_g,
        lambda_gp       = args.lambda_gp,
        lambda_disc     = args.lambda_disc,
        lambda_cont     = args.lambda_cont,
        infonce_temp    = args.infonce_temp,
    )

    trainer = InfoGANTrainer(cfg)

    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"[Resume] Training will continue from epoch {start_epoch + 1}")

    trainer.train()


if __name__ == '__main__':
    main()