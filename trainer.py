"""
InfoGAN Trainer — two independent improvement switches:

  use_wgan_gp : bool   replace BCE-GAN with WGAN-GP          [improvement 1]
  use_infonce : bool   replace Q-network MI with InfoNCE      [improvement 2]

Four resulting combinations (set via TrainerConfig):
  mode='vanilla'          use_wgan_gp=False  use_infonce=False
  mode='wgan_gp'          use_wgan_gp=True   use_infonce=False
  mode='infonce'          use_wgan_gp=False  use_infonce=True
  mode='wgan_gp+infonce'  use_wgan_gp=True   use_infonce=True

Model files (one per dataset, named model_<dataset>.py):
  model_mnist.py   ← already done
  model_svhn.py    ← B to implement
  model_celeba.py  ← B to implement

Usage:
    cfg = TrainerConfig(mode='vanilla', dataset='mnist')
    trainer = InfoGANTrainer(cfg)
    trainer.train()
"""

import os
import math
import importlib
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from tqdm import tqdm

from datasets import build_loader

TINY = 1e-8
VALID_MODES = ('vanilla', 'wgan_gp', 'infonce', 'wgan_gp+infonce')


# ---------------------------------------------------------------------------
# Dynamic model import  (picks model_mnist / model_svhn / model_celeba)
# ---------------------------------------------------------------------------

def _load_model_module(dataset: str):
    """
    Import model_<dataset>.py and return the module.

    Convention — each model file must export:
        Generator, DiscriminatorQ,
        sample_latent, concat_latent, parse_q_output,
        NOISE_DIM, CAT_DIM, CONT_DIM, LATENT_DIM
    """
    module_name = f"model_{dataset}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"Cannot find '{module_name}.py'. "
            f"Make sure model_{dataset}.py is in the same directory as trainer.py."
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    mode:    str = 'vanilla'   # one of VALID_MODES
    dataset: str = 'mnist'     # 'mnist' | 'svhn' | 'celeba'
    data_dir: str = './data'

    # training
    batch_size:        int   = 128
    max_epochs:        int   = 50
    updates_per_epoch: int   = 100

    # optimiser
    lr_d:       float = 2e-4
    lr_g:       float = 1e-3
    adam_beta1: float = 0.5
    adam_beta2: float = 0.999

    # MI loss weights
    lambda_disc: float = 1.0
    lambda_cont: float = 0.1

    # WGAN-GP
    lambda_gp: float = 10.0
    n_critic:  int   = 1

    # InfoNCE
    infonce_temp: float = 0.1

    # logging
    log_dir:        str = 'logs'
    checkpoint_dir: str = 'checkpoints'
    save_every:     int = 10
    vis_every:      int = 1

    def __post_init__(self):
        assert self.mode in VALID_MODES, \
            f"mode must be one of {VALID_MODES}, got '{self.mode}'"

    @property
    def use_wgan_gp(self) -> bool:
        return 'wgan_gp' in self.mode

    @property
    def use_infonce(self) -> bool:
        return 'infonce' in self.mode


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def bce_d_loss(real_d, fake_d):
    return (-torch.mean(torch.log(real_d + TINY))
            - torch.mean(torch.log(1. - fake_d + TINY)))

def bce_g_loss(fake_d):
    return -torch.mean(torch.log(fake_d + TINY))

def wgan_d_loss(real_d, fake_d):
    return torch.mean(fake_d) - torch.mean(real_d)

def wgan_g_loss(fake_d):
    return -torch.mean(fake_d)

def gradient_penalty(DQ, real_imgs, fake_imgs, device):
    B   = real_imgs.size(0)
    eps = torch.rand(B, 1, 1, 1, device=device)
    x_hat = (eps * real_imgs + (1 - eps) * fake_imgs).requires_grad_(True)
    d_hat, _ = DQ(x_hat)
    grads = torch.autograd.grad(
        outputs=d_hat, inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True, retain_graph=True,
    )[0]
    grad_norm = grads.view(B, -1).norm(2, dim=1)
    return torch.mean((grad_norm - 1.) ** 2)

def _split_categorical(tensor, cat_dims):
    if len(cat_dims) == 1:
        return (tensor,)
    return torch.split(tensor, cat_dims, dim=1)


def mi_orig_discrete(c_cat, cat_prob, cat_dims):
    losses = []
    for c_i, p_i in zip(_split_categorical(c_cat, cat_dims),
                        _split_categorical(cat_prob, cat_dims)):
        targets = c_i.argmax(dim=1)
        logits = torch.log(p_i + TINY)
        losses.append(F.cross_entropy(logits, targets))
    return torch.stack(losses).mean()

def mi_orig_continuous(c_cont, cont_mean, cont_std):
    if c_cont.numel() == 0:
        return c_cont.new_tensor(0.0)
    nll = (torch.log(cont_std + TINY)
           + 0.5 * ((c_cont - cont_mean) / (cont_std + TINY)) ** 2)
    return nll.mean()

def mi_infonce_discrete(c_cat, cat_prob, cat_dims, temperature=0.1):
    losses = []
    for c_i, p_i in zip(_split_categorical(c_cat, cat_dims),
                        _split_categorical(cat_prob, cat_dims)):
        log_q = torch.log(p_i + TINY)
        logits = torch.matmul(log_q, c_i.T) / temperature
        targets = torch.arange(c_i.size(0), device=c_i.device)
        losses.append(F.cross_entropy(logits, targets))
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class InfoGANTrainer:

    def __init__(self, cfg: TrainerConfig):
        self.cfg    = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] device={self.device}  dataset={cfg.dataset}  "
              f"mode={cfg.mode}  "
              f"(wgan_gp={cfg.use_wgan_gp}, infonce={cfg.use_infonce})")

        # ── load the right model file ────────────────────────────────────────
        m = _load_model_module(cfg.dataset)
        self.Generator      = m.Generator
        self.DiscriminatorQ = m.DiscriminatorQ
        self.sample_latent  = m.sample_latent
        self.concat_latent  = m.concat_latent
        self.parse_q_output = m.parse_q_output
        self.NOISE_DIM      = m.NOISE_DIM
        self.CAT_DIM        = m.CAT_DIM
        self.CAT_DIMS       = getattr(m, 'CAT_DIMS', (m.CAT_DIM,))
        self.CONT_DIM       = m.CONT_DIM
        self.IMAGE_VALUE_RANGE = getattr(m, 'IMAGE_VALUE_RANGE', (0, 1))

        # ── networks ────────────────────────────────────────────────────────
        self.G  = m.Generator().to(self.device)
        self.DQ = m.DiscriminatorQ().to(self.device)

        if cfg.use_wgan_gp:
            self.DQ.d_head = nn.Sequential(
                nn.Linear(1024, 1)
            ).to(self.device)

        # ── optimisers ──────────────────────────────────────────────────────
        self.opt_G = torch.optim.Adam(
            self.G.parameters(),
            lr=cfg.lr_g, betas=(cfg.adam_beta1, cfg.adam_beta2),
        )
        self.opt_DQ = torch.optim.Adam(
            self.DQ.parameters(),
            lr=cfg.lr_d, betas=(cfg.adam_beta1, cfg.adam_beta2),
        )

        # ── data ────────────────────────────────────────────────────────────
        self.loader = build_loader(
            cfg.dataset, data_dir=cfg.data_dir, batch_size=cfg.batch_size
        )

        # ── logging ─────────────────────────────────────────────────────────
        ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{cfg.dataset}_{cfg.mode}_{ts}"
        self.writer = SummaryWriter(os.path.join(cfg.log_dir, run_name))
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        # ── fixed latents for visualisation ──────────────────────────────────
        self.fixed_noise, self.fixed_c_cat, self.fixed_c_cont = \
            self._make_fixed_latents()

    # -----------------------------------------------------------------------
    # Fixed latents
    # -----------------------------------------------------------------------

    def _make_fixed_latents(self):
        device = self.device
        NOISE_DIM = self.NOISE_DIM
        CONT_DIM  = self.CONT_DIM

        first_cat_dim = self.CAT_DIMS[0]
        B = first_cat_dim * 10
        base = torch.empty(10, NOISE_DIM, device=device).uniform_(-1, 1)
        noise = base.repeat(first_cat_dim, 1)
        c_cat = self._make_cat_traversal(0, B)
        c_cont = torch.zeros(B, CONT_DIM, device=device)
        return noise, c_cat, c_cont

    def _make_cat_traversal(self, code_idx, batch_size):
        cat_dim = self.CAT_DIMS[code_idx]
        n_cols = batch_size // cat_dim
        c_cat = torch.zeros(batch_size, self.CAT_DIM, device=self.device)

        offset = 0
        for dim in self.CAT_DIMS:
            c_cat[:, offset] = 1.0
            offset += dim

        start = sum(self.CAT_DIMS[:code_idx])
        c_cat[:, start:start + cat_dim] = 0.0
        for value in range(cat_dim):
            row = slice(value * n_cols, (value + 1) * n_cols)
            c_cat[row, start + value] = 1.0
        return c_cat

    def _cat_entropy_target(self):
        return sum(math.log(dim) for dim in self.CAT_DIMS) / len(self.CAT_DIMS)

    # -----------------------------------------------------------------------
    # MI loss selector
    # -----------------------------------------------------------------------

    def _mi_loss(self, c_cat, cat_prob, c_cont, cont_mean, cont_std):
        if self.cfg.use_infonce:
            mi_disc = mi_infonce_discrete(c_cat, cat_prob, self.CAT_DIMS,
                                          self.cfg.infonce_temp)
        else:
            mi_disc = mi_orig_discrete(c_cat, cat_prob, self.CAT_DIMS)
        mi_cont = mi_orig_continuous(c_cont, cont_mean, cont_std)
        return mi_disc, mi_cont

    # -----------------------------------------------------------------------
    # Single training step
    # -----------------------------------------------------------------------

    def _step(self, real_imgs):
        cfg    = self.cfg
        device = self.device
        B      = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        z_noise, c_cat, c_cont = self.sample_latent(B, device)
        z = self.concat_latent(z_noise, c_cat, c_cont)

        fake_imgs = self.G(z)

        # D / Q update
        self.opt_DQ.zero_grad()
        real_d, _     = self.DQ(real_imgs)
        fake_d, q_out = self.DQ(fake_imgs.detach())
        cat_prob, cont_mean, cont_std = self.parse_q_output(q_out)

        if cfg.use_wgan_gp:
            d_loss = (wgan_d_loss(real_d, fake_d)
                      + cfg.lambda_gp * gradient_penalty(
                          self.DQ, real_imgs, fake_imgs.detach(), device))
        else:
            d_loss = bce_d_loss(real_d, fake_d)

        mi_disc, mi_cont = self._mi_loss(c_cat, cat_prob, c_cont,
                                          cont_mean, cont_std)
        mi_total = cfg.lambda_disc * mi_disc + cfg.lambda_cont * mi_cont
        (d_loss + mi_total).backward()
        self.opt_DQ.step()

        # G update
        self.opt_G.zero_grad()
        fake_imgs_g       = self.G(z)
        fake_d_g, q_out_g = self.DQ(fake_imgs_g)
        cat_prob_g, cont_mean_g, cont_std_g = self.parse_q_output(q_out_g)

        g_loss = wgan_g_loss(fake_d_g) if cfg.use_wgan_gp else bce_g_loss(fake_d_g)

        mi_disc_g, mi_cont_g = self._mi_loss(c_cat, cat_prob_g, c_cont,
                                              cont_mean_g, cont_std_g)
        mi_total_g = cfg.lambda_disc * mi_disc_g + cfg.lambda_cont * mi_cont_g
        (g_loss + mi_total_g).backward()
        self.opt_G.step()

        with torch.no_grad():
            li_disc = self._cat_entropy_target() - mi_disc.item()

        return {
            'd_loss' : d_loss.item(),
            'g_loss' : g_loss.item(),
            'mi_disc': mi_disc.item(),
            'mi_cont': mi_cont.item(),
            'LI_disc': li_disc,
        }

    # -----------------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _visualise(self, epoch):
        self.G.eval()
        z = self.concat_latent(self.fixed_noise, self.fixed_c_cat,
                                self.fixed_c_cont)
        imgs = self.G(z)
        grid = make_grid(imgs, nrow=10, normalize=True,
                         value_range=self.IMAGE_VALUE_RANGE)
        self.writer.add_image('traversal/c1_category', grid, epoch)

        NOISE_DIM = self.NOISE_DIM
        CONT_DIM  = self.CONT_DIM
        device    = self.device

        if len(self.CAT_DIMS) > 1:
            cat_dim = self.CAT_DIMS[0]
            base = torch.empty(10, NOISE_DIM, device=device).uniform_(-1, 1)
            zn = base.repeat(cat_dim, 1)
            c_cont = torch.zeros(zn.size(0), CONT_DIM, device=device)

            for code_idx in range(len(self.CAT_DIMS)):
                c_cat = self._make_cat_traversal(code_idx, zn.size(0))
                imgs_cat = self.G(self.concat_latent(zn, c_cat, c_cont))
                self.writer.add_image(
                    f'traversal/cat_{code_idx:02d}',
                    make_grid(imgs_cat, nrow=10, normalize=True,
                              value_range=self.IMAGE_VALUE_RANGE),
                    epoch,
                )

        if CONT_DIM == 0:
            self.G.train()
            return

        c0 = torch.zeros(10, self.CAT_DIM, device=device)
        offset = 0
        for dim in self.CAT_DIMS:
            c0[:, offset] = 1.
            offset += dim
        zn = torch.zeros(10, NOISE_DIM, device=device)
        sweep = torch.linspace(-2, 2, 10, device=device)

        for cont_idx in range(CONT_DIM):
            c_cont = torch.zeros(10, CONT_DIM, device=device)
            c_cont[:, cont_idx] = sweep
            imgs_cont = self.G(self.concat_latent(zn, c0, c_cont))
            self.writer.add_image(
                f'traversal/cont_{cont_idx:02d}',
                make_grid(imgs_cont, nrow=10, normalize=True,
                          value_range=self.IMAGE_VALUE_RANGE),
                epoch,
            )

        self.G.train()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        for epoch in range(cfg.max_epochs):
            self.G.train(); self.DQ.train()
            totals  = {k: 0.0 for k in
                       ['d_loss', 'g_loss', 'mi_disc', 'mi_cont', 'LI_disc']}
            data_it = iter(self.loader)
            pbar    = tqdm(range(cfg.updates_per_epoch),
                           desc=f'Epoch {epoch:03d}', leave=False)

            for _ in pbar:
                try:
                    imgs, _ = next(data_it)
                except StopIteration:
                    data_it = iter(self.loader)
                    imgs, _ = next(data_it)

                logs = self._step(imgs)
                for k, v in logs.items():
                    totals[k] += v
                pbar.set_postfix(D=f"{logs['d_loss']:.3f}",
                                 G=f"{logs['g_loss']:.3f}",
                                 LI=f"{logs['LI_disc']:.3f}")

            n   = cfg.updates_per_epoch
            avg = {k: v / n for k, v in totals.items()}
            print(f"Epoch {epoch:03d} | "
                  f"D={avg['d_loss']:.4f}  G={avg['g_loss']:.4f}  "
                  f"MI={avg['mi_disc']:.4f}  LI={avg['LI_disc']:.4f}  "
                  f"target≈{self._cat_entropy_target():.3f}")
            for k, v in avg.items():
                self.writer.add_scalar(f'train/{k}', v, epoch)

            if epoch % cfg.vis_every == 0:
                self._visualise(epoch)
            if (epoch + 1) % cfg.save_every == 0:
                self._save_checkpoint(epoch)

        self._save_checkpoint(cfg.max_epochs - 1, final=True)
        self.writer.close()
        print('Training complete.')

    # -----------------------------------------------------------------------
    # Checkpoint helpers
    # -----------------------------------------------------------------------

    def _save_checkpoint(self, epoch, final=False):
        tag  = 'final' if final else f'epoch{epoch:03d}'
        path = os.path.join(self.cfg.checkpoint_dir,
                            f"{self.cfg.dataset}_{self.cfg.mode}_{tag}.pt")
        torch.save({
            'epoch'       : epoch,
            'mode'        : self.cfg.mode,
            'dataset'     : self.cfg.dataset,
            'G_state'     : self.G.state_dict(),
            'DQ_state'    : self.DQ.state_dict(),
            'opt_G_state' : self.opt_G.state_dict(),
            'opt_DQ_state': self.opt_DQ.state_dict(),
        }, path)
        print(f'  Checkpoint → {path}')

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt['G_state'])
        self.DQ.load_state_dict(ckpt['DQ_state'])
        self.opt_G.load_state_dict(ckpt['opt_G_state'])
        self.opt_DQ.load_state_dict(ckpt['opt_DQ_state'])
        print(f'  Checkpoint ← {path}  (epoch {ckpt["epoch"]})')
        return ckpt['epoch']
