"""
InfoGAN Trainer for MNIST — vanilla InfoGAN only.

Reproduces Chen et al., NeurIPS 2016 (InfoGAN) on MNIST.

Architecture follows Appendix C.1 exactly:
  - Generator:      FC1024 → FC7x7x128 → ConvT64 → ConvT1 (Sigmoid)
  - Discriminator:  Conv64 → Conv128 → FC1024 → D head / Q head
  - Latent codes:   z(62) + c1_categorical(10) + c2,c3_continuous(2)

Training objective (Eq. 6):
  min_{G,Q} max_D  V(D,G) - λ * L_I(G,Q)
  where L_I is the mutual information lower bound.
"""

import os
import math
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from tqdm import tqdm

import model_mnist as m
from datasets import build_loader

TINY = 1e-8


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    data_dir: str  = './data'

    batch_size:        int   = 128
    max_epochs:        int   = 50
    updates_per_epoch: int   = 0   # 0 = full dataset

    lr_d:       float = 2e-4   # paper Appendix C.1
    lr_g:       float = 1e-3   # paper Appendix C.1
    adam_beta1: float = 0.5
    adam_beta2: float = 0.999

    lambda_disc: float = 1.0   # λ for discrete MI loss   (paper λ=1)
    lambda_cont: float = 0.1   # λ for continuous MI loss (paper λ=0.1)

    log_dir:        str = 'logs'
    checkpoint_dir: str = 'checkpoints'
    save_every:     int = 10
    vis_every:      int = 1


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def bce_d_loss(real_d, fake_d):
    """
    Standard GAN discriminator loss (Eq. 1):
      -E[log D(x)] - E[log(1 - D(G(z)))]
    """
    real_d = real_d.clamp(1e-6, 1 - 1e-6)
    fake_d = fake_d.clamp(1e-6, 1 - 1e-6)
    return (-torch.mean(torch.log(real_d))
            - torch.mean(torch.log(1. - fake_d)))

def bce_g_loss(fake_d):
    """
    Non-saturating generator loss (Goodfellow et al., 2014):
      -E[log D(G(z))]
    """
    fake_d = fake_d.clamp(1e-6, 1 - 1e-6)
    return -torch.mean(torch.log(fake_d))

def mi_disc_loss(c_cat, cat_prob):
    """
    MI lower bound for discrete code (Eq. 4):
      L_I^disc = H(c1) - H(c1|G(z,c)) ≈ log(K) - CrossEntropy(Q(c1|x), c1)
    We minimise CrossEntropy, which maximises L_I^disc.
    """
    targets = c_cat.argmax(dim=1)
    logits  = torch.log(cat_prob + TINY)
    return F.cross_entropy(logits, targets)

def mi_cont_loss(c_cont, cont_mean, cont_std):
    """
    MI lower bound for continuous codes (Eq. 5):
      L_I^cont ≈ -H(c2,c3|G(z,c)) modelled as Gaussian NLL
    """
    nll = (torch.log(cont_std + TINY)
           + 0.5 * ((c_cont - cont_mean) / (cont_std + TINY)) ** 2)
    return nll.mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class InfoGANTrainer:

    def __init__(self, cfg: TrainerConfig):
        self.cfg    = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] device={self.device}  dataset=mnist  mode=vanilla")

        self.G  = m.Generator().to(self.device)
        self.DQ = m.DiscriminatorQ().to(self.device)

        self.opt_G = torch.optim.Adam(
            self.G.parameters(),
            lr=cfg.lr_g, betas=(cfg.adam_beta1, cfg.adam_beta2),
        )
        self.opt_DQ = torch.optim.Adam(
            self.DQ.parameters(),
            lr=cfg.lr_d, betas=(cfg.adam_beta1, cfg.adam_beta2),
        )

        self.loader = build_loader(
            'mnist', data_dir=cfg.data_dir, batch_size=cfg.batch_size
        )

        ts          = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(cfg.log_dir, f"mnist_vanilla_{ts}"))
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        self.fixed_noise, self.fixed_c_cat, self.fixed_c_cont = \
            self._make_fixed_latents()

    # -----------------------------------------------------------------------
    # Fixed latents for TensorBoard
    # -----------------------------------------------------------------------

    def _make_fixed_latents(self):
        device = self.device
        base   = torch.FloatTensor(10, m.NOISE_DIM).uniform_(-1, 1)
        noise  = base.repeat_interleave(10, dim=0).to(device)   # (100, 62)
        c_cat  = torch.zeros(100, m.CAT_DIM, device=device)
        for i in range(10):
            c_cat[i*10:(i+1)*10, i] = 1.0
        c_cont = torch.zeros(100, m.CONT_DIM, device=device)
        return noise, c_cat, c_cont

    # -----------------------------------------------------------------------
    # Single training step
    # -----------------------------------------------------------------------

    def _step(self, real_imgs):
        cfg    = self.cfg
        device = self.device
        B      = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        z_noise, c_cat, c_cont = m.sample_latent(B, device)
        z         = m.concat_latent(z_noise, c_cat, c_cont)
        fake_imgs = self.G(z)

        # ── D / Q update ────────────────────────────────────────────────────
        self.opt_DQ.zero_grad()
        real_d, _     = self.DQ(real_imgs)
        fake_d, q_out = self.DQ(fake_imgs.detach())
        cat_prob, cont_mean, cont_std = m.parse_q_output(q_out)

        d_loss   = bce_d_loss(real_d, fake_d)
        mi_disc  = mi_disc_loss(c_cat, cat_prob)
        mi_cont  = mi_cont_loss(c_cont, cont_mean, cont_std)
        mi_total = cfg.lambda_disc * mi_disc + cfg.lambda_cont * mi_cont

        (d_loss + mi_total).backward()
        self.opt_DQ.step()

        # ── G update ────────────────────────────────────────────────────────
        self.opt_G.zero_grad()
        fake_imgs_g       = self.G(z)
        fake_d_g, q_out_g = self.DQ(fake_imgs_g)
        cat_prob_g, cont_mean_g, cont_std_g = m.parse_q_output(q_out_g)

        g_loss     = bce_g_loss(fake_d_g)
        mi_disc_g  = mi_disc_loss(c_cat, cat_prob_g)
        mi_cont_g  = mi_cont_loss(c_cont, cont_mean_g, cont_std_g)
        mi_total_g = cfg.lambda_disc * mi_disc_g + cfg.lambda_cont * mi_cont_g

        (g_loss + mi_total_g).backward()
        self.opt_G.step()

        with torch.no_grad():
            li = math.log(m.CAT_DIM) - mi_disc.item()

        return {
            'd_loss' : d_loss.item(),
            'g_loss' : g_loss.item(),
            'mi_disc': mi_disc.item(),
            'mi_cont': mi_cont.item(),
            'LI_disc': li,
        }

    # -----------------------------------------------------------------------
    # TensorBoard visualisation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _visualise(self, epoch):
        self.G.eval()
        z    = m.concat_latent(self.fixed_noise, self.fixed_c_cat, self.fixed_c_cont)
        imgs = self.G(z)
        self.writer.add_image('traversal/c1_category',
            make_grid(imgs, nrow=10, normalize=True, value_range=(0,1)), epoch)

        device = self.device
        c0    = torch.zeros(10, m.CAT_DIM,  device=device); c0[:, 0] = 1.0
        zn    = torch.zeros(10, m.NOISE_DIM, device=device)
        sweep = torch.linspace(-2, 2, 10, device=device)

        cc2 = torch.zeros(10, m.CONT_DIM, device=device); cc2[:, 0] = sweep
        self.writer.add_image('traversal/c2',
            make_grid(self.G(m.concat_latent(zn, c0, cc2)),
                      nrow=10, normalize=True, value_range=(0,1)), epoch)

        cc3 = torch.zeros(10, m.CONT_DIM, device=device); cc3[:, 1] = sweep
        self.writer.add_image('traversal/c3',
            make_grid(self.G(m.concat_latent(zn, c0, cc3)),
                      nrow=10, normalize=True, value_range=(0,1)), epoch)

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
            n_steps = cfg.updates_per_epoch if cfg.updates_per_epoch > 0 \
                      else len(self.loader)
            pbar    = tqdm(range(n_steps), desc=f'Epoch {epoch:03d}', leave=False)

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

            avg = {k: v / n_steps for k, v in totals.items()}
            print(f"Epoch {epoch:03d} | "
                  f"D={avg['d_loss']:.4f}  G={avg['g_loss']:.4f}  "
                  f"MI={avg['mi_disc']:.4f}  LI={avg['LI_disc']:.4f}  "
                  f"target≈{math.log(m.CAT_DIM):.3f}")
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
                            f"mnist_vanilla_{tag}.pt")
        torch.save({
            'epoch'       : epoch,
            'mode'        : 'vanilla',
            'dataset'     : 'mnist',
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