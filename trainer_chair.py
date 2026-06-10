"""
InfoGAN Trainer — supports both single (MNIST) and multiple (Chairs/SVHN) categorical codes.

Modes: vanilla, wgan_gp, infonce, wgan_gp+infonce
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


def _load_model_module(dataset: str):
    module_name = f"model_{dataset}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"Cannot find '{module_name}.py'. "
            f"Make sure model_{dataset}.py is in the same directory as trainer.py."
        )


@dataclass
class TrainerConfig:
    mode: str = 'vanilla'
    dataset: str = 'mnist'
    data_dir: str = './data'

    batch_size: int = 128
    max_epochs: int = 50
    updates_per_epoch: int = 0

    lr_d: float = 2e-4
    lr_g: float = 1e-3
    adam_beta1: float = 0.5
    adam_beta2: float = 0.999

    lambda_disc: float = 1.0
    lambda_cont: float = 0.1

    lambda_gp: float = 10.0
    n_critic: int = 1

    infonce_temp: float = 0.1

    log_dir: str = 'logs'
    checkpoint_dir: str = 'checkpoints'
    save_every: int = 10
    vis_every: int = 1

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
    real_d = real_d.clamp(1e-6, 1 - 1e-6)
    fake_d = fake_d.clamp(1e-6, 1 - 1e-6)
    return (-torch.mean(torch.log(real_d))
            - torch.mean(torch.log(1. - fake_d)))

def bce_g_loss(fake_d):
    fake_d = fake_d.clamp(1e-6, 1 - 1e-6)
    return -torch.mean(torch.log(fake_d))

def wgan_d_loss(real_d, fake_d):
    return torch.mean(fake_d) - torch.mean(real_d)

def wgan_g_loss(fake_d):
    return -torch.mean(fake_d)

def gradient_penalty(DQ, real_imgs, fake_imgs, device):
    B = real_imgs.size(0)
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

def mi_orig_discrete(c_cat, cat_prob, cat_dims=None):
    """
    Mutual information loss for discrete latent code(s).
    Supports multiple categorical codes via cat_dims.
    """
    if cat_dims is None or len(cat_dims) == 1:
        # Single categorical code (backward compatible)
        if isinstance(cat_prob, list):
            cat_prob = cat_prob[0]
        targets = c_cat.argmax(dim=1)
        logits = torch.log(cat_prob + TINY)
        return F.cross_entropy(logits, targets)

    # Multiple categorical codes
    loss = 0.0
    offset = 0
    probs = cat_prob if isinstance(cat_prob, list) else \
            [cat_prob[:, offset:offset + d] for d in cat_dims]
    for i, dim in enumerate(cat_dims):
        c_i = c_cat[:, offset:offset + dim]
        p_i = probs[i]
        targets = c_i.argmax(dim=1)
        logits = torch.log(p_i + TINY)
        loss += F.cross_entropy(logits, targets)
        offset += dim
    return loss / len(cat_dims)

def mi_orig_continuous(c_cont, cont_mean, cont_std):
    nll = (torch.log(cont_std + TINY)
           + 0.5 * ((c_cont - cont_mean) / (cont_std + TINY)) ** 2)
    return nll.mean()

def mi_infonce_discrete(c_cat, feat, temperature=0.1, proj_feat=None, proj_cat=None):
    """
    InfoNCE for discrete latent codes with learnable projection.
    Supports multi-code c_cat by concatenating codes and projecting to same dim.
    """
    B = c_cat.size(0)
    feat_dim = feat.size(1)
    cat_dim = c_cat.size(1)
    
    # 如果维度不匹配，使用投影
    if feat_dim != cat_dim:
        if proj_feat is None:
            proj_feat = nn.Linear(feat_dim, 256, bias=False).to(feat.device)
        if proj_cat is None:
            proj_cat = nn.Linear(cat_dim, 256, bias=False).to(c_cat.device)
        feat = proj_feat(feat)
        c_cat = proj_cat(c_cat)
    
    feat_n = F.normalize(feat, dim=1)
    c_n = F.normalize(c_cat, dim=1)
    logits = torch.matmul(feat_n, c_n.T) / temperature
    targets = torch.arange(B, device=c_cat.device)
    return F.cross_entropy(logits, targets)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class InfoGANTrainer:

    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] device={self.device}  dataset={cfg.dataset}  "
              f"mode={cfg.mode}  "
              f"(wgan_gp={cfg.use_wgan_gp}, infonce={cfg.use_infonce})")

        # load model module
        m = _load_model_module(cfg.dataset)
        self.Generator = m.Generator
        self.DiscriminatorQ = m.DiscriminatorQ
        self.sample_latent = m.sample_latent
        self.concat_latent = m.concat_latent
        self.parse_q_output = m.parse_q_output
        self.NOISE_DIM = m.NOISE_DIM
        self.CONT_DIM = m.CONT_DIM

        # Support multiple categorical codes
        self.CAT_DIMS = getattr(m, 'CAT_DIMS', (getattr(m, 'CAT_DIM', 10),))
        self.N_CATS = getattr(m, 'N_CATS', 1)
        self.CAT_DIM = sum(self.CAT_DIMS)

        # networks
        self.G = m.Generator().to(self.device)
        self.DQ = m.DiscriminatorQ().to(self.device)

        if cfg.use_wgan_gp:
            self.DQ.d_head = nn.Sequential(
                nn.Linear(1024, 1)
            ).to(self.device)

        # optimisers
        self.opt_G = torch.optim.Adam(
            self.G.parameters(),
            lr=cfg.lr_g, betas=(cfg.adam_beta1, cfg.adam_beta2),
        )
        self.opt_DQ = torch.optim.Adam(
            self.DQ.parameters(),
            lr=cfg.lr_d, betas=(cfg.adam_beta1, cfg.adam_beta2),
        )

        # data
        self.loader = build_loader(
            cfg.dataset, data_dir=cfg.data_dir, batch_size=cfg.batch_size
        )

        # logging
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{cfg.dataset}_{cfg.mode}_{ts}"
        self.writer = SummaryWriter(os.path.join(cfg.log_dir, run_name))
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        # fixed latents for visualisation
        self.fixed_noise, self.fixed_c_cat, self.fixed_c_cont = \
            self._make_fixed_latents()

        self.proj_feat = None
        self.proj_cat = None
        if cfg.use_infonce:
            feat_dim = 1024  # shared_fc 输出维度
            cat_dim = self.CAT_DIM  # 60 for chairs, 10 for mnist
            if feat_dim != cat_dim:
                self.proj_feat = nn.Linear(feat_dim, 256, bias=False).to(self.device)
                self.proj_cat = nn.Linear(cat_dim, 256, bias=False).to(self.device)
                # 加入优化器
                self.opt_DQ = torch.optim.Adam(
                    list(self.DQ.parameters()) + list(self.proj_feat.parameters()) + list(self.proj_cat.parameters()),
                    lr=cfg.lr_d, betas=(cfg.adam_beta1, cfg.adam_beta2),
                )

    # -----------------------------------------------------------------------
    # Fixed latents
    # -----------------------------------------------------------------------

    def _make_fixed_latents(self):
        B = 100
        device = self.device
        NOISE_DIM = self.NOISE_DIM
        CONT_DIM = self.CONT_DIM

        base = torch.FloatTensor(10, NOISE_DIM).uniform_(-1, 1)
        noise = base.repeat_interleave(10, dim=0).to(device)

        c_cat = torch.zeros(B, self.CAT_DIM, device=device)
        # Traverse the first categorical code, keep others fixed at class 0
        n_cols = min(self.CAT_DIMS[0], 10)
        for i in range(n_cols):
            c_cat[i * 10:(i + 1) * 10, i] = 1.0

        c_cont = torch.zeros(B, CONT_DIM, device=device)
        return noise, c_cat, c_cont

    # -----------------------------------------------------------------------
    # MI loss selector
    # -----------------------------------------------------------------------

    def _mi_loss(self, c_cat, cat_prob, c_cont, cont_mean, cont_std, feat=None):
        if self.cfg.use_infonce:
            assert feat is not None, "feat (trunk output) required for InfoNCE"
            mi_disc = mi_infonce_discrete(
                c_cat, feat, self.cfg.infonce_temp,
                proj_feat=self.proj_feat, proj_cat=self.proj_cat
            )
        else:
            mi_disc = mi_orig_discrete(c_cat, cat_prob, cat_dims=self.CAT_DIMS)
        mi_cont = mi_orig_continuous(c_cont, cont_mean, cont_std)
        return mi_disc, mi_cont

    # -----------------------------------------------------------------------
    # Single training step
    # -----------------------------------------------------------------------

    def _step(self, real_imgs):
        cfg = self.cfg
        device = self.device
        B = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        z_noise, c_cat, c_cont = self.sample_latent(B, device)
        z = self.concat_latent(z_noise, c_cat, c_cont)

        fake_imgs = self.G(z)

        # D / Q update
        self.opt_DQ.zero_grad()
        real_d, _ = self.DQ(real_imgs)
        fake_d, q_out = self.DQ(fake_imgs.detach())
        cat_prob, cont_mean, cont_std = self.parse_q_output(q_out)

        if cfg.use_infonce:
            with torch.no_grad():
                _conv = self.DQ.shared_conv(fake_imgs.detach())
            fake_feat = self.DQ.shared_fc(_conv)

        if cfg.use_wgan_gp:
            d_loss = (wgan_d_loss(real_d, fake_d)
                      + cfg.lambda_gp * gradient_penalty(
                          self.DQ, real_imgs, fake_imgs.detach(), device))
        else:
            d_loss = bce_d_loss(real_d, fake_d)

        if cfg.lambda_disc > 0 or cfg.lambda_cont > 0:
            mi_disc, mi_cont = self._mi_loss(c_cat, cat_prob, c_cont,
                                             cont_mean, cont_std,
                                             feat=fake_feat if cfg.use_infonce else None)
            mi_total = cfg.lambda_disc * mi_disc + cfg.lambda_cont * mi_cont
            (d_loss + mi_total).backward()
            mi_disc_val = mi_disc
        else:
            d_loss.backward()
            with torch.no_grad():
                mi_disc_val, mi_cont = self._mi_loss(c_cat, cat_prob, c_cont,
                                                     cont_mean, cont_std)
        self.opt_DQ.step()

        # G update
        self.opt_G.zero_grad()
        fake_imgs_g = self.G(z)
        fake_d_g, q_out_g = self.DQ(fake_imgs_g)
        cat_prob_g, cont_mean_g, cont_std_g = self.parse_q_output(q_out_g)
        if cfg.use_infonce:
            _conv_g = self.DQ.shared_conv(fake_imgs_g)
            fake_feat_g = self.DQ.shared_fc(_conv_g)

        g_loss = wgan_g_loss(fake_d_g) if cfg.use_wgan_gp else bce_g_loss(fake_d_g)

        if cfg.lambda_disc > 0 or cfg.lambda_cont > 0:
            mi_disc_g, mi_cont_g = self._mi_loss(c_cat, cat_prob_g, c_cont,
                                                 cont_mean_g, cont_std_g,
                                                 feat=fake_feat_g if cfg.use_infonce else None)
            mi_total_g = cfg.lambda_disc * mi_disc_g + cfg.lambda_cont * mi_cont_g
            (g_loss + mi_total_g).backward()
        else:
            g_loss.backward()
        self.opt_G.step()

        with torch.no_grad():
            mi_val = mi_disc_val.item()
            li_disc = math.log(self.CAT_DIMS[0]) - mi_val if math.isfinite(mi_val) else 0.0

        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'mi_disc': mi_disc_val.item(),
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
        n_cols = min(self.CAT_DIMS[0], 10)
        grid = make_grid(imgs, nrow=n_cols, normalize=True, value_range=(0, 1))
        self.writer.add_image('traversal/c1_category', grid, epoch)

        NOISE_DIM = self.NOISE_DIM
        CONT_DIM = self.CONT_DIM
        device = self.device

        # Fixed first categorical code to class 0, sweep continuous codes
        c0 = torch.zeros(10, self.CAT_DIM, device=device)
        c0[:, 0] = 1.0
        zn = torch.zeros(10, NOISE_DIM, device=device)

        for ci in range(CONT_DIM):
            sweep = torch.linspace(-2, 2, 10, device=device)
            c_cont = torch.zeros(10, CONT_DIM, device=device)
            c_cont[:, ci] = sweep
            imgs_ci = self.G(self.concat_latent(zn, c0, c_cont))
            label = f'c{ci + 2}_cont' if CONT_DIM > 1 else 'c_cont'
            self.writer.add_image(f'traversal/{label}',
                                  make_grid(imgs_ci, nrow=10, normalize=True, value_range=(0, 1)),
                                  epoch)

        self.G.train()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    def train(self, start_epoch=0):
        cfg = self.cfg
        for epoch in range(start_epoch, cfg.max_epochs):
            self.G.train()
            self.DQ.train()
            totals = {k: 0.0 for k in
                      ['d_loss', 'g_loss', 'mi_disc', 'mi_cont', 'LI_disc']}
            data_it = iter(self.loader)
            n_steps = cfg.updates_per_epoch if cfg.updates_per_epoch > 0 else len(self.loader)
            pbar = tqdm(range(n_steps),
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

            n = n_steps
            avg = {k: v / n for k, v in totals.items()}
            print(f"Epoch {epoch:03d} | "
                  f"D={avg['d_loss']:.4f}  G={avg['g_loss']:.4f}  "
                  f"MI={avg['mi_disc']:.4f}  LI={avg['LI_disc']:.4f}  "
                  f"target≈{math.log(self.CAT_DIMS[0]):.3f}")
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
        tag = 'final' if final else f'epoch{epoch:03d}'
        path = os.path.join(self.cfg.checkpoint_dir,
                            f"{self.cfg.dataset}_{self.cfg.mode}_{tag}.pt")
        torch.save({
            'epoch': epoch,
            'mode': self.cfg.mode,
            'dataset': self.cfg.dataset,
            'G_state': self.G.state_dict(),
            'DQ_state': self.DQ.state_dict(),
            'opt_G_state': self.opt_G.state_dict(),
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