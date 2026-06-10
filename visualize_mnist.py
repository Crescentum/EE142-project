"""
Visualize vanilla InfoGAN results on MNIST.

Figures produced:
  Figure 2: (a) Varying c1 — digit type
            (b) Varying c1 on GAN baseline — no structure [optional]
            (c) Varying c2 — continuous code sweep
            (d) Varying c3 — continuous code sweep
  Figure 1: L_I convergence curve (InfoGAN vs GAN baseline)
  Classification error: Section 7.2, target ~5%

Usage:
    # Figure 2 (a)(c)(d) + classification error
    python visualize_mnist.py --ckpt checkpoints/mnist_vanilla_final.pt

    # Full Figure 2 with GAN baseline subplot (b)
    python visualize_mnist.py \
        --ckpt     checkpoints/mnist_vanilla_final.pt \
        --ckpt_gan checkpoints/mnist_vanilla_gan_final.pt
"""

import os
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

import torch
from torchvision.utils import make_grid

import model_mnist as m
from datasets import denorm

TINY = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy_img(tensor):
    img = tensor.cpu().clamp(0, 1).numpy()
    return img[0] if img.shape[0] == 1 else img.transpose(1, 2, 0)

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _render_grid(ax, imgs, n_cols, title):
    grid   = make_grid(imgs, nrow=n_cols, normalize=False, padding=2, pad_value=0)
    img_np = _to_numpy_img(grid)
    ax.imshow(img_np, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(title, fontsize=10, pad=4)
    ax.axis('off')

def _load_G(ckpt_path, device):
    G    = m.Generator().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt['G_state'])
    G.eval()
    print(f"Loaded: epoch={ckpt['epoch']}  mode={ckpt.get('mode','vanilla')}  dataset=mnist")
    return G, ckpt

def _load_DQ(ckpt_path, device):
    DQ   = m.DiscriminatorQ().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    DQ.load_state_dict(ckpt['DQ_state'])
    DQ.eval()
    return DQ


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------

@torch.no_grad()
def _make_c1_grid(G, device, n_rows=5, n_cols=10):
    """Vary c1 across columns, fix z_noise per row."""
    row_noises = torch.FloatTensor(n_rows, m.NOISE_DIM).uniform_(-1, 1).to(device)
    z_noise    = row_noises.repeat_interleave(n_cols, dim=0)
    c_cat      = torch.zeros(n_rows * n_cols, m.CAT_DIM, device=device)
    for row in range(n_rows):
        for col in range(n_cols):
            c_cat[row * n_cols + col, col] = 1.0
    c_cont = torch.zeros(n_rows * n_cols, m.CONT_DIM, device=device)
    return denorm(G(m.concat_latent(z_noise, c_cat, c_cont)), '01')

@torch.no_grad()
def _make_cont_grid(G, device, cont_idx=0, cont_range=2.0, n_rows=5, n_cols=10):
    """Sweep one continuous code across columns, fix z_noise per row."""
    row_noises = torch.FloatTensor(n_rows, m.NOISE_DIM).uniform_(-1, 1).to(device)
    z_noise    = row_noises.repeat_interleave(n_cols, dim=0)
    c_cat      = torch.zeros(n_rows * n_cols, m.CAT_DIM, device=device)
    c_cat[:, 0] = 1.0
    sweep  = torch.linspace(-cont_range, cont_range, n_cols, device=device)
    c_cont = torch.zeros(n_rows * n_cols, m.CONT_DIM, device=device)
    for row in range(n_rows):
        for col in range(n_cols):
            c_cont[row * n_cols + col, cont_idx] = sweep[col]
    return denorm(G(m.concat_latent(z_noise, c_cat, c_cont)), '01')


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_figure2(G_infogan, device, G_gan=None,
                 cont_range=2.0, n_rows=5, n_cols=10,
                 save_path='results/mnist_vanilla_figure2.png'):
    _ensure_dir(save_path)
    has_baseline = G_gan is not None
    n_plots      = 4 if has_baseline else 3
    fig, axes    = plt.subplots(1, n_plots,
                                figsize=(n_plots * 3.5, n_rows * 0.7 + 1.2))
    if n_plots == 1:
        axes = [axes]

    _render_grid(axes[0], _make_c1_grid(G_infogan, device, n_rows, n_cols),
                 n_cols, '(a) Varying $c_1$ on InfoGAN\n(Digit type)')

    if has_baseline:
        _render_grid(axes[1], _make_c1_grid(G_gan, device, n_rows, n_cols),
                     n_cols, '(b) Varying $c_1$ on GAN\n(No clear meaning)')
        c_idx = 2
    else:
        c_idx = 1

    lc = 'c' if has_baseline else 'b'
    ld = 'd' if has_baseline else 'c'

    _render_grid(axes[c_idx],
                 _make_cont_grid(G_infogan, device, 0, cont_range, n_rows, n_cols),
                 n_cols, f'({lc}) Varying $c_2$ ∈ [−{cont_range}, {cont_range}]')
    _render_grid(axes[c_idx+1],
                 _make_cont_grid(G_infogan, device, 1, cont_range, n_rows, n_cols),
                 n_cols, f'({ld}) Varying $c_3$ ∈ [−{cont_range}, {cont_range}]')

    plt.suptitle('Manipulating latent codes on MNIST', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Figure 2 saved → {save_path}")


# ---------------------------------------------------------------------------
# Figure 1 — L_I convergence curve
# ---------------------------------------------------------------------------

def plot_mi_curve(li_infogan, li_baseline,
                  save_path='results/mnist_figure1_mi.png'):
    """
    li_infogan  : list of per-epoch LI values from InfoGAN run
    li_baseline : list of per-epoch LI values from GAN baseline run
                  (trained with lambda_disc=0, lambda_cont=0)
    """
    _ensure_dir(save_path)
    target = math.log(m.CAT_DIM)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(li_infogan,  color='#185FA5', linewidth=2,   label='InfoGAN')
    ax.plot(li_baseline, color='#2E8B2E', linewidth=1.5, label='GAN')
    ax.axhline(target, color='#E24B4A', linewidth=1, linestyle='--',
               label=f'$H(c)$ = log(10) ≈ {target:.3f}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('$\\mathcal{L}_I$', fontsize=13)
    ax.set_title('Mutual Information Lower Bound $\\mathcal{L}_I$ over Training',
                 fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Figure 1 saved → {save_path}")


# ---------------------------------------------------------------------------
# Classification error  (Section 7.2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_classification_error(G, DQ, device, n_samples_per_class=256):
    """
    Hungarian-matched classification error on generated images.
    Paper Section 7.2, target ~5%.
    """
    G.eval(); DQ.eval()
    all_pred, all_label = [], []

    for k in range(m.CAT_DIM):
        z_noise = torch.FloatTensor(n_samples_per_class, m.NOISE_DIM) \
                      .uniform_(-1, 1).to(device)
        c_cat   = torch.zeros(n_samples_per_class, m.CAT_DIM, device=device)
        c_cat[:, k] = 1.0
        c_cont  = torch.zeros(n_samples_per_class, m.CONT_DIM, device=device)
        z       = m.concat_latent(z_noise, c_cat, c_cont)

        _, q_out       = DQ(G(z))
        cat_prob, _, _ = m.parse_q_output(q_out)
        predicted      = cat_prob.argmax(dim=1).cpu().numpy()

        all_pred.extend(predicted.tolist())
        all_label.extend([k] * n_samples_per_class)

    all_pred  = np.array(all_pred)
    all_label = np.array(all_label)

    cost = np.zeros((m.CAT_DIM, m.CAT_DIM), dtype=np.int64)
    for p, l in zip(all_pred, all_label):
        cost[p, l] += 1

    row_ind, col_ind = linear_sum_assignment(-cost)
    correct = cost[row_ind, col_ind].sum()
    total   = len(all_pred)
    error   = 1.0 - correct / total
    print(f"  Classification error: {error*100:.2f}%  "
          f"({correct}/{total} correct after Hungarian matching)  "
          f"target: ~5%")
    return error


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='MNIST InfoGAN visualization')
    p.add_argument('--ckpt',     required=True)
    p.add_argument('--ckpt_gan', default=None,
                   help='GAN baseline checkpoint for subplot (b)')
    p.add_argument('--out_dir',  default='results')
    p.add_argument('--device',   default='cuda')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    G,  ckpt = _load_G(args.ckpt, device)
    DQ       = _load_DQ(args.ckpt, device)

    G_gan = None
    if args.ckpt_gan:
        G_gan, _ = _load_G(args.ckpt_gan, device)

    plot_figure2(G, device, G_gan=G_gan,
                 save_path=f'{args.out_dir}/mnist_vanilla_figure2.png')
    compute_classification_error(G, DQ, device)


if __name__ == '__main__':
    main()