"""
visualize.py — Reproduce all figures needed for the IEEE report.

Figure 2 (4 subplots, exact match to paper):
  (a) Varying c1 on InfoGAN  — each column = one digit category
  (b) Varying c1 on GAN      — no clear structure (needs GAN baseline ckpt)
  (c) Varying c2 on InfoGAN  — rotation sweep -2 to 2
  (d) Varying c3 on InfoGAN  — width sweep -2 to 2

Figure 1:
  L_I convergence curve: InfoGAN (blue) vs GAN baseline (green)

Other utilities:
  compute_classification_error  — Section 7.2, target ~5%
  plot_loss_curves              — training diagnostics
  plot_mode_comparison          — vanilla vs wgan_gp vs infonce

Usage:
    # Generate Figure 2 (a)(c)(d) from InfoGAN checkpoint only:
    python visualize.py --ckpt checkpoints/mnist_vanilla_final.pt --dataset mnist

    # Generate full Figure 2 including (b) GAN baseline:
    python visualize.py --ckpt checkpoints/mnist_vanilla_final.pt \
                        --ckpt_gan checkpoints/mnist_gan_baseline_final.pt \
                        --dataset mnist
"""

import os
import math
import importlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from datasets import DATASET_CFG, denorm, build_loader

TINY = 1e-8


# ---------------------------------------------------------------------------
# Dynamic model import (same convention as trainer.py)
# ---------------------------------------------------------------------------

def _load_model_module(dataset: str):
    module_name = f"model_{dataset}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"Cannot find '{module_name}.py'. "
            f"Make sure model_{dataset}.py is in the same directory."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy_img(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.cpu().clamp(0, 1).numpy()
    if img.shape[0] == 1:
        return img[0]               # grayscale → (H, W)
    return img.transpose(1, 2, 0)  # RGB → (H, W, C)


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _render_grid(ax, imgs: torch.Tensor, n_cols: int, title: str):
    """Render a batch of images as a grid on a matplotlib axis."""
    grid    = make_grid(imgs, nrow=n_cols, normalize=False, padding=2,
                        pad_value=0)
    img_np  = _to_numpy_img(grid)
    cmap    = 'gray' if img_np.ndim == 2 else None
    ax.imshow(img_np, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(title, fontsize=10, pad=4)
    ax.axis('off')


def _num_cat_codes(m) -> int:
    return getattr(m, 'N_CATS', 1)


def _cat_code_width(m) -> int:
    return _num_cat_codes(m) * m.CAT_DIM


# ---------------------------------------------------------------------------
# Core grid builder — exact layout matching paper Figure 2
#
# Layout convention (from paper caption):
#   - Each COLUMN = one fixed value of the varying code
#   - Each ROW    = different fixed z_noise / other codes
#
# (a) c1 categorical:
#     10 columns = 10 categories, 5 rows = 5 different z_noise samples
#     → 5×10 = 50 images
#
# (c)(d) continuous sweep:
#     10 columns = sweep from -2 to +2, 5 rows = 5 different z_noise samples
# ---------------------------------------------------------------------------

@torch.no_grad()
def _make_c1_grid(G, m, device, pixel_range, n_rows=5, n_cols=10):
    """
    (a)/(b): Fix z_noise per row, vary c1 across columns.
    Each column = one categorical value → should show one digit type.
    """
    NOISE_DIM = m.NOISE_DIM
    CAT_DIM   = m.CAT_DIM
    CONT_DIM  = m.CONT_DIM
    N_CATS    = _num_cat_codes(m)

    # one z_noise vector per row, repeated across all 10 columns
    row_noises = torch.FloatTensor(n_rows, NOISE_DIM).uniform_(-1, 1).to(device)
    z_noise    = row_noises.repeat_interleave(n_cols, dim=0)  # (n_rows*n_cols, 62)

    # c1: column j → category j
    # Layout: indices 0..n_cols-1 = row 0, n_cols..2*n_cols-1 = row 1, etc.
    # (because repeat_interleave repeats each row n_cols times consecutively)
    c_cat = torch.zeros(n_rows * n_cols, _cat_code_width(m), device=device)
    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            c_cat[idx, col] = 1.0
            for code_idx in range(1, N_CATS):
                c_cat[idx, code_idx * CAT_DIM] = 1.0

    c_cont = torch.zeros(n_rows * n_cols, CONT_DIM, device=device)
    z      = m.concat_latent(z_noise, c_cat, c_cont)
    imgs   = G(z)
    return denorm(imgs, pixel_range)


@torch.no_grad()
def _make_cont_grid(G, m, device, pixel_range, cont_idx=0,
                    cont_range=2.0, n_rows=5, n_cols=10):
    """
    (c)/(d): Fix z_noise per row and c1 (class 0), sweep one continuous code.
    Each column = one value of c_i swept from -cont_range to +cont_range.
    """
    NOISE_DIM = m.NOISE_DIM
    CAT_DIM   = m.CAT_DIM
    CONT_DIM  = m.CONT_DIM
    N_CATS    = _num_cat_codes(m)

    row_noises = torch.FloatTensor(n_rows, NOISE_DIM).uniform_(-1, 1).to(device)
    z_noise    = row_noises.repeat_interleave(n_cols, dim=0)

    c_cat = torch.zeros(n_rows * n_cols, _cat_code_width(m), device=device)
    c_cat[:, 0] = 1.0   # fix c1 to class 0
    for code_idx in range(1, N_CATS):
        c_cat[:, code_idx * CAT_DIM] = 1.0

    sweep  = torch.linspace(-cont_range, cont_range, n_cols, device=device)
    c_cont = torch.zeros(n_rows * n_cols, CONT_DIM, device=device)
    for row in range(n_rows):
        for col in range(n_cols):
            c_cont[row * n_cols + col, cont_idx] = sweep[col]

    z    = m.concat_latent(z_noise, c_cat, c_cont)
    imgs = G(z)
    return denorm(imgs, pixel_range)


# ---------------------------------------------------------------------------
# Figure 2 — main function
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_figure2(
    G_infogan,              # trained InfoGAN generator
    m,                      # model module (model_mnist etc.)
    device      : torch.device,
    pixel_range : str  = '01',
    G_gan       = None,     # optional: GAN baseline generator (for subplot b)
    cont_range  : float = 2.0,
    n_rows      : int   = 5,
    n_cols      : int   = 10,
    save_path   : str   = 'results/figure2.png',
):
    """
    Reproduce Figure 2 of the InfoGAN paper.

      (a) Varying c1 on InfoGAN  — columns = digit categories
      (b) Varying c1 on GAN      — no structure (requires G_gan)
      (c) Varying c2 on InfoGAN  — rotation
      (d) Varying c3 on InfoGAN  — width

    If G_gan is None, subplot (b) is skipped and only (a)(c)(d) are saved.
    """
    _ensure_dir(save_path)
    G_infogan.eval()

    has_baseline = G_gan is not None
    cont_plots = min(getattr(m, 'CONT_DIM', 0), 2)
    n_plots = 1 + int(has_baseline) + cont_plots

    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 3.5, n_rows * 0.7 + 1.2))
    if n_plots == 1:
        axes = [axes]

    # (a) InfoGAN c1 traversal
    imgs_a = _make_c1_grid(G_infogan, m, device, pixel_range, n_rows, n_cols)
    _render_grid(axes[0], imgs_a, n_cols,
                 '(a) Varying $c_1$ on InfoGAN')

    if has_baseline:
        # (b) GAN baseline c1 traversal — should look random
        G_gan.eval()
        imgs_b = _make_c1_grid(G_gan, m, device, pixel_range, n_rows, n_cols)
        _render_grid(axes[1], imgs_b, n_cols,
                     '(b) Varying $c_1$ on GAN\n(No clear meaning)')
        c_idx = 2   # (c) and (d) go to axes[2] and axes[3]
    else:
        c_idx = 1

    if cont_plots == 0:
        plt.suptitle('Manipulating categorical latent codes', fontsize=12, y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure 2 saved 鈫?{save_path}")
        return

    # (c) c2 sweep — rotation
    imgs_c = _make_cont_grid(G_infogan, m, device, pixel_range,
                              cont_idx=0, cont_range=cont_range,
                              n_rows=n_rows, n_cols=n_cols)
    _render_grid(axes[c_idx], imgs_c, n_cols,
                 f'({"c" if has_baseline else "b"}) Varying $c_2$ '
                 f'∈ [−{cont_range}, {cont_range}]\n(Rotation)')

    # (d) c3 sweep — width
    imgs_d = _make_cont_grid(G_infogan, m, device, pixel_range,
                              cont_idx=1, cont_range=cont_range,
                              n_rows=n_rows, n_cols=n_cols)
    _render_grid(axes[c_idx + 1], imgs_d, n_cols,
                 f'({"d" if has_baseline else "c"}) Varying $c_3$ '
                 f'∈ [−{cont_range}, {cont_range}]\n(Width)')

    plt.suptitle('Manipulating latent codes on MNIST', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Figure 2 saved → {save_path}")


# ---------------------------------------------------------------------------
# Figure 1 — L_I convergence curve
# ---------------------------------------------------------------------------

def plot_mi_curve(
    li_infogan  : list,
    li_baseline : list,
    save_path   : str = 'results/figure1_mi_curve.png',
    cat_dim     : int = 10,
):
    """
    Reproduce Figure 1: L_I lower bound over training.
    li_infogan  : list of per-epoch LI values from InfoGAN training log
    li_baseline : list of per-epoch LI values from GAN baseline (lambda=0)
    """
    _ensure_dir(save_path)
    target = math.log(cat_dim)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(li_infogan,  color='#185FA5', linewidth=2,   label='InfoGAN')
    ax.plot(li_baseline, color='#2E8B2E', linewidth=1.5, label='GAN')
    ax.axhline(target, color='#E24B4A', linewidth=1, linestyle='--',
               label=f'$H(c)$ = log(10) ≈ {target:.3f}')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('$\\mathcal{L}_I$', fontsize=13)
    ax.set_title('Mutual Information Lower Bound over Training', fontsize=12)
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
def compute_classification_error(
    G,
    DQ,
    device     : torch.device,
    m,
    n_samples_per_class : int = 256,
) -> float:
    """
    Hungarian-matched classification error using GENERATED images.

    Correct procedure (paper Section 7.2):
      1. For each category k, generate images with c1 = k
      2. Feed generated images through Q → get predicted cluster
      3. Hungarian-match clusters to categories
      4. error = 1 - accuracy after optimal assignment

    NOTE: Q is trained on generated images, NOT real images.
    Feeding real images to Q gives ~70% error (domain mismatch).
    Target: ~5% error on MNIST.
    """
    G.eval()
    DQ.eval()
    CAT_DIM   = m.CAT_DIM
    NOISE_DIM = m.NOISE_DIM
    CONT_DIM  = m.CONT_DIM

    all_pred, all_label = [], []

    for k in range(CAT_DIM):
        z_noise = torch.FloatTensor(n_samples_per_class, NOISE_DIM).uniform_(-1, 1).to(device)
        c_cat   = torch.zeros(n_samples_per_class, CAT_DIM, device=device)
        c_cat[:, k] = 1.0
        c_cont  = torch.zeros(n_samples_per_class, CONT_DIM, device=device)
        z       = m.concat_latent(z_noise, c_cat, c_cont)

        fake_imgs        = G(z)
        _, q_out         = DQ(fake_imgs)
        cat_prob, _, _   = m.parse_q_output(q_out)
        predicted        = cat_prob.argmax(dim=1).cpu().numpy()

        all_pred.extend(predicted.tolist())
        all_label.extend([k] * n_samples_per_class)

    all_pred  = np.array(all_pred)
    all_label = np.array(all_label)

    cost = np.zeros((CAT_DIM, CAT_DIM), dtype=np.int64)
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
# Training loss curves (diagnostic)
# ---------------------------------------------------------------------------

def plot_loss_curves(
    history   : dict,
    mode      : str = 'vanilla',
    save_path : str = 'results/loss_curves.png',
    cat_dim   : int = 10,
):
    _ensure_dir(save_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Training curves — {mode}', fontsize=13)
    specs = [('d_loss', 'D loss', '#E24B4A'),
             ('g_loss', 'G loss', '#1D9E75'),
             ('LI_disc','$\\mathcal{L}_I$', '#185FA5')]
    for ax, (key, title, col) in zip(axes, specs):
        if key not in history:
            ax.set_visible(False); continue
        ax.plot(history[key], color=col, linewidth=1.5)
        if key == 'LI_disc':
            ax.axhline(math.log(cat_dim), color='#888780', linestyle='--',
                       linewidth=1, label=f'target≈{math.log(cat_dim):.3f}')
            ax.legend(fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Loss curves saved → {save_path}")


# ---------------------------------------------------------------------------
# Mode comparison grid
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_mode_comparison(
    generators  : dict,
    m,
    device      : torch.device,
    pixel_range : str = '01',
    n_samples   : int = 10,
    save_path   : str = 'results/mode_comparison.png',
):
    """Side-by-side: same latent code fed to different training modes."""
    _ensure_dir(save_path)
    z_noise, c_cat, c_cont = m.sample_latent(n_samples, device)
    z = m.concat_latent(z_noise, c_cat, c_cont)

    n_modes = len(generators)
    fig, axes = plt.subplots(n_modes, 1,
                             figsize=(n_samples * 0.9, n_modes * 1.3))
    if n_modes == 1:
        axes = [axes]

    for ax, (name, G) in zip(axes, generators.items()):
        G.eval()
        imgs = denorm(G(z), pixel_range)
        grid = make_grid(imgs, nrow=n_samples, normalize=False, padding=2)
        img_np = _to_numpy_img(grid)
        cmap   = 'gray' if img_np.ndim == 2 else None
        ax.imshow(img_np, cmap=cmap, vmin=0, vmax=1)
        ax.set_ylabel(name, fontsize=10, rotation=0, labelpad=65, va='center')
        ax.axis('off')

    fig.suptitle('Generated samples — same latent code, different training modes',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Mode comparison saved → {save_path}")


# ---------------------------------------------------------------------------
# CLI — load checkpoint(s) and generate all figures
# ---------------------------------------------------------------------------

def run_all(
    ckpt_path     : str,
    dataset       : str = 'mnist',
    results_dir   : str = 'results',
    device_str    : str = 'cuda',
    ckpt_gan_path : str = None,   # optional GAN baseline for subplot (b)
):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    meta   = DATASET_CFG[dataset]
    m      = _load_model_module(dataset)

    os.makedirs(results_dir, exist_ok=True)

    # load InfoGAN checkpoint
    G  = m.Generator().to(device)
    DQ = m.DiscriminatorQ().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt['G_state'])
    DQ.load_state_dict(ckpt['DQ_state'])
    mode = ckpt.get('mode', 'vanilla')
    print(f"Loaded: epoch={ckpt['epoch']}  mode={mode}  dataset={dataset}")

    # optionally load GAN baseline
    G_gan = None
    if ckpt_gan_path:
        G_gan = m.Generator().to(device)
        ckpt_gan = torch.load(ckpt_gan_path, map_location=device)
        G_gan.load_state_dict(ckpt_gan['G_state'])
        print(f"Loaded GAN baseline: epoch={ckpt_gan['epoch']}")

    # Figure 2
    plot_figure2(
        G, m, device,
        pixel_range = meta.pixel_range,
        G_gan       = G_gan,
        save_path   = f'{results_dir}/{dataset}_{mode}_figure2.png',
    )

    # Classification error (MNIST only)
    if dataset == 'mnist':
        compute_classification_error(G, DQ, device, m)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',     required=True,
                   help='InfoGAN checkpoint (.pt)')
    p.add_argument('--ckpt_gan', default=None,
                   help='GAN baseline checkpoint for subplot (b) [optional]')
    p.add_argument('--dataset',  default='mnist',
                   choices=['mnist', 'svhn', 'celeba'])
    p.add_argument('--out_dir',  default='results')
    p.add_argument('--device',   default='cuda')
    args = p.parse_args()

    run_all(args.ckpt, args.dataset, args.out_dir,
            args.device, args.ckpt_gan)
