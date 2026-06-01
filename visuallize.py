"""
visualize.py — All visualisations needed for the IEEE report.

Functions:
    plot_latent_traversal   : 10×10 grid varying one latent code at a time
                              → reproduces Figure 2 of the paper
    plot_mi_curve           : L_I convergence curve (InfoGAN vs GAN baseline)
                              → reproduces Figure 1 of the paper
    plot_loss_curves        : D/G loss over training (diagnostic)
    compute_classification_error : cluster c1 → digit, report error rate
                              → reproduces the "5% error" claim in Section 7.2
    save_comparison_grid    : side-by-side real vs generated images

Usage:
    from visualize import plot_latent_traversal, compute_classification_error
    plot_latent_traversal(G, device, save_path='results/traversal.png')
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')           # no display needed on cluster
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from models import (
    Generator, DiscriminatorQ,
    sample_latent, concat_latent, parse_q_output,
    NOISE_DIM, CAT_DIM, CONT_DIM,
)
from datasets import DATASET_CFG, denorm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy_img(tensor: torch.Tensor) -> np.ndarray:
    """(C,H,W) tensor in [0,1] → (H,W,C) or (H,W) numpy array."""
    img = tensor.cpu().clamp(0, 1).numpy()
    if img.shape[0] == 1:
        return img[0]           # grayscale → (H,W)
    return img.transpose(1, 2, 0)   # RGB → (H,W,C)


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Latent traversal grid  (Figure 2 in the paper)
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_latent_traversal(
    G           : Generator,
    device      : torch.device,
    dataset     : str  = 'mnist',
    n_rows      : int  = 10,        # one row per category / sweep step
    n_cols      : int  = 10,        # samples per row (different z_noise)
    cont_range  : float = 2.0,      # sweep continuous codes from -range to +range
    save_path   : str  = 'results/traversal.png',
    pixel_range : str  = '01',
):
    """
    Reproduce Figure 2 of the InfoGAN paper.

    Generates four sub-figures:
      (a) Varying c1 (categorical) — each row = one category
      (b) Varying c2 (continuous)  — left-to-right sweep from -range to +range
      (c) Varying c3 (continuous)  — left-to-right sweep from -range to +range
      (d) GAN baseline (same z, no MI regularisation effect visible)

    All other latent variables are fixed within each sub-figure.
    """
    _ensure_dir(save_path)
    G.eval()
    meta = DATASET_CFG[dataset]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('InfoGAN latent traversal', fontsize=14, y=1.01)

    # ── (a) vary c1 — categorical ──────────────────────────────────────────
    fixed_noise = torch.FloatTensor(n_cols, NOISE_DIM).uniform_(-1, 1).to(device)
    # same noise for all rows, only c1 differs
    z_noise = fixed_noise.repeat(n_rows, 1)        # (100, 62)

    c_cat = torch.zeros(n_rows * n_cols, CAT_DIM, device=device)
    for row in range(n_rows):
        c_cat[row * n_cols:(row + 1) * n_cols, row] = 1.0

    c_cont = torch.zeros(n_rows * n_cols, CONT_DIM, device=device)
    z = concat_latent(z_noise, c_cat, c_cont)
    imgs_cat = G(z)
    imgs_cat = denorm(imgs_cat, pixel_range)

    _plot_grid(axes[0], imgs_cat, n_rows, n_cols, pixel_range,
               title='(a) Varying $c_1$ — digit type')

    # ── (b) vary c2 — continuous (e.g. rotation) ──────────────────────────
    # Fix z_noise and c1 (class 0), sweep c2
    base_noise = torch.FloatTensor(1, NOISE_DIM).uniform_(-1, 1).to(device)
    z_noise_b  = base_noise.expand(n_rows * n_cols, -1)

    c_cat_b    = torch.zeros(n_rows * n_cols, CAT_DIM, device=device)
    c_cat_b[:, 0] = 1.0                           # all class-0

    c2_vals    = torch.linspace(-cont_range, cont_range, n_cols, device=device)
    c_cont_b   = torch.zeros(n_rows * n_cols, CONT_DIM, device=device)
    # different z_noise per row (to show generalisation across shapes)
    row_noises = torch.FloatTensor(n_rows, NOISE_DIM).uniform_(-1, 1).to(device)
    z_noise_b2 = row_noises.repeat_interleave(n_cols, dim=0)
    for row in range(n_rows):
        c_cont_b[row * n_cols:(row + 1) * n_cols, 0] = c2_vals   # sweep c2

    z_b = concat_latent(z_noise_b2, c_cat_b, c_cont_b)
    imgs_c2 = G(z_b)
    imgs_c2 = denorm(imgs_c2, pixel_range)
    _plot_grid(axes[1], imgs_c2, n_rows, n_cols, pixel_range,
               title=f'(b) Varying $c_2$ ∈ [{-cont_range}, {cont_range}] — rotation')

    # ── (c) vary c3 — continuous (e.g. width) ─────────────────────────────
    c_cont_c   = torch.zeros(n_rows * n_cols, CONT_DIM, device=device)
    for row in range(n_rows):
        c_cont_c[row * n_cols:(row + 1) * n_cols, 1] = c2_vals   # sweep c3
    z_c = concat_latent(z_noise_b2, c_cat_b, c_cont_c)
    imgs_c3 = G(z_c)
    imgs_c3 = denorm(imgs_c3, pixel_range)
    _plot_grid(axes[2], imgs_c3, n_rows, n_cols, pixel_range,
               title=f'(c) Varying $c_3$ ∈ [{-cont_range}, {cont_range}] — width')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Latent traversal saved → {save_path}")


def _plot_grid(ax, imgs: torch.Tensor, n_rows: int, n_cols: int,
               pixel_range: str, title: str):
    """Helper: render a (n_rows*n_cols, C, H, W) tensor as a grid on ax."""
    grid = make_grid(imgs, nrow=n_cols, normalize=False, padding=2)
    img_np = _to_numpy_img(grid)
    cmap = 'gray' if img_np.ndim == 2 else None
    ax.imshow(img_np, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.axis('off')


# ---------------------------------------------------------------------------
# 2. MI convergence curve  (Figure 1 in the paper)
# ---------------------------------------------------------------------------

def plot_mi_curve(
    li_infogan  : list,     # L_I values per epoch for InfoGAN
    li_baseline : list,     # L_I values per epoch for plain GAN (no MI reg)
    save_path   : str = 'results/mi_curve.png',
):
    """
    Reproduce Figure 1: L_I lower bound over training iterations.

    Args:
        li_infogan  : list of average L_I per epoch from InfoGAN training
        li_baseline : list from a GAN trained without MI regularisation
        save_path   : output PNG path
    """
    _ensure_dir(save_path)
    target = math.log(CAT_DIM)       # H(c1) = log(10) ≈ 2.303

    epochs = range(len(li_infogan))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, li_infogan,  label='InfoGAN',  color='#185FA5', linewidth=2)
    ax.plot(epochs, li_baseline, label='GAN (baseline)',
            color='#888780', linewidth=1.5, linestyle='--')
    ax.axhline(target, color='#E24B4A', linewidth=1, linestyle=':',
               label=f'H(c) = log(10) ≈ {target:.3f}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('$\\mathcal{L}_I$ (lower bound)', fontsize=12)
    ax.set_title('Mutual information lower bound over training', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=min(min(li_infogan), min(li_baseline)) - 0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  MI curve saved → {save_path}")


# ---------------------------------------------------------------------------
# 3. Training loss curves (diagnostic)
# ---------------------------------------------------------------------------

def plot_loss_curves(
    history   : dict,       # {'d_loss': [...], 'g_loss': [...], 'LI_disc': [...]}
    mode      : str = 'vanilla',
    save_path : str = 'results/loss_curves.png',
):
    """
    Plot D loss, G loss, and L_I over epochs for one training run.
    Pass the history dict that trainer.py accumulates.
    """
    _ensure_dir(save_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Training curves — mode: {mode}', fontsize=13)

    keys_titles = [
        ('d_loss',  'Discriminator loss'),
        ('g_loss',  'Generator loss'),
        ('LI_disc', '$\\mathcal{L}_I$ (MI lower bound)'),
    ]
    colours = ['#E24B4A', '#1D9E75', '#185FA5']

    for ax, (key, title), col in zip(axes, keys_titles, colours):
        if key not in history:
            ax.set_visible(False)
            continue
        vals = history[key]
        ax.plot(vals, color=col, linewidth=1.5)
        if key == 'LI_disc':
            ax.axhline(math.log(CAT_DIM), color='#888780',
                       linestyle='--', linewidth=1,
                       label=f'target ≈ {math.log(CAT_DIM):.3f}')
            ax.legend(fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Loss curves saved → {save_path}")


# ---------------------------------------------------------------------------
# 4. Classification error rate  (Section 7.2 claim: ~5% on MNIST)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_classification_error(
    G          : Generator,
    DQ         : DiscriminatorQ,
    dataloader,             # MNIST test loader
    device     : torch.device,
    n_batches  : int = 20,
) -> float:
    """
    Evaluate how well c1 (categorical code) recovers digit identity.

    Method (paper Section 7.2):
      1. For each real image x, forward through Q to get the predicted
         category argmax(Q(c1|x)).
      2. Build a confusion matrix between predicted cluster and true label.
      3. Use the Hungarian algorithm to find the optimal cluster→digit
         assignment (since c1 categories are inherently unordered).
      4. Report classification error = 1 - accuracy after optimal assignment.

    Returns:
        error_rate : float in [0, 1]
    """
    G.eval()
    DQ.eval()

    all_pred  = []
    all_label = []

    for i, (imgs, labels) in enumerate(dataloader):
        if i >= n_batches:
            break
        imgs = imgs.to(device)
        _, q_out = DQ(imgs)
        cat_prob, _, _ = parse_q_output(q_out)
        pred_cluster = cat_prob.argmax(dim=1).cpu().numpy()
        all_pred.extend(pred_cluster.tolist())
        all_label.extend(labels.numpy().tolist())

    all_pred  = np.array(all_pred)
    all_label = np.array(all_label)

    n_classes = CAT_DIM   # 10
    # Build cost matrix (negate count for minimisation)
    cost = np.zeros((n_classes, n_classes), dtype=np.int64)
    for p, l in zip(all_pred, all_label):
        if l < n_classes:
            cost[p, l] += 1

    # Hungarian algorithm: maximise total count = minimise -count
    row_ind, col_ind = linear_sum_assignment(-cost)
    correct = cost[row_ind, col_ind].sum()
    total   = len(all_pred)
    error   = 1.0 - correct / total

    print(f"  Classification error: {error*100:.2f}%  "
          f"({correct}/{total} correct after optimal cluster assignment)")
    return error


# ---------------------------------------------------------------------------
# 5. Mode comparison grid  (vanilla vs wgan_gp vs infonce)
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_mode_comparison(
    generators  : dict,     # {'vanilla': G1, 'wgan_gp': G2, 'infonce': G3}
    device      : torch.device,
    pixel_range : str  = '01',
    n_samples   : int  = 10,
    save_path   : str  = 'results/mode_comparison.png',
):
    """
    Side-by-side comparison of generated samples from different training modes.
    Each row = one mode; columns = samples with the same latent code.
    """
    _ensure_dir(save_path)

    z_noise, c_cat, c_cont = sample_latent(n_samples, device)
    z = concat_latent(z_noise, c_cat, c_cont)

    n_modes = len(generators)
    fig, axes = plt.subplots(n_modes, 1,
                             figsize=(n_samples * 0.8, n_modes * 1.2))
    if n_modes == 1:
        axes = [axes]

    for ax, (mode_name, G) in zip(axes, generators.items()):
        G.eval()
        imgs = G(z)
        imgs = denorm(imgs, pixel_range)
        grid = make_grid(imgs, nrow=n_samples, normalize=False, padding=2)
        img_np = _to_numpy_img(grid)
        cmap   = 'gray' if img_np.ndim == 2 else None
        ax.imshow(img_np, cmap=cmap, vmin=0, vmax=1)
        ax.set_ylabel(mode_name, fontsize=11, rotation=0,
                      labelpad=60, va='center')
        ax.axis('off')

    fig.suptitle('Generated samples — same latent code across modes', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Mode comparison saved → {save_path}")


# ---------------------------------------------------------------------------
# 6. Ablation: lambda sweep  (how λ affects disentanglement)
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_lambda_ablation(
    generators  : dict,     # {'λ=0.1': G1, 'λ=1.0': G2, 'λ=5.0': G3}
    device      : torch.device,
    pixel_range : str = '01',
    save_path   : str = 'results/lambda_ablation.png',
):
    """
    Show how the MI regularisation weight λ affects disentanglement.
    Generates the c1-traversal grid for each λ value side by side.
    """
    _ensure_dir(save_path)
    n_models = len(generators)

    fixed_noise = torch.FloatTensor(10, NOISE_DIM).uniform_(-1, 1).to(device)
    z_noise = fixed_noise.repeat_interleave(10, dim=0)
    c_cat   = torch.zeros(100, CAT_DIM, device=device)
    for i in range(10):
        c_cat[i*10:(i+1)*10, i] = 1.0
    c_cont = torch.zeros(100, CONT_DIM, device=device)
    z = concat_latent(z_noise, c_cat, c_cont)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (label, G) in zip(axes, generators.items()):
        G.eval()
        imgs = G(z)
        imgs = denorm(imgs, pixel_range)
        _plot_grid(ax, imgs, 10, 10, pixel_range,
                   title=f'{label}')

    fig.suptitle('Effect of λ on disentanglement (c₁ traversal)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Lambda ablation saved → {save_path}")


# ---------------------------------------------------------------------------
# Convenience: load a checkpoint and run all visualisations at once
# ---------------------------------------------------------------------------

def run_all(
    checkpoint_path : str,
    dataset         : str  = 'mnist',
    results_dir     : str  = 'results',
    device_str      : str  = 'cuda',
):
    """
    Load a saved checkpoint and generate all figures for the report.

    Example:
        python visualize.py --ckpt checkpoints/mnist_vanilla_final.pt
    """
    from datasets import build_loader

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    meta   = DATASET_CFG[dataset]

    G  = Generator().to(device)
    DQ = DiscriminatorQ().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt['G_state'])
    DQ.load_state_dict(ckpt['DQ_state'])
    mode = ckpt.get('mode', 'vanilla')
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}  mode={mode}")

    os.makedirs(results_dir, exist_ok=True)

    # 1. Latent traversal
    plot_latent_traversal(
        G, device,
        dataset     = dataset,
        pixel_range = meta.pixel_range,
        save_path   = f'{results_dir}/{dataset}_{mode}_traversal.png',
    )

    # 2. Classification error (MNIST only)
    if dataset == 'mnist':
        test_loader = build_loader(dataset, batch_size=128,
                                   split='test', num_workers=2)
        err = compute_classification_error(G, DQ, test_loader, device)
        print(f"  → Classification error: {err*100:.2f}%  (paper target: ~5%)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',    required=True, help='path to .pt checkpoint')
    p.add_argument('--dataset', default='mnist',
                   choices=['mnist', 'svhn', 'celeba'])
    p.add_argument('--out_dir', default='results')
    p.add_argument('--device',  default='cuda')
    args = p.parse_args()

    run_all(args.ckpt, args.dataset, args.out_dir, args.device)