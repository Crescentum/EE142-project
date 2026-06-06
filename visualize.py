"""
visualize.py — All visualisations needed for the IEEE report.

Supports both MNIST (1 cat code, 2 cont codes, grayscale) and
SVHN (4 cat codes, 4 cont codes, RGB).

Functions:
    plot_latent_traversal   : 10×10 grid varying one latent code at a time
    plot_mi_curve           : L_I convergence curve (InfoGAN vs GAN baseline)
    plot_loss_curves        : D/G loss over training (diagnostic)
    compute_classification_error : cluster c1 → digit, report error rate
    save_comparison_grid    : side-by-side real vs generated images

Usage:
    from visualize import plot_latent_traversal, compute_classification_error
    plot_latent_traversal(G, DQ, device, dataset='svhn', save_path='results/traversal.png')
"""

import os
import math
import importlib
import numpy as np
import matplotlib
matplotlib.use('Agg')           # no display needed on cluster
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid


# ---------------------------------------------------------------------------
# Dynamic model import
# ---------------------------------------------------------------------------

def _load_model_module(dataset: str):
    module_name = f"model_{dataset}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Cannot find '{module_name}.py'")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy_img(tensor: torch.Tensor, value_range=(-1, 1)) -> np.ndarray:
    """(C,H,W) tensor → (H,W,C) or (H,W) numpy array in [0,1]."""
    # Normalize to [0,1] if needed
    low, high = value_range
    img = tensor.cpu().float().numpy()
    img = (img - low) / (high - low)
    img = np.clip(img, 0, 1)
    if img.shape[0] == 1:
        return img[0]           # grayscale → (H,W)
    return img.transpose(1, 2, 0)   # RGB → (H,W,C)


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)


def _get_value_range(dataset: str):
    """Return (min, max) pixel values for generated images."""
    # MNIST uses Sigmoid → [0,1]; SVHN uses Tanh → [-1,1]
    return (-1, 1) if dataset == 'svhn' else (0, 1)


# ---------------------------------------------------------------------------
# 1. Latent traversal grid  (Figure 2 / Figure 5 in the paper)
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_latent_traversal(
    G,
    device      : torch.device,
    dataset     : str  = 'mnist',
    n_rows      : int  = 10,
    n_cols      : int  = 10,
    cont_range  : float = 2.0,
    save_path   : str  = 'results/traversal.png',
):
    """
    Reproduce latent traversal figures from the InfoGAN paper.

    For MNIST: 1 categorical + 2 continuous codes.
    For SVHN:  4 categorical + 4 continuous codes (only first cat & first 2 cont shown).
    """
    _ensure_dir(save_path)
    G.eval()

    m = _load_model_module(dataset)
    NOISE_DIM  = m.NOISE_DIM
    CAT_DIM    = m.CAT_DIM
    CONT_DIM   = m.CONT_DIM
    N_CATS     = getattr(m, 'N_CATS', 1)
    concat_latent = m.concat_latent
    value_range = _get_value_range(dataset)

    # Determine number of subplots: 1 (cat) + min(2, CONT_DIM) cont traversals
    n_cont_show = min(2, CONT_DIM)
    n_subplots = 1 + n_cont_show

    fig, axes = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 6))
    if n_subplots == 1:
        axes = [axes]
    fig.suptitle(f'InfoGAN latent traversal ({dataset.upper()})', fontsize=14, y=1.01)

    # ── (a) vary first categorical code c1 ────────────────────────────────
    fixed_noise = torch.FloatTensor(n_cols, NOISE_DIM).uniform_(-1, 1).to(device)
    z_noise = fixed_noise.repeat(n_rows, 1)        # (100, NOISE_DIM)

    # Build c_cat: vary first categorical code, fix others to class 0
    c_cat = torch.zeros(n_rows * n_cols, N_CATS * CAT_DIM, device=device)
    for row in range(n_rows):
        c_cat[row * n_cols:(row + 1) * n_cols, row] = 1.0  # first code = row
        for j in range(1, N_CATS):
            c_cat[row * n_cols:(row + 1) * n_cols, j * CAT_DIM] = 1.0

    c_cont = torch.zeros(n_rows * n_cols, CONT_DIM, device=device)
    z = concat_latent(z_noise, c_cat, c_cont)
    imgs_cat = G(z)

    _plot_grid(axes[0], imgs_cat, n_rows, n_cols, value_range,
               title='(a) Varying $c_1$ — categorical')

    # ── (b, c) vary continuous codes ────────────────────────────────────
    base_noise = torch.FloatTensor(1, NOISE_DIM).uniform_(-1, 1).to(device)
    row_noises = torch.FloatTensor(n_rows, NOISE_DIM).uniform_(-1, 1).to(device)
    z_noise_b2 = row_noises.repeat_interleave(n_cols, dim=0)

    # Fix first categorical code to class 0, others to class 0
    c_cat_b = torch.zeros(n_rows * n_cols, N_CATS * CAT_DIM, device=device)
    c_cat_b[:, 0] = 1.0
    for j in range(1, N_CATS):
        c_cat_b[:, j * CAT_DIM] = 1.0

    sweep = torch.linspace(-cont_range, cont_range, n_cols, device=device)

    for ci in range(n_cont_show):
        c_cont_b = torch.zeros(n_rows * n_cols, CONT_DIM, device=device)
        for row in range(n_rows):
            c_cont_b[row * n_cols:(row + 1) * n_cols, ci] = sweep

        z_b = concat_latent(z_noise_b2, c_cat_b, c_cont_b)
        imgs_ci = G(z_b)

        labels = ['rotation', 'width', 'cont2', 'cont3']
        _plot_grid(axes[1 + ci], imgs_ci, n_rows, n_cols, value_range,
                   title=f'(b) Varying $c_{{ci+2}}$ ∈ [{-cont_range}, {cont_range}] — {labels[ci]}')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Latent traversal saved → {save_path}")


def _plot_grid(ax, imgs: torch.Tensor, n_rows: int, n_cols: int,
               value_range: tuple, title: str):
    """Helper: render a (n_rows*n_cols, C, H, W) tensor as a grid on ax."""
    low, high = value_range
    grid = make_grid(imgs, nrow=n_cols, normalize=True, value_range=value_range, padding=2)
    img_np = _to_numpy_img(grid, value_range=value_range)
    cmap = 'gray' if img_np.ndim == 2 else None
    ax.imshow(img_np, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.axis('off')


# ---------------------------------------------------------------------------
# 2. MI convergence curve  (Figure 1 in the paper)
# ---------------------------------------------------------------------------

def plot_mi_curve(
    li_infogan  : list,
    li_baseline : list,
    cat_dim     : int = 10,
    save_path   : str = 'results/mi_curve.png',
):
    """Reproduce Figure 1: L_I lower bound over training iterations."""
    _ensure_dir(save_path)
    target = math.log(cat_dim)
    epochs = range(len(li_infogan))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, li_infogan,  label='InfoGAN',  color='#185FA5', linewidth=2)
    ax.plot(epochs, li_baseline, label='GAN (baseline)',
            color='#888780', linewidth=1.5, linestyle='--')
    ax.axhline(target, color='#E24B4A', linewidth=1, linestyle=':',
               label=f'H(c) = log({cat_dim}) ≈ {target:.3f}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('$\mathcal{L}_I$ (lower bound)', fontsize=12)
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
    history   : dict,
    mode      : str = 'vanilla',
    cat_dim   : int = 10,
    save_path : str = 'results/loss_curves.png',
):
    """Plot D loss, G loss, and L_I over epochs."""
    _ensure_dir(save_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Training curves — mode: {mode}', fontsize=13)

    keys_titles = [
        ('d_loss',  'Discriminator loss'),
        ('g_loss',  'Generator loss'),
        ('LI_disc', '$\mathcal{L}_I$ (MI lower bound)'),
    ]
    colours = ['#E24B4A', '#1D9E75', '#185FA5']

    for ax, (key, title), col in zip(axes, keys_titles, colours):
        if key not in history:
            ax.set_visible(False)
            continue
        vals = history[key]
        ax.plot(vals, color=col, linewidth=1.5)
        if key == 'LI_disc':
            ax.axhline(math.log(cat_dim), color='#888780',
                       linestyle='--', linewidth=1,
                       label=f'target ≈ {math.log(cat_dim):.3f}')
            ax.legend(fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Loss curves saved → {save_path}")


# ---------------------------------------------------------------------------
# 4. Classification error rate  (MNIST only, Section 7.2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_classification_error(
    G,
    DQ,
    dataloader,
    device     : torch.device,
    dataset    : str = 'mnist',
    n_batches  : int = 20,
) -> float:
    """
    Evaluate how well the FIRST categorical code c1 recovers digit identity.
    Only meaningful for MNIST (single categorical code = digit class).
    """
    m = _load_model_module(dataset)
    CAT_DIM = m.CAT_DIM
    N_CATS  = getattr(m, 'N_CATS', 1)
    parse_q_output = m.parse_q_output

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

        # For multiple categorical codes, use the first one
        if isinstance(cat_prob, list):
            pred_cluster = cat_prob[0].argmax(dim=1).cpu().numpy()
        else:
            pred_cluster = cat_prob.argmax(dim=1).cpu().numpy()

        all_pred.extend(pred_cluster.tolist())
        all_label.extend(labels.numpy().tolist())

    all_pred  = np.array(all_pred)
    all_label = np.array(all_label)

    n_classes = CAT_DIM
    cost = np.zeros((n_classes, n_classes), dtype=np.int64)
    for p, l in zip(all_pred, all_label):
        if l < n_classes:
            cost[p, l] += 1

    row_ind, col_ind = linear_sum_assignment(-cost)
    correct = cost[row_ind, col_ind].sum()
    total   = len(all_pred)
    error   = 1.0 - correct / total

    print(f"  Classification error: {error*100:.2f}%  "
          f"({correct}/{total} correct after optimal cluster assignment)")
    return error


# ---------------------------------------------------------------------------
# 5. Mode comparison grid
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_mode_comparison(
    generators  : dict,
    device      : torch.device,
    dataset     : str = 'mnist',
    n_samples   : int  = 10,
    save_path   : str  = 'results/mode_comparison.png',
):
    """Side-by-side comparison of generated samples from different training modes."""
    _ensure_dir(save_path)

    m = _load_model_module(dataset)
    sample_latent = m.sample_latent
    concat_latent = m.concat_latent
    value_range = _get_value_range(dataset)

    z_noise, c_cat, c_cont = sample_latent(n_samples, device)
    z = concat_latent(z_noise, c_cat, c_cont)

    n_modes = len(generators)
    fig, axes = plt.subplots(n_modes, 1, figsize=(n_samples * 0.8, n_modes * 1.2))
    if n_modes == 1:
        axes = [axes]

    for ax, (mode_name, G) in zip(axes, generators.items()):
        G.eval()
        imgs = G(z)
        grid = make_grid(imgs, nrow=n_samples, normalize=True,
                         value_range=value_range, padding=2)
        img_np = _to_numpy_img(grid, value_range=value_range)
        cmap = 'gray' if img_np.ndim == 2 else None
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
# 6. Ablation: lambda sweep
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_lambda_ablation(
    generators  : dict,
    device      : torch.device,
    dataset     : str = 'mnist',
    save_path   : str = 'results/lambda_ablation.png',
):
    """Show how the MI regularisation weight λ affects disentanglement."""
    _ensure_dir(save_path)
    n_models = len(generators)

    m = _load_model_module(dataset)
    NOISE_DIM = m.NOISE_DIM
    CAT_DIM   = m.CAT_DIM
    CONT_DIM  = m.CONT_DIM
    N_CATS    = getattr(m, 'N_CATS', 1)
    concat_latent = m.concat_latent
    value_range = _get_value_range(dataset)

    fixed_noise = torch.FloatTensor(10, NOISE_DIM).uniform_(-1, 1).to(device)
    z_noise = fixed_noise.repeat_interleave(10, dim=0)

    # Vary first categorical code
    c_cat = torch.zeros(100, N_CATS * CAT_DIM, device=device)
    for i in range(10):
        c_cat[i*10:(i+1)*10, i] = 1.0
        for j in range(1, N_CATS):
            c_cat[i*10:(i+1)*10, j * CAT_DIM] = 1.0

    c_cont = torch.zeros(100, CONT_DIM, device=device)
    z = concat_latent(z_noise, c_cat, c_cont)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (label, G) in zip(axes, generators.items()):
        G.eval()
        imgs = G(z)
        _plot_grid(ax, imgs, 10, 10, value_range, title=f'{label}')

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
    """Load a saved checkpoint and generate all figures for the report."""
    from datasets import build_loader

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    m = _load_model_module(dataset)
    Generator      = m.Generator
    DiscriminatorQ = m.DiscriminatorQ

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
        save_path   = f'{results_dir}/{dataset}_{mode}_traversal.png',
    )

    # 2. Classification error (MNIST only — SVHN labels are 1-9, not 0-9, and more complex)
    if dataset == 'mnist':
        test_loader = build_loader(dataset, batch_size=128,
                                   split='test', num_workers=2)
        err = compute_classification_error(G, DQ, test_loader, device, dataset=dataset)
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