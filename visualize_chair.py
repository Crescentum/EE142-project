"""
visualize.py — Reproduce all figures needed for the report.
Compatible with MNIST, SVHN, CelebA, and Chairs (multiple categorical codes).
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

# Fallback configurations for datasets not in DATASET_CFG
DEFAULT_CFG = {
    'mnist': {'pixel_range': '01', 'img_size': 28, 'channels': 1},
    'svhn': {'pixel_range': '11', 'img_size': 32, 'channels': 3},
    'celeba': {'pixel_range': '11', 'img_size': 32, 'channels': 3},
    'chairs': {'pixel_range': '01', 'img_size': 64, 'channels': 1},
}


def _load_model_module(dataset: str):
    module_name = f"model_{dataset}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"Cannot find '{module_name}.py'. "
            f"Make sure model_{dataset}.py is in the same directory."
        )


def _to_numpy_img(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.cpu().clamp(0, 1).numpy()
    if img.shape[0] == 1:
        return img[0]
    return img.transpose(1, 2, 0)


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _render_grid(ax, imgs: torch.Tensor, n_cols: int, title: str):
    grid = make_grid(imgs, nrow=n_cols, normalize=False, padding=2, pad_value=0)
    img_np = _to_numpy_img(grid)
    cmap = 'gray' if img_np.ndim == 2 else None
    ax.imshow(img_np, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(title, fontsize=10, pad=4)
    ax.axis('off')


@torch.no_grad()
def _make_cat_grid(G, m, device, pixel_range, code_idx=0, n_rows=5, n_cols=10):
    NOISE_DIM = m.NOISE_DIM
    CONT_DIM = m.CONT_DIM
    CAT_DIMS = getattr(m, 'CAT_DIMS', (m.CAT_DIM,))
    total_cat = sum(CAT_DIMS)
    cat_dim = CAT_DIMS[code_idx]
    n_cols = min(n_cols, cat_dim)
    row_noises = torch.FloatTensor(n_rows, NOISE_DIM).uniform_(-1, 1).to(device)
    z_noise = row_noises.repeat_interleave(n_cols, dim=0)
    c_cat = torch.zeros(n_rows * n_cols, total_cat, device=device)
    offset = 0
    for i, dim in enumerate(CAT_DIMS):
        if i == code_idx:
            for row in range(n_rows):
                for col in range(n_cols):
                    c_cat[row * n_cols + col, offset + col] = 1.0
        else:
            c_cat[:, offset] = 1.0
        offset += dim
    c_cont = torch.zeros(n_rows * n_cols, CONT_DIM, device=device)
    z = m.concat_latent(z_noise, c_cat, c_cont)
    imgs = G(z)
    return denorm(imgs, pixel_range)


@torch.no_grad()
def _make_cont_grid(G, m, device, pixel_range, cont_idx=0,
                    cont_range=2.0, n_rows=5, n_cols=10):
    NOISE_DIM = m.NOISE_DIM
    CONT_DIM = m.CONT_DIM
    CAT_DIMS = getattr(m, 'CAT_DIMS', (m.CAT_DIM,))
    total_cat = sum(CAT_DIMS)
    row_noises = torch.FloatTensor(n_rows, NOISE_DIM).uniform_(-1, 1).to(device)
    z_noise = row_noises.repeat_interleave(n_cols, dim=0)
    c_cat = torch.zeros(n_rows * n_cols, total_cat, device=device)
    offset = 0
    for dim in CAT_DIMS:
        c_cat[:, offset] = 1.0
        offset += dim
    sweep = torch.linspace(-cont_range, cont_range, n_cols, device=device)
    c_cont = torch.zeros(n_rows * n_cols, CONT_DIM, device=device)
    for row in range(n_rows):
        for col in range(n_cols):
            c_cont[row * n_cols + col, cont_idx] = sweep[col]
    z = m.concat_latent(z_noise, c_cat, c_cont)
    imgs = G(z)
    return denorm(imgs, pixel_range)


@torch.no_grad()
def plot_figure2(
    G_infogan, m, device, pixel_range='01',
    G_gan=None, cont_range=2.0, n_rows=5, n_cols=10,
    save_path='results/figure2.png',
):
    _ensure_dir(save_path)
    G_infogan.eval()
    CAT_DIMS = getattr(m, 'CAT_DIMS', (m.CAT_DIM,))
    CONT_DIM = m.CONT_DIM
    N_CATS = getattr(m, 'N_CATS', 1)
    has_baseline = G_gan is not None
    n_cat_plots = N_CATS + (1 if has_baseline else 0)
    n_cont_plots = CONT_DIM
    n_plots = n_cat_plots + n_cont_plots

    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 3.5, n_rows * 0.7 + 1.2))
    if n_plots == 1:
        axes = [axes]
    ax_idx = 0

    for ci in range(N_CATS):
        label = f'$c_{{cat,{ci + 1}}}$' if N_CATS > 1 else '$c_1$'
        imgs = _make_cat_grid(G_infogan, m, device, pixel_range,
                              code_idx=ci, n_rows=n_rows, n_cols=n_cols)
        _render_grid(axes[ax_idx], imgs, min(n_cols, CAT_DIMS[ci]),
                     f'Varying {label} on InfoGAN')
        ax_idx += 1

    if has_baseline:
        G_gan.eval()
        imgs_b = _make_cat_grid(G_gan, m, device, pixel_range,
                                code_idx=0, n_rows=n_rows, n_cols=n_cols)
        _render_grid(axes[ax_idx], imgs_b, min(n_cols, CAT_DIMS[0]),
                     'Varying $c_1$ on GAN\n(No clear meaning)')
        ax_idx += 1

    for ci in range(CONT_DIM):
        label = f'$c_{{cont,{ci + 1}}}$' if CONT_DIM > 2 else f'$c_{{ci + 2}}$'
        imgs = _make_cont_grid(G_infogan, m, device, pixel_range,
                               cont_idx=ci, cont_range=cont_range,
                               n_rows=n_rows, n_cols=n_cols)
        _render_grid(axes[ax_idx], imgs, n_cols,
                     f'Varying {label} ∈ [−{cont_range}, {cont_range}]')
        ax_idx += 1

    dataset_name = 'Chairs' if N_CATS == 3 and CONT_DIM == 1 else \
                   ('SVHN' if N_CATS > 1 else 'MNIST')
    plt.suptitle(f'Manipulating latent codes on {dataset_name}',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Figure 2 saved → {save_path}")


def plot_mi_curve(
    li_infogan, li_baseline=None,
    save_path='results/figure1_mi_curve.png', cat_dim=10,
):
    _ensure_dir(save_path)
    target = math.log(cat_dim)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(li_infogan, color='#185FA5', linewidth=2, label='InfoGAN')
    if li_baseline:
        ax.plot(li_baseline, color='#2E8B2E', linewidth=1.5, label='GAN')
    ax.axhline(target, color='#E24B4A', linewidth=1, linestyle='--',
               label=f'$H(c)$ = log({cat_dim}) ≈ {target:.3f}')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('$\\mathcal{L}_I$', fontsize=13)
    ax.set_title('Mutual Information Lower Bound over Training', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Figure 1 saved → {save_path}")


@torch.no_grad()
def compute_classification_error(
    G, DQ, device, m, n_samples_per_class=256,
) -> float:
    G.eval()
    DQ.eval()
    NOISE_DIM = m.NOISE_DIM
    CONT_DIM = m.CONT_DIM
    CAT_DIMS = getattr(m, 'CAT_DIMS', (m.CAT_DIM,))
    N_CATS = getattr(m, 'N_CATS', 1)
    total_cat = sum(CAT_DIMS)
    all_errors = []

    for code_idx in range(N_CATS):
        cat_dim = CAT_DIMS[code_idx]
        all_pred, all_label = [], []
        for k in range(cat_dim):
            z_noise = torch.FloatTensor(n_samples_per_class, NOISE_DIM).uniform_(-1, 1).to(device)
            c_cat = torch.zeros(n_samples_per_class, total_cat, device=device)
            offset = 0
            for i, dim in enumerate(CAT_DIMS):
                if i == code_idx:
                    c_cat[:, offset + k] = 1.0
                else:
                    c_cat[:, offset] = 1.0
                offset += dim
            c_cont = torch.zeros(n_samples_per_class, CONT_DIM, device=device)
            z = m.concat_latent(z_noise, c_cat, c_cont)
            fake_imgs = G(z)
            _, q_out = DQ(fake_imgs)
            cat_prob, _, _ = m.parse_q_output(q_out)
            if isinstance(cat_prob, list):
                prob_i = cat_prob[code_idx]
            else:
                prob_i = cat_prob
            predicted = prob_i.argmax(dim=1).cpu().numpy()
            all_pred.extend(predicted.tolist())
            all_label.extend([k] * n_samples_per_class)

        all_pred = np.array(all_pred)
        all_label = np.array(all_label)
        cost = np.zeros((cat_dim, cat_dim), dtype=np.int64)
        for p, l in zip(all_pred, all_label):
            cost[p, l] += 1
        row_ind, col_ind = linear_sum_assignment(-cost)
        correct = cost[row_ind, col_ind].sum()
        total = len(all_pred)
        error = 1.0 - correct / total
        all_errors.append(error)
        code_label = f'cat_{code_idx}' if N_CATS > 1 else 'c1'
        print(f"  Classification error ({code_label}): {error * 100:.2f}%  "
              f"({correct}/{total} after Hungarian matching)")

    avg_error = np.mean(all_errors)
    if N_CATS > 1:
        print(f"  Average classification error: {avg_error * 100:.2f}%")
    return avg_error


def plot_loss_curves(
    history, mode='vanilla', save_path='results/loss_curves.png', cat_dim=10,
):
    _ensure_dir(save_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Training curves — {mode}', fontsize=13)
    specs = [('d_loss', 'D loss', '#E24B4A'),
             ('g_loss', 'G loss', '#1D9E75'),
             ('LI_disc', '$\\mathcal{L}_I$', '#185FA5')]
    for ax, (key, title, col) in zip(axes, specs):
        if key not in history:
            ax.set_visible(False)
            continue
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


@torch.no_grad()
def plot_mode_comparison(
    generators, m, device, pixel_range='01', n_samples=10,
    save_path='results/mode_comparison.png',
):
    _ensure_dir(save_path)
    z_noise, c_cat, c_cont = m.sample_latent(n_samples, device)
    z = m.concat_latent(z_noise, c_cat, c_cont)

    n_modes = len(generators)
    fig, axes = plt.subplots(n_modes, 1, figsize=(n_samples * 0.9, n_modes * 1.3))
    if n_modes == 1:
        axes = [axes]

    for ax, (name, G) in zip(axes, generators.items()):
        G.eval()
        imgs = denorm(G(z), pixel_range)
        grid = make_grid(imgs, nrow=n_samples, normalize=False, padding=2)
        img_np = _to_numpy_img(grid)
        cmap = 'gray' if img_np.ndim == 2 else None
        ax.imshow(img_np, cmap=cmap, vmin=0, vmax=1)
        ax.set_ylabel(name, fontsize=10, rotation=0, labelpad=65, va='center')
        ax.axis('off')

    fig.suptitle('Generated samples — same latent code, different training modes',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Mode comparison saved → {save_path}")


def run_all(
    ckpt_path, dataset='mnist', results_dir='results',
    device_str='cuda', ckpt_gan_path=None,
):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    meta = DATASET_CFG.get(dataset, DEFAULT_CFG.get(dataset, {'pixel_range': '01'}))
    m = _load_model_module(dataset)

    os.makedirs(results_dir, exist_ok=True)

    G = m.Generator().to(device)
    DQ = m.DiscriminatorQ().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt['G_state'])
    DQ.load_state_dict(ckpt['DQ_state'])
    mode = ckpt.get('mode', 'vanilla')
    print(f"Loaded: epoch={ckpt['epoch']}  mode={mode}  dataset={dataset}")

    G_gan = None
    if ckpt_gan_path:
        G_gan = m.Generator().to(device)
        ckpt_gan = torch.load(ckpt_gan_path, map_location=device)
        G_gan.load_state_dict(ckpt_gan['G_state'])
        print(f"Loaded GAN baseline: epoch={ckpt_gan['epoch']}")

    plot_figure2(
        G, m, device,
        pixel_range=getattr(meta, 'pixel_range', '01'),
        G_gan=G_gan,
        save_path=f'{results_dir}/{dataset}_{mode}_figure2.png',
    )

    compute_classification_error(G, DQ, device, m)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--ckpt_gan', default=None)
    p.add_argument('--dataset', default='mnist',
                   choices=['mnist', 'svhn', 'celeba', 'chairs'])
    p.add_argument('--out_dir', default='results')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    run_all(args.ckpt, args.dataset, args.out_dir, args.device, args.ckpt_gan)