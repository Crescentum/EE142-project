"""
InfoGAN network architectures for CelebA.

CelebA in the InfoGAN paper uses 10 independent categorical latent codes,
each with 10 categories. The generator maps the concatenated latent vector

    z = [noise(128) || c_1(10) || ... || c_10(10)]

to a 64x64 RGB face image in [-1, 1]. The discriminator and recognition
network share a convolutional trunk; D predicts real/fake and Q predicts the
posterior of each categorical code.
"""

from __future__ import annotations

import torch
import torch.nn as nn


NOISE_DIM = 128
CAT_DIMS = (10,) * 10
CAT_DIM = sum(CAT_DIMS)
CONT_DIM = 0
LATENT_DIM = NOISE_DIM + CAT_DIM + CONT_DIM
Q_OUT_DIM = CAT_DIM
IMAGE_VALUE_RANGE = (-1, 1)


def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


class Generator(nn.Module):
    """Map a 228-D InfoGAN latent vector to a 64x64 RGB CelebA image."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 1024, bias=False),
            nn.BatchNorm1d(4 * 4 * 1024),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh(),
        )

        self.apply(_weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.fc(z)
        out = out.view(-1, 1024, 4, 4)
        return self.deconv(out)


class DiscriminatorQ(nn.Module):
    """Shared D/Q network for 64x64 RGB CelebA images."""

    def __init__(self, q_out_dim: int = Q_OUT_DIM):
        super().__init__()

        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.d_head = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

        self.q_head = nn.Sequential(
            nn.Linear(1024, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, q_out_dim),
        )

        self.apply(_weights_init)

    def forward(self, x: torch.Tensor):
        feat = self.shared_conv(x)
        feat = self.shared_fc(feat)
        d_out = self.d_head(feat)
        q_out = self.q_head(feat)
        return d_out, q_out


def parse_q_output(q_out: torch.Tensor):
    """Return categorical probabilities and empty continuous posterior tensors."""
    cat_probs = []
    offset = 0
    for dim in CAT_DIMS:
        logits = q_out[:, offset:offset + dim]
        cat_probs.append(torch.softmax(logits, dim=1))
        offset += dim

    cat_prob = torch.cat(cat_probs, dim=1)
    empty = q_out.new_empty(q_out.size(0), 0)
    return cat_prob, empty, empty


def sample_latent(batch_size: int, device: torch.device):
    z_noise = torch.empty(batch_size, NOISE_DIM, device=device).uniform_(-1, 1)

    codes = []
    for dim in CAT_DIMS:
        cat_idx = torch.randint(0, dim, (batch_size,), device=device)
        c_i = torch.zeros(batch_size, dim, device=device)
        c_i.scatter_(1, cat_idx.unsqueeze(1), 1.0)
        codes.append(c_i)

    c_cat = torch.cat(codes, dim=1)
    c_cont = torch.empty(batch_size, 0, device=device)
    return z_noise, c_cat, c_cont


def concat_latent(z_noise: torch.Tensor,
                  c_cat: torch.Tensor,
                  c_cont: torch.Tensor) -> torch.Tensor:
    if c_cont.numel() == 0:
        return torch.cat([z_noise, c_cat], dim=1)
    return torch.cat([z_noise, c_cat, c_cont], dim=1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 8
    G = Generator().to(device)
    DQ = DiscriminatorQ().to(device)

    z_noise, c_cat, c_cont = sample_latent(B, device)
    z = concat_latent(z_noise, c_cat, c_cont)
    fake_imgs = G(z)
    d_out, q_out = DQ(fake_imgs)
    cat_prob, cont_mean, cont_std = parse_q_output(q_out)

    print("=== CelebA shape checks ===")
    print(f"z         : {z.shape}")
    print(f"fake_imgs : {fake_imgs.shape}")
    print(f"d_out     : {d_out.shape}")
    print(f"q_out     : {q_out.shape}")
    print(f"cat_prob  : {cat_prob.shape}")
    print(f"cont_mean : {cont_mean.shape}")
    print(f"cont_std  : {cont_std.shape}")
    print(f"image range: [{fake_imgs.min():.3f}, {fake_imgs.max():.3f}]")
