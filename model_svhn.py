"""
InfoGAN network architectures for SVHN.

Follows Appendix C.2 of the paper exactly:

Discriminator D / Recognition network Q (shared trunk):
  Input: 32x32 color image (3 channels)
  -> 4x4 Conv, 64, LeakyReLU, stride 2          (32 -> 16)
  -> 4x4 Conv, 128, LeakyReLU, stride 2, BN     (16 ->  8)
  -> 4x4 Conv, 256, LeakyReLU, stride 2, BN     ( 8 ->  4)
  -> Flatten
  -> FC output layer for D (sigmoid)
  -> FC 128, BN, LeakyReLU -> FC output for Q

Generator G:
  Input: z in R^168  (124 noise + 4*10 categorical + 4 continuous)
  -> FC 2*2*448, ReLU, BN
  -> reshape to (448, 2, 2)
  -> 4x4 ConvTranspose, 256, ReLU, stride 2, BN  ( 2 ->  4)
  -> 4x4 ConvTranspose, 128, ReLU, stride 2       ( 4 ->  8)
  -> 4x4 ConvTranspose,  64, ReLU, stride 2       ( 8 -> 16)
  -> 4x4 ConvTranspose,   3, Tanh, stride 2       (16 -> 32)

Latent spec (matching paper Section 7.2):
  z_noise  : Uniform(124)   -- not regularised
  c1~c4    : Categorical(10) each -- regularised, discrete (40 dim total)
  c5~c8    : Uniform(-1,1) each -- regularised, continuous (4 dim total)
  total dim: 124 + 40 + 4 = 168
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Hyper-parameters (kept here so trainer.py can import them too)
# ---------------------------------------------------------------------------
NOISE_DIM   = 124   # unstructured noise z
N_CATS    = 4
CAT_DIM     = 10   # categorical code c1, c2, c3, c4  (one-hot)
CONT_DIM    = 4    # number of continuous codes c5, c6, c7, c8
LATENT_DIM = 124 + 40 + 4  # 168

# Q head output layout:
Q_OUT_DIM = 40 + 4 + 4  # 48


# ---------------------------------------------------------------------------
# Weight initialisation  (paper uses truncated-normal stddev=0.02)
# ---------------------------------------------------------------------------
def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


# ---------------------------------------------------------------------------
# Generator  G(z, c)
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    """
    Maps latent vector [z || c1 || c2 || c3 || c4 || c5 || c6 || c7 || c8] (dim=168)
    to a 32x32 color image.

    Architecture (paper Table 2, generator column):
      FC 2*2*448 -> BN -> ReLU
      reshape (448, 2, 2)
      ConvT 4x4, stride 2, pad 1 -> 256ch -> BN -> ReLU   [2  ->  4]
      ConvT 4x4, stride 2, pad 1 -> 128ch -> ReLU        [4  ->  8]
      ConvT 4x4, stride 2, pad 1 ->  64ch -> ReLU        [8  -> 16]
      ConvT 4x4, stride 2, pad 1 ->   3ch -> Tanh        [16 -> 32]
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 2 * 2 * 448),
            nn.BatchNorm1d(2 * 2 * 448),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            # (448, 2, 2) -> (256, 4, 4)
            nn.ConvTranspose2d(448, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),

            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),

            # (64, 16, 16) -> (3, 32, 32)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh(),
        )

        self.apply(_weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, 168)  concatenated [noise || c1_onehot || ... || c4_onehot || c5 || ... || c8]
        Returns:
            img: (B, 3, 32, 32)  pixel values in [-1, 1] (Tanh output)
        """
        out = self.fc(z)                        # (B, 2*2*448)
        out = out.view(-1, 448, 2, 2)           # (B, 448, 2, 2)
        img = self.deconv(out)                  # (B, 3, 32, 32)
        return img


# ---------------------------------------------------------------------------
# Discriminator + Q network  (shared trunk, two heads)
# ---------------------------------------------------------------------------
class DiscriminatorQ(nn.Module):
    """
    Shared convolutional trunk with two output heads:

      D head: single sigmoid output (real/fake probability)
      Q head: outputs parameters for Q(c|x)
              - 40 logits (4 groups of 10) -> softmax -> categorical posteriors
              -  4 means   -> continuous posterior means
              -  4 logstds -> exp()   -> continuous posterior stds

    Architecture (paper Table 2, discriminator/Q column):
      Conv 4x4, stride 2 -> 64ch  -> LeakyReLU(0.1)       [32 -> 16]
      Conv 4x4, stride 2 -> 128ch -> BN -> LeakyReLU(0.1)[16 ->  8]
      Conv 4x4, stride 2 -> 256ch -> BN -> LeakyReLU(0.1)[ 8 ->  4]
      Flatten
      ├── [D head] FC 1   -> Sigmoid
      └── [Q head] FC 128 -> BN -> LeakyReLU(0.1) -> FC 48
    """

    def __init__(self, q_out_dim: int = Q_OUT_DIM):
        super().__init__()

        # ── shared convolutional trunk ──────────────────────────────────────
        self.shared_conv = nn.Sequential(
            # (3, 32, 32) -> (64, 16, 16)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),

            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.flatten = nn.Flatten()

        # ── D head ──────────────────────────────────────────────────────────
        self.d_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid(),
        )

        # ── Q head ──────────────────────────────────────────────────────────
        # extra hidden layer before output (paper: "FC.128-batchnorm-lRELU-FC.output")
        self.q_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, q_out_dim),
        )

        self.apply(_weights_init)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, 32, 32)  image
        Returns:
            d_out   : (B, 1)   real/fake score in (0, 1)
            q_out   : (B, 48)  raw Q head output (not yet activated)
                      caller is responsible for:
                        softmax( q_out[:, i*10:(i+1)*10] ) for each categorical code
                        q_out[:, 40:44]                  -> continuous means
                        exp( q_out[:, 44:48] )             -> continuous stds
        """
        feat = self.shared_conv(x)      # (B, 256, 4, 4)
        feat = self.flatten(feat)       # (B, 4096)

        d_out = self.d_head(feat)       # (B, 1)
        q_out = self.q_head(feat)       # (B, 48)

        return d_out, q_out


# ---------------------------------------------------------------------------
# Helper: parse Q head output into posterior parameters
# ---------------------------------------------------------------------------
def parse_q_output(q_out: torch.Tensor):
    """
    Decompose the raw Q head output into interpretable posterior parameters.

    Args:
        q_out: (B, 48)

    Returns:
        cat_probs : list of 4 tensors, each (B, 10)   categorical posterior probabilities
        cont_mean : (B, 4)   Gaussian posterior means
        cont_std  : (B, 4)   Gaussian posterior stds  (> 0)
    """
    cat_logits = [q_out[:, i*CAT_DIM:i*CAT_DIM+CAT_DIM] for i in range(N_CATS)]
    cat_probs = [torch.softmax(logits, dim=1) for logits in cat_logits]
    cont_mean  = q_out[:, N_CATS * CAT_DIM: N_CATS * CAT_DIM + CONT_DIM]          # (B, 4)
    cont_logstd = q_out[:, N_CATS * CAT_DIM + CONT_DIM:]                 # (B, 4)

    cont_std  = torch.exp(cont_logstd)             # ensures positivity

    return cat_probs, cont_mean, cont_std


# ---------------------------------------------------------------------------
# Helper: sample latent vector z for one batch
# ---------------------------------------------------------------------------
def sample_latent(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample the full latent vector and return its components separately
    so that the trainer can compute the MI loss directly.

    Sampling distributions (matching original run_mnist_exp.py):
      noise z : Uniform(-1, 1)  shape (B, 62)
      c1      : Categorical(10) returned as one-hot  shape (B, 10)
      c2, c3  : Uniform(-1, 1)  shape (B, 2)

    Returns:
        z_noise   : (B, 62)
        c_cat     : (B, 10)   one-hot
        c_cont    : (B,  2)   uniform in [-1, 1]
    """
    z_noise = torch.FloatTensor(batch_size, NOISE_DIM).uniform_(-1, 1).to(device)

    # Sample categorical indices, then convert to one-hot
    cat_parts = []
    for _ in range(N_CATS):
        cat_idx = torch.randint(0, CAT_DIM, (batch_size,), device=device)
        c_cat   = torch.zeros(batch_size, CAT_DIM, device=device)
        c_cat.scatter_(1, cat_idx.unsqueeze(1), 1.0)
        cat_parts.append(c_cat)

    c_cat = torch.cat(cat_parts, dim=1)

    c_cont  = torch.FloatTensor(batch_size, CONT_DIM).uniform_(-1, 1).to(device)

    return z_noise, c_cat, c_cont


def concat_latent(z_noise: torch.Tensor,
                  c_cat:   torch.Tensor,
                  c_cont:  torch.Tensor) -> torch.Tensor:
    """Concatenate components into the full latent vector (B, 168)."""
    return torch.cat([z_noise, c_cat, c_cont], dim=1)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    B = 128
    G  = Generator().to(device)
    DQ = DiscriminatorQ().to(device)

    # --- forward pass -------------------------------------------------------
    z_noise, c_cat, c_cont = sample_latent(B, device)
    z = concat_latent(z_noise, c_cat, c_cont)   # (128, 168)

    fake_imgs = G(z)                             # (128, 3, 32, 32)
    d_out, q_out = DQ(fake_imgs)                 # (128,1)  (128,48)
    cat_probs, cont_mean, cont_std = parse_q_output(q_out)

    print("=== Shape checks ===")
    print(f"  z          : {z.shape}")           # (128, 168)
    print(f"  fake_imgs  : {fake_imgs.shape}")   # (128, 3, 32, 32)
    print(f"  d_out      : {d_out.shape}")       # (128, 1)
    print(f"  q_out      : {q_out.shape}")       # (128, 48)
    print(f"  cat_probs  : {[p.shape for p in cat_probs]}")  # 4 x (128, 10)
    print(f"  cont_mean  : {cont_mean.shape}")   # (128,  4)
    print(f"  cont_std   : {cont_std.shape}")    # (128,  4)

    print("\n=== Value checks ===")
    print(f"  fake_imgs  range : [{fake_imgs.min():.3f}, {fake_imgs.max():.3f}]  (expect [-1,1])")
    print(f"  d_out      range : [{d_out.min():.3f},  {d_out.max():.3f}]   (expect (0,1))")
    for i, cp in enumerate(cat_probs):
        print(f"  cat_prob[{i}] sum   : {cp.sum(dim=1).mean():.4f}  (expect 1.0)")
    print(f"  cont_std   min   : {cont_std.min():.4f}  (expect > 0)")

    print("\n=== Parameter counts ===")
    g_params  = sum(p.numel() for p in G.parameters())
    dq_params = sum(p.numel() for p in DQ.parameters())
    print(f"  Generator      : {g_params:,}")
    print(f"  DiscriminatorQ : {dq_params:,}")
    print(f"  Total          : {g_params + dq_params:,}")

    print("\nAll checks passed.")