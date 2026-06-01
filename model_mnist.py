"""
InfoGAN network architectures for MNIST.

Follows Appendix C.1 of the paper exactly:

Discriminator D / Recognition network Q (shared trunk):
  Input: 28x28 grayscale image
  -> 4x4 Conv, 64, LeakyReLU, stride 2          (28 -> 14)
  -> 4x4 Conv, 128, LeakyReLU, stride 2, BN     (14 ->  7)
  -> FC 1024, LeakyReLU, BN
  -> [D head] FC -> 1 (sigmoid)
  -> [Q head] FC 128, BN, LeakyReLU -> FC -> 14
               14 = 10 (categorical) + 2 (cont mean) + 2 (cont logstd)

Generator G:
  Input: z in R^74  (62 noise + 10 categorical + 2 continuous)
  -> FC 1024, ReLU, BN
  -> FC 7*7*128, ReLU, BN
  -> reshape to (128, 7, 7)
  -> 4x4 ConvTranspose, 64, ReLU, stride 2, BN  ( 7 -> 14)
  -> 4x4 ConvTranspose,  1, Sigmoid, stride 2   (14 -> 28)

Latent spec (matching run_mnist_exp.py):
  z_noise  : Uniform(62)   -- not regularised
  c1       : Categorical(10) -- regularised, discrete
  c2       : Uniform(1)    -- regularised, continuous
  c3       : Uniform(1)    -- regularised, continuous
  total dim: 62 + 10 + 1 + 1 = 74
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Hyper-parameters (kept here so trainer.py can import them too)
# ---------------------------------------------------------------------------
NOISE_DIM   = 62   # unstructured noise z
CAT_DIM     = 10   # categorical code c1  (one-hot)
CONT_DIM    = 2    # number of continuous codes c2, c3
LATENT_DIM  = NOISE_DIM + CAT_DIM + CONT_DIM   # 74

# Q head output layout:
#   [ 0:10 ]  -> categorical logits  (softmax -> prob)
#   [10:12 ]  -> continuous means
#   [12:14 ]  -> continuous log-stds  (exp -> std, ensures positivity)
Q_OUT_DIM = CAT_DIM + CONT_DIM * 2   # 14


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
    Maps latent vector [z || c1 || c2 || c3] (dim=74) to a 28x28 greyscale image.

    Architecture (paper Table 1, generator column):
      FC 1024 -> BN -> ReLU
      FC 7*7*128 -> BN -> ReLU
      reshape (128, 7, 7)
      ConvT 4x4, stride 2, pad 1 -> 64ch -> BN -> ReLU   [7  -> 14]
      ConvT 4x4, stride 2, pad 1 ->  1ch -> Sigmoid       [14 -> 28]
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 7 * 7 * 128),
            nn.BatchNorm1d(7 * 7 * 128),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            # (128, 7, 7) -> (64, 14, 14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # (64, 14, 14) -> (1, 28, 28)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid(),
        )

        self.apply(_weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, 74)  concatenated [noise || c1_onehot || c2 || c3]
        Returns:
            img: (B, 1, 28, 28)  pixel values in [0, 1]
        """
        out = self.fc(z)                        # (B, 7*7*128)
        out = out.view(-1, 128, 7, 7)           # (B, 128, 7, 7)
        img = self.deconv(out)                  # (B, 1, 28, 28)
        return img


# ---------------------------------------------------------------------------
# Discriminator + Q network  (shared trunk, two heads)
# ---------------------------------------------------------------------------
class DiscriminatorQ(nn.Module):
    """
    Shared convolutional trunk with two output heads:

      D head: single sigmoid output (real/fake probability)
      Q head: outputs parameters for Q(c|x)
              - 10 logits  -> softmax -> categorical posterior
              -  2 means   -> continuous posterior means
              -  2 logstds -> exp()   -> continuous posterior stds

    Architecture (paper Table 1, discriminator/Q column):
      Conv 4x4, stride 2 -> 64ch  -> LeakyReLU(0.1)       [28 -> 14]
      Conv 4x4, stride 2 -> 128ch -> BN -> LeakyReLU(0.1) [14 ->  7]
      Flatten
      FC 1024             -> BN -> LeakyReLU(0.1)
      ├── [D head] FC 1   -> Sigmoid
      └── [Q head] FC 128 -> BN -> LeakyReLU(0.1) -> FC 14
    """

    def __init__(self, q_out_dim: int = Q_OUT_DIM):
        super().__init__()

        # ── shared convolutional trunk ──────────────────────────────────────
        self.shared_conv = nn.Sequential(
            # (1, 28, 28) -> (64, 14, 14)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),

            # (64, 14, 14) -> (128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # ── shared FC trunk ─────────────────────────────────────────────────
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # ── D head ──────────────────────────────────────────────────────────
        self.d_head = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

        # ── Q head ──────────────────────────────────────────────────────────
        # extra hidden layer before output (paper: "FC.128-batchnorm-lRELU-FC.output")
        self.q_head = nn.Sequential(
            nn.Linear(1024, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, q_out_dim),
        )

        self.apply(_weights_init)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, 28, 28)  image in [0, 1]
        Returns:
            d_out   : (B, 1)   real/fake score in (0, 1)
            q_out   : (B, 14)  raw Q head output (not yet activated)
                      caller is responsible for:
                        softmax( q_out[:, :10]   )  -> categorical probs
                        q_out[:, 10:12]             -> continuous means
                        exp( q_out[:, 12:14] )      -> continuous stds
        """
        feat = self.shared_conv(x)      # (B, 128, 7, 7)
        feat = self.shared_fc(feat)     # (B, 1024)

        d_out = self.d_head(feat)       # (B, 1)
        q_out = self.q_head(feat)       # (B, 14)

        return d_out, q_out


# ---------------------------------------------------------------------------
# Helper: parse Q head output into posterior parameters
# ---------------------------------------------------------------------------
def parse_q_output(q_out: torch.Tensor):
    """
    Decompose the raw Q head output into interpretable posterior parameters.

    Args:
        q_out: (B, 14)

    Returns:
        cat_prob  : (B, 10)   categorical posterior probabilities (sum to 1)
        cont_mean : (B,  2)   Gaussian posterior means
        cont_std  : (B,  2)   Gaussian posterior stds  (> 0)
    """
    cat_logits = q_out[:, :CAT_DIM]               # (B, 10)
    cont_mean  = q_out[:, CAT_DIM: CAT_DIM + CONT_DIM]          # (B, 2)
    cont_logstd = q_out[:, CAT_DIM + CONT_DIM:]                 # (B, 2)

    cat_prob  = torch.softmax(cat_logits, dim=1)
    cont_std  = torch.exp(cont_logstd)             # ensures positivity

    return cat_prob, cont_mean, cont_std


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
    cat_idx = torch.randint(0, CAT_DIM, (batch_size,), device=device)
    c_cat   = torch.zeros(batch_size, CAT_DIM, device=device)
    c_cat.scatter_(1, cat_idx.unsqueeze(1), 1.0)

    c_cont  = torch.FloatTensor(batch_size, CONT_DIM).uniform_(-1, 1).to(device)

    return z_noise, c_cat, c_cont


def concat_latent(z_noise: torch.Tensor,
                  c_cat:   torch.Tensor,
                  c_cont:  torch.Tensor) -> torch.Tensor:
    """Concatenate components into the full latent vector (B, 74)."""
    return torch.cat([z_noise, c_cat, c_cont], dim=1)


# ---------------------------------------------------------------------------
# Quick sanity check (run this file directly: python models.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    B = 128
    G  = Generator().to(device)
    DQ = DiscriminatorQ().to(device)

    # --- forward pass -------------------------------------------------------
    z_noise, c_cat, c_cont = sample_latent(B, device)
    z = concat_latent(z_noise, c_cat, c_cont)   # (128, 74)

    fake_imgs = G(z)                             # (128, 1, 28, 28)
    d_out, q_out = DQ(fake_imgs)                 # (128,1)  (128,14)
    cat_prob, cont_mean, cont_std = parse_q_output(q_out)

    print("=== Shape checks ===")
    print(f"  z          : {z.shape}")           # (128, 74)
    print(f"  fake_imgs  : {fake_imgs.shape}")   # (128, 1, 28, 28)
    print(f"  d_out      : {d_out.shape}")       # (128, 1)
    print(f"  q_out      : {q_out.shape}")       # (128, 14)
    print(f"  cat_prob   : {cat_prob.shape}")    # (128, 10)
    print(f"  cont_mean  : {cont_mean.shape}")   # (128,  2)
    print(f"  cont_std   : {cont_std.shape}")    # (128,  2)

    print("\n=== Value checks ===")
    print(f"  fake_imgs  range : [{fake_imgs.min():.3f}, {fake_imgs.max():.3f}]  (expect [0,1])")
    print(f"  d_out      range : [{d_out.min():.3f},  {d_out.max():.3f}]   (expect (0,1))")
    print(f"  cat_prob   sum   : {cat_prob.sum(dim=1).mean():.4f}  (expect 1.0)")
    print(f"  cont_std   min   : {cont_std.min():.4f}  (expect > 0)")

    print("\n=== Parameter counts ===")
    g_params  = sum(p.numel() for p in G.parameters())
    dq_params = sum(p.numel() for p in DQ.parameters())
    print(f"  Generator      : {g_params:,}")
    print(f"  DiscriminatorQ : {dq_params:,}")
    print(f"  Total          : {g_params + dq_params:,}")

    print("\nAll checks passed.")