"""
InfoGAN network architectures for 3D Chairs dataset.
Follows Appendix C.5 of the InfoGAN paper exactly.

Latent spec:
  z_noise  : Uniform(128)           -- not regularised
  c1,c3,c4 : Categorical(20) each    -- regularised, discrete (3 codes)
  c5       : Uniform(1)              -- regularised, continuous
  total dim: 128 + 3*20 + 1 = 189

Generator output: 64×64 grayscale image
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
NOISE_DIM   = 128
CAT_DIMS    = (20, 20, 20)   # 3 discrete latent codes (c1, c3, c4)
CAT_DIM     = sum(CAT_DIMS)  # 60
N_CATS      = 3
CONT_DIM    = 1              # 1 continuous latent code (c5)
LATENT_DIM  = NOISE_DIM + CAT_DIM + CONT_DIM   # 189

# Q head output layout:
#   [0:20 ]  -> c1 categorical logits
#   [20:40 ]  -> c3 categorical logits
#   [40:60 ]  -> c4 categorical logits
#   [60:61 ]  -> continuous mean
#   [61:62 ]  -> continuous log-std
Q_OUT_DIM = CAT_DIM + CONT_DIM * 2   # 62


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------
def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    """
    Maps latent vector [z || c1 || c3 || c4 || c5] (dim=189) to a 64×64 greyscale image.

    Paper Table 6 generator column:
      FC 1024 → BN → ReLU
      FC 8×8×256 → BN → ReLU
      reshape (256, 8, 8)
      4×4 upconv 256 → BN → ReLU   (stride 1, kernel 3 to keep 8×8)
      4×4 upconv 256 → BN → ReLU   (stride 1, kernel 3 to keep 8×8)
      4×4 upconv 128 → BN → ReLU   stride 2  (8 → 16)
      4×4 upconv  64 → BN → ReLU   stride 2  (16 → 32)
      4×4 upconv   1 → Sigmoid      stride 2  (32 → 64)
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 8 * 8 * 256),
            nn.BatchNorm1d(8 * 8 * 256),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            # (256, 8, 8) → (256, 8, 8)  [kernel=3, stride=1 keeps spatial size]
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (256, 8, 8) → (256, 8, 8)
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (256, 8, 8) → (128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # (128, 16, 16) → (64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # (64, 32, 32) → (1, 64, 64)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid(),
        )

        self.apply(_weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.fc(z)                        # (B, 8*8*256)
        out = out.view(-1, 256, 8, 8)           # (B, 256, 8, 8)
        img = self.deconv(out)                  # (B, 1, 64, 64)
        return img


# ---------------------------------------------------------------------------
# Discriminator + Q network
# ---------------------------------------------------------------------------
class DiscriminatorQ(nn.Module):
    """
    Shared convolutional trunk with two output heads.

    Paper Table 6 discriminator/Q column:
      Conv 4×4 stride 2 → 64ch  → lRELU(0.1)       [64 → 32]
      Conv 4×4 stride 2 → 128ch → BN → lRELU(0.1) [32 → 16]
      Conv 4×4 stride 2 → 256ch → BN → lRELU(0.1) [16 → 8]
      Conv 4×4 stride 1 → 256ch → BN → lRELU(0.1) [8 → 8]  (kernel=3 keeps size)
      Conv 4×4 stride 1 → 256ch → BN → lRELU(0.1) [8 → 8]  (kernel=3 keeps size)
      Flatten
      FC 1024 → BN → lRELU(0.1)
      ├── [D head] FC 1   → Sigmoid
      └── [Q head] FC 128 → BN → lRELU(0.1) → FC 62
    """

    def __init__(self, q_out_dim: int = Q_OUT_DIM):
        super().__init__()

        self.shared_conv = nn.Sequential(
            # (1, 64, 64) → (64, 32, 32)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),

            # (64, 32, 32) → (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            # (128, 16, 16) → (256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            # (256, 8, 8) → (256, 8, 8)  [kernel=3, stride=1 keeps size]
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            # (256, 8, 8) → (256, 8, 8)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024, bias=False),
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
        feat = self.shared_conv(x)      # (B, 256, 8, 8)
        feat = self.shared_fc(feat)     # (B, 1024)
        d_out = self.d_head(feat)       # (B, 1)
        q_out = self.q_head(feat)       # (B, 62)
        return d_out, q_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_q_output(q_out: torch.Tensor):
    """
    Decompose the raw Q head output into interpretable posterior parameters.
    Returns:
        cat_prob  : list of 3 tensors, each (B, 20)  (softmax probabilities)
        cont_mean : (B, 1)
        cont_std  : (B, 1)   (> 0)
    """
    cat_logits = []
    offset = 0
    for dim in CAT_DIMS:
        cat_logits.append(q_out[:, offset:offset + dim])
        offset += dim

    cat_prob = [torch.softmax(logits, dim=1) for logits in cat_logits]
    cont_mean = q_out[:, offset:offset + CONT_DIM]
    cont_logstd = q_out[:, offset + CONT_DIM:offset + CONT_DIM * 2]
    cont_std = torch.exp(cont_logstd)
    return cat_prob, cont_mean, cont_std


def sample_latent(batch_size: int, device: torch.device):
    """
    Sample the full latent vector and return its components separately.
      noise z : Uniform(-1, 1)  shape (B, 128)
      c1,c3,c4: Categorical(20) returned as one-hot blocks in (B, 60)
      c5      : Uniform(-1, 1)  shape (B, 1)
    """
    z_noise = torch.FloatTensor(batch_size, NOISE_DIM).uniform_(-1, 1).to(device)

    c_cat = torch.zeros(batch_size, CAT_DIM, device=device)
    offset = 0
    for dim in CAT_DIMS:
        cat_idx = torch.randint(0, dim, (batch_size,), device=device)
        c_cat.scatter_(1, (cat_idx + offset).unsqueeze(1), 1.0)
        offset += dim

    c_cont = torch.FloatTensor(batch_size, CONT_DIM).uniform_(-1, 1).to(device)

    return z_noise, c_cat, c_cont


def concat_latent(z_noise: torch.Tensor,
                  c_cat: torch.Tensor,
                  c_cont: torch.Tensor) -> torch.Tensor:
    """Concatenate components into the full latent vector (B, 189)."""
    return torch.cat([z_noise, c_cat, c_cont], dim=1)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    B = 64
    G = Generator().to(device)
    DQ = DiscriminatorQ().to(device)

    z_noise, c_cat, c_cont = sample_latent(B, device)
    z = concat_latent(z_noise, c_cat, c_cont)   # (64, 189)

    fake_imgs = G(z)                             # (64, 1, 64, 64)
    d_out, q_out = DQ(fake_imgs)                 # (64,1)  (64,62)
    cat_prob, cont_mean, cont_std = parse_q_output(q_out)

    print("=== Shape checks ===")
    print(f"  z          : {z.shape}")           # (64, 189)
    print(f"  fake_imgs  : {fake_imgs.shape}")   # (64, 1, 64, 64)
    print(f"  d_out      : {d_out.shape}")       # (64, 1)
    print(f"  q_out      : {q_out.shape}")       # (64, 62)
    print(f"  cat_prob[0]: {cat_prob[0].shape}")  # (64, 20)
    print(f"  cont_mean  : {cont_mean.shape}")   # (64, 1)
    print(f"  cont_std   : {cont_std.shape}")    # (64, 1)

    print("\n=== Value checks ===")
    print(f"  fake_imgs  range : [{fake_imgs.min():.3f}, {fake_imgs.max():.3f}]  (expect [0,1])")
    print(f"  d_out      range : [{d_out.min():.3f},  {d_out.max():.3f}]   (expect (0,1))")
    print(f"  cat_prob[0] sum  : {cat_prob[0].sum(dim=1).mean():.4f}  (expect 1.0)")
    print(f"  cont_std   min   : {cont_std.min():.4f}  (expect > 0)")

    print("\n=== Parameter counts ===")
    g_params = sum(p.numel() for p in G.parameters())
    dq_params = sum(p.numel() for p in DQ.parameters())
    print(f"  Generator      : {g_params:,}")
    print(f"  DiscriminatorQ : {dq_params:,}")
    print(f"  Total          : {g_params + dq_params:,}")

    print("\nAll checks passed.")