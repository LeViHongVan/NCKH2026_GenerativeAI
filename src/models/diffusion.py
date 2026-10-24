import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal timestep embedding as used in DDPM.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock1D(nn.Module):
    """
    Residual block with Conv1D, GroupNorm, SiLU activation and Timestep conditioning.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.01):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        num_groups = min(8, out_channels)
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout1d(dropout)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout1d(dropout)
        )

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        # Add time step conditioning
        t_proj = self.time_mlp(time_emb).unsqueeze(-1)
        h = h + t_proj
        h = self.block2(h)
        return h + self.residual_conv(x)


class UNet1D(nn.Module):
    """
    Modified 1D U-Net for Malware Embedding Diffusion.
    Input shape: (batch_size, 1, 104)
    Output shape: (batch_size, 1, 104) - Predicted noise epsilon
    """
    def __init__(self, in_dim: int = 104, base_channels: int = 32, time_emb_dim: int = 64):
        super().__init__()
        self.in_dim = in_dim
        self.base_channels = base_channels

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # Input projection
        self.init_conv = nn.Conv1d(1, base_channels, kernel_size=3, padding=1)

        # Encoder (Downsample)
        self.down1 = ResidualBlock1D(base_channels, base_channels * 2, time_emb_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # 104 -> 52

        self.down2 = ResidualBlock1D(base_channels * 2, base_channels * 4, time_emb_dim)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # 52 -> 26

        # Bottleneck
        self.mid1 = ResidualBlock1D(base_channels * 4, base_channels * 4, time_emb_dim)
        self.mid2 = ResidualBlock1D(base_channels * 4, base_channels * 4, time_emb_dim)

        # Decoder (Upsample)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')  # 26 -> 52
        self.up_conv2 = ResidualBlock1D(base_channels * 4 + base_channels * 4, base_channels * 2, time_emb_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')  # 52 -> 104
        self.up_conv1 = ResidualBlock1D(base_channels * 2 + base_channels * 2, base_channels, time_emb_dim)

        # Output projection
        self.final_conv = nn.Sequential(
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, 1, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 104)
        t_emb = self.time_mlp(timestep)

        x0 = self.init_conv(x)          # (B, 32, 104)

        d1 = self.down1(x0, t_emb)      # (B, 64, 104)
        p1 = self.pool1(d1)             # (B, 64, 52)

        d2 = self.down2(p1, t_emb)      # (B, 128, 52)
        p2 = self.pool2(d2)             # (B, 128, 26)

        m = self.mid1(p2, t_emb)        # (B, 128, 26)
        m = self.mid2(m, t_emb)         # (B, 128, 26)

        u2 = self.up2(m)                # (B, 128, 52)
        u2 = torch.cat([u2, d2], dim=1) # (B, 256, 52)
        u2 = self.up_conv2(u2, t_emb)   # (B, 64, 52)

        u1 = self.up1(u2)               # (B, 64, 104)
        u1 = torch.cat([u1, d1], dim=1) # (B, 128, 104)
        u1 = self.up_conv1(u1, t_emb)   # (B, 32, 104)

        out = self.final_conv(u1)       # (B, 1, 104)
        return out


class Diffusion1D:
    """
    Modified Diffusion Schedule & Sampler for Malware Embeddings.
    Incorporates the paper's key modification:
    - Linear Beta Schedule with T_max = 1000
    - Truncated Forward & Reverse Process up to T/2 = 500 steps (retaining signal).
    """
    def __init__(self, timesteps: int = 1000, max_truncated_step: int = 500, beta_start: float = 0.0001, beta_end: float = 0.02, device: torch.device = torch.device('cpu')):
        self.timesteps = timesteps
        self.T_half = max_truncated_step
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    def forward_diffusion(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q(x_t | x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*epsilon
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    @torch.no_grad()
    def sample(self, model: nn.Module, real_anchor: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Algorithm 1: Truncated Reverse Diffusion Sampling
        Input: real_anchor (subset of real embeddings to seed forward noise up to T/2)
        Output: synthetic samples x_0
        """
        model.eval()
        # Seed initial noisy state from real samples perturbed to T_half
        # If real_anchor is provided, pick random samples and noise them to T_half
        idx = torch.randint(0, len(real_anchor), (num_samples,), device=self.device)
        anchor_batch = real_anchor[idx].unsqueeze(1) # (B, 1, 104)

        t_start = torch.full((num_samples,), self.T_half - 1, device=self.device, dtype=torch.long)
        x_t, _ = self.forward_diffusion(anchor_batch, t_start)

        # Reverse loop from T_half down to 1
        for t in reversed(range(1, self.T_half)):
            t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            predicted_noise = model(x_t, t_tensor)

            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]

            # Equation from Algorithm 1
            mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise)
            if t > 1:
                z = torch.randn_like(x_t)
                sigma_t = torch.sqrt(beta_t)
                x_t = mean + sigma_t * z
            else:
                x_t = mean

        # Squeeze to (num_samples, 104) and clamp to [-1, 1]
        x_0 = torch.clamp(x_t.squeeze(1), -1.0, 1.0)
        return x_0
