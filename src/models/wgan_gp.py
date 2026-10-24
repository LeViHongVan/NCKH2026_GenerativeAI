import torch
import torch.nn as nn

class Conv1DWGANGenerator(nn.Module):
    """
    WGAN-GP Generator using Conv1D and Dense layers.
    Architecture based on Table 5 in IEEE Access 2025 Paper:
    - Input: Noise vector z (latent_dim)
    - Reshape to 1D channel sequence
    - Conv1D (filter 64, kernel 3, padding same) + BatchNorm + LeakyReLU
    - Conv1D (filter 32, kernel 3, padding same) + BatchNorm + LeakyReLU
    - Conv1D (filter 16, kernel 3, padding same) + BatchNorm + LeakyReLU
    - Flatten -> Dense(output_dim) -> Tanh (scales to [-1, 1])
    """
    def __init__(self, latent_dim: int = 64, output_dim: int = 104):
        super(Conv1DWGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.initial = nn.Sequential(
            nn.Linear(latent_dim, 64 * 32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.final_dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 32, output_dim),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.initial(z)
        x = x.view(-1, 64, 32)
        x = self.conv_blocks(x)
        out = self.final_dense(x)
        return out


class Conv1DWGANCritic(nn.Module):
    """
    WGAN-GP Critic (Discriminator) using Conv1D and Dense layers.
    Architecture based on Table 5 in IEEE Access 2025 Paper:
    - Input: Embedding vector x (input_dim = 104)
    - Reshape / Linear to Conv1D sequence
    - Conv1D (filter 64, kernel 3, padding same) + LeakyReLU
    - Conv1D (filter 128, kernel 3, padding same) + LeakyReLU
    - Conv1D (filter 256, kernel 3, padding same) + LeakyReLU
    - Flatten -> Dense(1) (No sigmoid, output unbounded Wasserstein score)
    """
    def __init__(self, input_dim: int = 104):
        super(Conv1DWGANCritic, self).__init__()
        self.input_dim = input_dim

        self.initial = nn.Sequential(
            nn.Linear(input_dim, 64 * 32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.final_dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.initial(x)
        h = h.view(-1, 64, 32)
        h = self.conv_blocks(h)
        score = self.final_dense(h)
        return score


def compute_gradient_penalty(critic: nn.Module, real_samples: torch.Tensor, fake_samples: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Computes the Gradient Penalty for WGAN-GP.
    GP = E[ (||nabla_{x_hat} D(x_hat)||_2 - 1)^2 ]
    """
    alpha = torch.rand((real_samples.size(0), 1), device=device)
    alpha = alpha.expand_as(real_samples)

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)

    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty
