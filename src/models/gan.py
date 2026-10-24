import torch
import torch.nn as nn

class DenseGenerator(nn.Module):
    """
    Vanilla GAN Generator adapted for 1D Malware Embeddings.
    Architecture based on Table 5 in IEEE Access 2025 Paper:
    - Input: Noise vector z (latent_dim)
    - Dense(200) -> BN(0.8) -> LeakyReLU
    - Dense(400) -> BN(0.8) -> LeakyReLU
    - Dense(800) -> BN(0.8) -> LeakyReLU
    - Dense(output_dim) -> Tanh (scales outputs to [-1, 1])
    """
    def __init__(self, latent_dim: int = 64, output_dim: int = 104):
        super(DenseGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.BatchNorm1d(200, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(200, 400),
            nn.BatchNorm1d(400, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(400, 800),
            nn.BatchNorm1d(800, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(800, output_dim),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class DenseDiscriminator(nn.Module):
    """
    Vanilla GAN Discriminator adapted for 1D Malware Embeddings.
    Architecture based on Table 5 in IEEE Access 2025 Paper:
    - Input: Embedding vector x (input_dim = 104)
    - Dense(400) -> LeakyReLU
    - Dense(200) -> LeakyReLU
    - Dense(1) -> Sigmoid / Logits
    """
    def __init__(self, input_dim: int = 104):
        super(DenseDiscriminator, self).__init__()
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(400, 200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
