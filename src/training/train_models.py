import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple

from src.models.gan import DenseGenerator, DenseDiscriminator
from src.models.wgan_gp import Conv1DWGANGenerator, Conv1DWGANCritic, compute_gradient_penalty
from src.models.diffusion import UNet1D, Diffusion1D

def train_gan_model(
    data: np.ndarray,
    epochs: int = 40,
    batch_size: int = 32,
    latent_dim: int = 64,
    device: torch.device = torch.device('cpu')
) -> Tuple[DenseGenerator, DenseDiscriminator]:
    """
    Trains Vanilla GAN on 104D malware embeddings.
    """
    tensor_data = torch.tensor(data, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=True, drop_last=True)

    generator = DenseGenerator(latent_dim=latent_dim, output_dim=104).to(device)
    discriminator = DenseDiscriminator(input_dim=104).to(device)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.9))
    opt_D = optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.5, 0.9))

    for epoch in range(1, epochs + 1):
        for real_batch, in loader:
            real_batch = real_batch.to(device)
            bs = real_batch.size(0)

            # Labels with label smoothing (0.9 for real, 0.0 for fake)
            real_label = torch.full((bs, 1), 0.9, device=device)
            fake_label = torch.zeros((bs, 1), device=device)

            # ----------------- Train Discriminator -----------------
            opt_D.zero_grad()
            out_real = discriminator(real_batch)
            loss_real = criterion(out_real, real_label)

            z = torch.randn(bs, latent_dim, device=device)
            fake_batch = generator(z)
            out_fake = discriminator(fake_batch.detach())
            loss_fake = criterion(out_fake, fake_label)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            # ----------------- Train Generator -----------------
            opt_G.zero_grad()
            out_gen = discriminator(fake_batch)
            loss_G = criterion(out_gen, torch.ones((bs, 1), device=device))
            loss_G.backward()
            opt_G.step()

    return generator, discriminator


def train_wgan_gp_model(
    data: np.ndarray,
    epochs: int = 40,
    batch_size: int = 32,
    latent_dim: int = 64,
    n_critic: int = 3,
    device: torch.device = torch.device('cpu')
) -> Tuple[Conv1DWGANGenerator, Conv1DWGANCritic]:
    """
    Trains Conv1D WGAN-GP on 104D malware embeddings.
    """
    tensor_data = torch.tensor(data, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=True, drop_last=True)

    generator = Conv1DWGANGenerator(latent_dim=latent_dim, output_dim=104).to(device)
    critic = Conv1DWGANCritic(input_dim=104).to(device)

    opt_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    opt_C = optim.Adam(critic.parameters(), lr=0.0001, betas=(0.5, 0.9))

    for epoch in range(1, epochs + 1):
        for real_batch, in loader:
            real_batch = real_batch.to(device)
            bs = real_batch.size(0)

            # ----------------- Train Critic -----------------
            for _ in range(n_critic):
                opt_C.zero_grad()
                z = torch.randn(bs, latent_dim, device=device)
                fake_batch = generator(z)

                c_real = critic(real_batch)
                c_fake = critic(fake_batch.detach())

                gp = compute_gradient_penalty(critic, real_batch, fake_batch.detach(), device)
                loss_C = -torch.mean(c_real) + torch.mean(c_fake) + 10.0 * gp
                loss_C.backward()
                opt_C.step()

            # ----------------- Train Generator -----------------
            opt_G.zero_grad()
            z = torch.randn(bs, latent_dim, device=device)
            gen_batch = generator(z)
            loss_G = -torch.mean(critic(gen_batch))
            loss_G.backward()
            opt_G.step()

    return generator, critic


def train_diffusion_model(
    data: np.ndarray,
    epochs: int = 40,
    batch_size: int = 32,
    timesteps: int = 1000,
    device: torch.device = torch.device('cpu')
) -> Tuple[UNet1D, Diffusion1D]:
    """
    Trains 1D U-Net Diffusion Model with truncated T/2 schedule on 104D malware embeddings.
    """
    tensor_data = torch.tensor(data, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=True, drop_last=True)

    diffusion = Diffusion1D(timesteps=timesteps, max_truncated_step=500, device=device)
    model = UNet1D(in_dim=104, base_channels=32).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999))
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        for real_batch, in loader:
            real_batch = real_batch.to(device).unsqueeze(1) # (B, 1, 104)
            bs = real_batch.size(0)

            # Sample random timesteps uniformly up to T/2 (as per paper modification)
            t = torch.randint(1, diffusion.T_half, (bs,), device=device).long()
            x_noisy, noise = diffusion.forward_diffusion(real_batch, t)

            optimizer.zero_grad()
            predicted_noise = model(x_noisy, t)
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()

    return model, diffusion
