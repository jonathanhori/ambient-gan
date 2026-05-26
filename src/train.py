"""
train.py — AmbientGAN WGAN-GP training loop for MNIST
======================================================

Usage:
  # Single run
  python train.py --config configs/mnist_blockpixels.yaml

  # Sweep for figure 7 (vary block_prob)
  python train.py --config configs/mnist_blockpixels.yaml \
      --override measurement.block_prob=0.9 experiment_name=mnist_bp_p09
"""

import os
import math
import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model import Generator, Discriminator, initialize_weights, LATENT_DIM
from measurements.apply_measurements import apply_measurement  # assumed available


# ---------------------------------------------------------------------------
# WGAN-GP loss functions
# ---------------------------------------------------------------------------

def gradient_penalty(D: nn.Module, 
                      real: torch.Tensor, 
                      fake: torch.Tensor, 
                      device: torch.device) -> torch.Tensor:
    """
    Computes the gradient penalty term from Gulrajani et al. 2017.

    Interpolates between real and fake measurements, passes through D,
    and penalises the gradient norm deviating from 1.
    
    Steps:
      1. Sample alpha ~ Uniform[0,1] of shape (B, 1, 1, 1)
      2. Compute interpolated = alpha * real + (1 - alpha) * fake
      3. Set requires_grad=True on interpolated
      4. Pass interpolated through D to get d_interp
      5. Compute gradients of d_interp w.r.t. interpolated using
         torch.autograd.grad — remember create_graph=True
      6. Compute gradient norm per sample, return mean((norm - 1)^2)

    Args:
        D:      Discriminator
        real:   (B, 1, 28, 28) real measurements
        fake:   (B, 1, 28, 28) fake measurements f(G(z))
        device: torch device

    Returns:
        penalty: scalar tensor
    """

    B = real.shape[0]

    # Random interpolation weight, one per sample in the batch
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    d_interp = D(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,   # needed so penalty itself is differentiable
        retain_graph=True,
    )[0]

    # Flatten per sample and compute L2 norm
    gradients = gradients.view(B, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

def discriminator_loss(d_real: torch.Tensor, 
                       d_fake: torch.Tensor, 
                       gp: torch.Tensor, 
                       lambda_gp: float) -> torch.Tensor:
    """
    WGAN-GP discriminator loss.
    D wants to maximise d_real - d_fake, so we minimise fake - real.

    Args:
        d_real:    D scores for real measurements, (B, 1)
        d_fake:    D scores for fake measurements, (B, 1)
        gp:        gradient penalty scalar
        lambda_gp: penalty coefficient (default 10)

    Returns:
        loss: scalar tensor
    """
    return d_fake.mean() - d_real.mean() + lambda_gp * gp

def generator_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """
    WGAN-GP generator loss.
    G wants to maximise d_fake, so we minimise -d_fake.

    Args:
        d_fake: D scores for fake measurements f(G(z)), (B, 1)

    Returns:
        loss: scalar tensor
    """
    return -d_fake.mean()


# ---------------------------------------------------------------------------
# Latent sampling
# ---------------------------------------------------------------------------

def sample_latent(batch_size: int, 
                  latent_dim: int, 
                  device: torch.device) -> torch.Tensor:
    """
    Sample z ~ Uniform[-1, 1]^latent_dim, matching the paper.
    
    Uniform(0, 1) * 2 - 1 ~ Uniform(-1, 1)

    Args:
        batch_size: B
        latent_dim: dimensionality of z
        device:     torch device

    Returns:
        z: (B, latent_dim)
    """
    return torch.rand(batch_size, latent_dim, device = device) * 2 - 1 

# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(G: nn.Module,
               D: nn.Module,
               opt_G: torch.optim.Optimizer,
               opt_D: torch.optim.Optimizer,
               real_images: torch.Tensor,
               measurement_cfg: dict,
               lambda_gp: float,
               n_critic: int,
               latent_dim: int,
               device: torch.device) -> dict:
    """
    One full training step: n_critic discriminator updates then one
    generator update. This is the core AmbientGAN training loop.

    The key difference from standard GAN training:
      Standard GAN:  D sees real images      vs G(z)
      AmbientGAN:    D sees real measurements vs f(G(z))

    Note: real_images come in clean from the dataloader. The measurement
    is applied here, so D never sees a clean image on either side.

    Args:
        G:               Generator
        D:               Discriminator
        opt_G:           Generator optimizer
        opt_D:           Discriminator optimizer
        real_images:     (B, 1, 28, 28) clean images from dataloader
        measurement_cfg: dict specifying measurement type and params
        lambda_gp:       gradient penalty coefficient
        n_critic:        number of D updates per G update
        latent_dim:      dimensionality of z
        device:          torch device

    Returns:
        dict with keys 'loss_D' and 'loss_G' (floats, for logging)
    """
    B = real_images.shape[0]
    real_images = real_images.to(device)

    # Apply measurement to real images so D sees real measurements
    real_measurements = apply_measurement(real_images, type = measurement_cfg['type']) #prob?

    # ------------------------------------------------------------------
    # Discriminator updates (n_critic steps)
    # ------------------------------------------------------------------
    for _ in range(n_critic):

        # Sample z and generate fake images
        # Detach fake_images from G's computation graph for D update
        z = sample_latent(batch_size = B, latent_dim = latent_dim, device = device)
        fake_images = G(z).detach()

        # Apply measurement to fake images
        fake_measurements = apply_measurement(fake_images, type = measurement_cfg['type'])

        # Compute D scores
        d_real = D(real_measurements)
        d_fake = D(fake_measurements)

        # Compute gradient penalty
        gp = gradient_penalty(D, real_images, fake_images, device = device)

        # Compute and backprop D loss
        loss_D = discriminator_loss(d_real, d_fake, gp = gp, lambda_gp = lambda_gp)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

    # ------------------------------------------------------------------
    # Generator update (one step)
    # ------------------------------------------------------------------

    # Sample fresh z and generate fake images
    z = sample_latent(batch_size = B, latent_dim = latent_dim, device = device)
    fake_images = G(z)

    # Apply measurement
    fake_measurements = apply_measurement(fake_images, type = measurement_cfg['type'])

    # Compute D score on fake measurements
    d_fake = D(fake_measurements)

    # Compute and backprop G loss
    loss_G = generator_loss(d_fake)
    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    return {'loss_D': loss_D.item(), 'loss_G': loss_G.item()}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict) -> None:
    """
    Full training loop. Saves checkpoints and results for Person C
    to use for figure generation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['training']['seed'])

    # Setup output directories
    exp_name = config['experiment_name']
    results_dir = os.path.join(config['output']['results_dir'], exp_name)
    ckpt_dir    = os.path.join(config['output']['checkpoint_dir'], exp_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save exact config used — critical for reproducibility
    with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    # MNIST normalized to [0, 1] to match generator's sigmoid output
    # Note: do NOT apply measurement here — that happens in train_step
    if config['dataset']['name'] == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),  # scales to [0, 1]
        ])
        dataset = datasets.MNIST(
            root=config['dataset']['data_dir'],
            train=True,
            download=True,
            transform=transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config['dataset']['batch_size'],
            shuffle=True,
            drop_last=True,    # keeps batch size fixed — important for WGAN-GP
            num_workers=2,
            pin_memory=True
        )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    G = Generator(latent_dim=config['model']['latent_dim']).to(device)
    D = Discriminator().to(device)
    G.apply(initialize_weights)
    D.apply(initialize_weights)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    # Adam with beta1=0, beta2=0.9 — standard for WGAN-GP
    # Note: beta1=0 (not 0.9) is important — momentum interferes with
    # the Wasserstein critic updates
    opt_G = torch.optim.Adam(
        G.parameters(),
        lr=config['training']['lr_g'],
        betas=(0.0, 0.9)
    )
    opt_D = torch.optim.Adam(
        D.parameters(),
        lr=config['training']['lr_d'],
        betas=(0.0, 0.9)
    )

    # ------------------------------------------------------------------
    # Results log — Person C reads this to make figure 7
    # ------------------------------------------------------------------
    results = {
        'epoch': [],
        'inception_score_mean': [],
        'inception_score_std': []
    }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    n_epochs      = config['training']['n_epochs']
    n_critic      = config['training']['n_critic']
    lambda_gp     = config['training']['lambda_gp']
    latent_dim    = config['model']['latent_dim']
    eval_every    = config['evaluation']['inception_score_every_n_epochs']
    save_every    = config['training']['save_every_n_epochs']
    measurement_cfg = config['measurement']

    for epoch in range(1, n_epochs + 1):
        G.train()
        D.train()

        for batch_idx, (real_images, _) in enumerate(dataloader):
            losses = train_step(
                G=G, D=D,
                opt_G=opt_G, opt_D=opt_D,
                real_images=real_images,
                measurement_cfg=measurement_cfg,
                lambda_gp=lambda_gp,
                n_critic=n_critic,
                latent_dim=latent_dim,
                device=device
            )

        # Evaluate
        if epoch % eval_every == 0:
            G.eval()
            # TODO: call inception score function here
            # is_mean, is_std = compute_inception_score(G, n_samples, device)
            is_mean, is_std = 0.0, 0.0  # replace with actual IS computation

            results['epoch'].append(epoch)
            results['inception_score_mean'].append(is_mean)
            results['inception_score_std'].append(is_std)

            print(f"Epoch {epoch}/{n_epochs}  "
                  f"loss_D={losses['loss_D']:.3f}  "
                  f"loss_G={losses['loss_G']:.3f}  "
                  f"IS={is_mean:.2f}±{is_std:.2f}")

            with open(os.path.join(results_dir, 'results.yaml'), 'w') as f:
                yaml.dump(results, f)

        # Checkpoint
        if epoch % save_every == 0:
            torch.save(G.state_dict(),
                       os.path.join(ckpt_dir, f'G_epoch{epoch}.pt'))
            torch.save(D.state_dict(),
                       os.path.join(ckpt_dir, f'D_epoch{epoch}.pt'))

    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def apply_overrides(config: dict, overrides: list) -> dict:
    for override in overrides:
        key_path, value = override.split('=')
        keys = key_path.split('.')
        d = config
        for k in keys[:-1]:
            d = d[k]
        try:
            value = float(value) if '.' in value else int(value)
        except ValueError:
            pass
        d[keys[-1]] = value
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   required=True)
    parser.add_argument('--override', nargs='*', default=[])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = apply_overrides(config, args.override or [])

    train(config)