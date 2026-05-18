"""
models.py — AmbientGAN Generator and Discriminator for MNIST
=============================================================
Architecture: WGAN-GP (Wasserstein GAN with Gradient Penalty)
Reference:    Bora et al. 2018 (AmbientGAN), Gulrajani et al. 2017 (WGAN-GP)
Dataset:      MNIST, 28x28 grayscale images

Conventions used throughout:
  B  = batch size
  C  = number of channels
  H  = height
  W  = width
  z  = latent vector, shape (B, latent_dim)

Quick shape sanity check:
  >>> import torch
  >>> G = Generator()
  >>> D = Discriminator()
  >>> z = torch.randn(4, 128)        # batch of 4
  >>> x = G(z)
  >>> print(x.shape)                 # should be (4, 1, 28, 28)
  >>> print(D(x).shape)              # should be (4, 1)
"""

import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LATENT_DIM   = 128   # dimensionality of the noise vector z
MODEL_DIM    = 64    #
IMAGE_SIZE   = 28    # MNIST images are 28x28
IMAGE_CHANNELS = 1   # grayscale


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    Maps a latent vector z ~ Uniform[-1, 1]^latent_dim to a 28x28 image.

    Architecture (WGAN-GP paper, MNIST variant, appendix Section 10.5):
      Linear → reshape → 3x DeconvBlock → Sigmoid

    The output is in [-1, 1]. During training the dataloader should also
    normalize real MNIST images to [-1, 1] so the scales match.

    Forward pass shapes:
      Input:  (B, latent_dim)
      Output: (B, 1, 28, 28)
    """

    def __init__(self, 
                 latent_dim: int = LATENT_DIM, 
                 model_dim: int = MODEL_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        
        self.linear_input = nn.Sequential(
            # Input is latent Z: (B, latent_dim) = (64, 128)
            nn.Linear(latent_dim, 4 * 4 * 4 * self.model_dim),
            nn.ReLU()
        )
        
        self.deconv1 = nn.Sequential(
            # Input is (B, 4 * self.model_dim, 4, 4) = (64, 256, 4, 4)). 4 * self.model_dim input channels
            nn.ConvTranspose2d(4 * self.model_dim, 2 * self.model_dim, 
                               kernel_size = 5, stride = 2, padding = 1, output_padding = 1),
            nn.ReLU()
        )
        
        self.deconv2 = nn.Sequential(
            # Input is (B, 2 * self.model_dim, 7, 7) = (64, 128, 7, 7))
            nn.ConvTranspose2d(2 * self.model_dim, self.model_dim,
                               kernel_size = 5, stride = 2, padding = 2, output_padding = 1),
            nn.ReLU()
        )
        
        self.deconv3 = nn.Sequential(
            # Input is (B, self.model_dim, 7, 7) = (64, 64, 14, 14))
            nn.ConvTranspose2d(self.model_dim, 1,
                               kernel_size = 5, stride = 2, padding = 2, output_padding = 1)
        )


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) — sample via torch.rand(B, latent_dim)*2 - 1
                                  (Uniform[-1,1], matching the paper)
        Returns:
            x: (B, 1, 28, 28)  — generated image, values in [-1, 1]
        """
        B = z.shape[0] # minibatch size
        
        # Output is projected and reshaped to have 4 * model_dim channels
        h = self.linear_input(z).view(B, 4 * self.model_dim, 4, 4)
        # Output will have 2 * model_dim channels
        h = self.deconv1(h)
        h = h[:, :, :7, :7] # Crop so upsampling matches 1 x 28 x 28 at end of generator
        h = self.deconv2(h)
        h = self.deconv3(h)
        return torch.sigmoid(h)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    Maps a 28x28 image (real measurement or fake measurement) to a scalar
    score. Higher score = more likely real. No sigmoid — WGAN-GP uses raw
    logits, not probabilities.

    Architecture (WGAN-GP paper, MNIST variant, appendix Section 10.5):
      3x ConvBlock → flatten → Linear → scalar output
      
    Convolutional layers mirror those from Generator

    IMPORTANT: No BatchNorm in the discriminator for WGAN-GP.
    The gradient penalty enforces the Lipschitz constraint instead.
    Using BatchNorm in D breaks the penalty. Use LayerNorm or nothing.

    Forward pass shapes:
      Input:  (B, 1, 28, 28)
      Output: (B, 1)             ← raw score, no activation
    """

    def __init__(self, model_dim: int = MODEL_DIM):
        super().__init__()
        
        self.model_dim = model_dim
        
        self.conv1 = nn.Sequential(
            # Input is measurement y: (B, 1, 28, 28)
            nn.Conv2d(1, self.model_dim, kernel_size = 5, stride = 2, padding = 2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.conv2 = nn.Sequential(
            # Input is (B, model_dim, 14, 14)
            nn.Conv2d(self.model_dim, 2 * self.model_dim, 
                      kernel_size = 5, stride = 2, padding = 2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.conv3 = nn.Sequential(
            # Input is (B, 2 * model_dim, 7, 7)
            nn.Conv2d(2 * self.model_dim, 4 * self.model_dim, 
                      kernel_size = 5, stride = 2, padding = 2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.linear_output = nn.Linear(4 * 4 * 4 * self.model_dim, 1)


    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 28, 28) — either a real measurement or f(G(z))
        Returns:
            score: (B, 1) — raw scalar, no activation
        """
        B = y.shape[0]
        
        h = self.conv1(y)
        h = self.conv2(h)
        h = self.conv3(h)
        h = h.view(B, -1)
        
        out = self.linear_output(h)
        return out


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------

def initialize_weights(module: nn.Module) -> None:
    """
    Apply to both G and D after construction:
        G.apply(initialize_weights)
        D.apply(initialize_weights)

    WGAN-GP is less sensitive to weight init than vanilla GAN, but the
    DCGAN-style init (normal with mean=0, std=0.02) is standard.
    Batch norm weight is initialized to 1, bias to 0.
    
    The original model implementation from the authors implement some alternative 
        initializations for each layer. We start by the default initialization.
        - Linear: line 55 in src/mnist/gen/wganlib/linear.py
        - Conv2d/Deconv2d: line 49-55 in src/mnist/gen/wganlib/deconv2d.py
        
    """
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(module.weight, gain=math.sqrt(2))
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Sanity check - test dimensions
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    G = Generator(latent_dim=LATENT_DIM)
    D = Discriminator()

    G.apply(initialize_weights)
    D.apply(initialize_weights)

    # --- shape checks ---
    B = 4
    z = torch.rand(B, LATENT_DIM) * 2 - 1      # Uniform[-1, 1]
    x_fake = G(z)
    score  = D(x_fake)

    assert x_fake.shape == (B, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), \
        f"Generator output shape wrong: {x_fake.shape}"
    assert score.shape == (B, 1), \
        f"Discriminator output shape wrong: {score.shape}"
    assert x_fake.min() >= -1.0 and x_fake.max() <= 1.0, \
        "Generator output out of [-1, 1] range — check Tanh"

    print("All shape checks passed.")
    print(f"  G output: {x_fake.shape}   min={x_fake.min():.3f} max={x_fake.max():.3f}")
    print(f"  D output: {score.shape}    min={score.min():.3f} max={score.max():.3f}")

    # --- parameter counts ---
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"\n  Generator parameters:     {g_params:,}")
    print(f"  Discriminator parameters: {d_params:,}")

    # --- gradient flow check ---
    # Make sure gradients can flow back through D to G (needed for G update)
    score_fake = D(G(torch.rand(B, LATENT_DIM) * 2 - 1))
    loss_G = -score_fake.mean()
    loss_G.backward()
    print("\n  Gradient flow check passed (G → D backward ok).")
