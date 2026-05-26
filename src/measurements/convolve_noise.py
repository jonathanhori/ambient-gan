"""
measurements/convolve_noise.py
Convolve + Noise measurement model from the AmbientGAN paper.
Gaussian kernel of radius 1 with additive Gaussian noise with zero mean and SD sigma. 
"""

import torch
from torchvision import transforms
 

def convolve_noise(x, noise_sigma = 0.5, kernel_size = 3):
    """
    Apply Convolve+Noise measurement to a batch of images.
    Uses a fixed Gaussian blur with kernel size 3 (radius 1 pixel) by default,
    followed by additive Gaussian noise with SD sigma.
 
    Args:
        x:           image tensor of shape (batch_size, 1, 28, 28),
                     values in [-1, 1]
        noise_sigma: standard deviation of additive Gaussian noise
 
    Returns:
        y: corrupted image tensor, same shape as x
    """
    blur    = transforms.GaussianBlur(kernel_size=kernel_size, sigma = 1)
    blurred = blur(x)
    noise   = torch.normal(mean=0, std=noise_sigma, size=blurred.shape).to(blurred.device)
    return blurred + noise
