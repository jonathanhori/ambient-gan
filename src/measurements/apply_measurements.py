"""
measurements/apply_measurement.py
Utility function that applies the correct measurement model to a minibatch of MNIST
images before training.  
"""

import torch
from torchvision import transforms
from measurements.block_pixels import block_pixels
from measurements.convolve_noise import convolve_noise

def apply_measurement(images, cfg):
    """
    Apply the correct measurement function based on cfg['type'].
 
    Args:
        x:   image tensor of shape (batch_size, 1, 28, 28)
        cfg: dict with key 'type' and measurement-specific params:
               block_pixels:   {'type': 'block_pixels', 'block_prob': float}
               convolve_noise: {'type': 'convolve_noise', 'noise_sigma': float}
 
    Returns:
        corrupted images, same shape as input
    """
    if cfg['type'] == 'block_pixels':
        return block_pixels(images, p=cfg['block_prob'])
    elif cfg['type'] == 'convolve_noise':  # convolve_noise
        return convolve_noise(images, noise_sigma=cfg['noise_sigma'])
    elif cfg['type'] == 'identity': #identity for testing purposes
        return images
    else:
        raise ValueError(f"Unknown measurement type: '{cfg['type']}'. "
                         f"Expected 'block_pixels' or 'convolve_noise'.")