"""
measurements/block_pixels.py

Block-Pixels measurement model from the AmbientGAN paper.
Each pixel is independently set to zero with probability p.
"""

import torch


def block_pixels(x, p):
    mask = torch.bernoulli(torch.ones_like(x) * (1 - p))
    return x * mask