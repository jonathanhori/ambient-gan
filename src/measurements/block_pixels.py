"""
measurements/block_pixels.py
Block-Pixels measurement model from the AmbientGAN paper.
Each pixel is independently set to zero with probability p.
"""
 
import torch
from torchvision import transforms
 
 
def block_pixels(x, p):
    B, C, H, W = x.shape
    mask = torch.bernoulli(torch.ones(B, 1, H, W, device=x.device) * (1 - p))
    return x * mask  # broadcasts across C
