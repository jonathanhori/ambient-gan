
import torch
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