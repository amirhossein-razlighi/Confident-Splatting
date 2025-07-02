import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple

def confidence_with_gumbel(splats, temperature: float = 2.0, seed: int = None) -> Tensor:
    """
    Returns confidence values perturbed by Gumbel noise.
    Args:
        splats: dict with keys "conf_alpha" and "conf_beta"
        temperature: scaling factor for Gumbel noise
        seed: optional seed for reproducibility
    Returns:
        Tensor of shape [N], with Gumbel-perturbed confidence scores
    """
    if seed is not None:
        torch.manual_seed(seed)

    alpha = F.softplus(splats["conf_alpha"]) + 1e-6
    beta  = F.softplus(splats["conf_beta"]) + 1e-6
    conf  = alpha / (alpha + beta)  # Expected confidence

    # Add Gumbel noise
    gumbel_noise = sample_gumbel(conf.shape, device=conf.device)
    perturbed_conf = F.sigmoid(conf * gumbel_noise / temperature)
    return perturbed_conf.squeeze(-1)

def confidence_values(splats) -> Tensor:
    """
    Returns expected value of beta distribution -> alpha / (alpha + beta)
    """
    alpha = F.softplus(splats["conf_alpha"]) + 1e-6
    beta  = F.softplus(splats["conf_beta"]) + 1e-6
    conf  = alpha / (alpha + beta)  # Expected confidence
    return conf.squeeze(-1)

def sample_gumbel(shape, eps=1e-8, device="cuda"):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)
