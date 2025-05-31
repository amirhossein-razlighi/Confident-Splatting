import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple

def confidence_values_with_gumbel_noise(splats, conf_temperature) -> Tensor:
    """
    Returns sigmoid(confs + conf_bias)  ∈  (0,1), shape [N].
    """
    gumbel_noise = sample_gumbel(splats["confs"].shape, device="cuda")
    scaled_logits = (splats["confs"] + gumbel_noise) / conf_temperature
    conf_vals = torch.sigmoid(scaled_logits).squeeze(-1)
    return conf_vals

def confidence_values(splats) -> Tensor:
    """
    Returns expected value of beta distribution -> alpha / (alpha + beta)
    """
    # conf_logits = splats["confs"].squeeze(-1) + splats["conf_bias"].squeeze(-1)
    # return torch.sigmoid(conf_logits)
    alpha = F.softplus(splats["conf_alpha"]) + 1e-6
    beta  = F.softplus(splats["conf_beta"]) + 1e-6
    conf  = alpha / (alpha + beta)  # Expected confidence
    return conf.squeeze(-1)

def sample_gumbel(shape, eps=1e-8, device="cuda"):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)
