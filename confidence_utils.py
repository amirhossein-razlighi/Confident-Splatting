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

    conf = confidence_values(splats)

    # Add Gumbel noise
    gumbel_noise = sample_gumbel(conf.shape, device=conf.device)
    perturbed_conf = F.sigmoid(conf * gumbel_noise / temperature)
    return perturbed_conf

def confidence_values(splats) -> Tensor:
    """
    Return per-splat confidence values.

    Beta mode uses E[Beta(alpha, beta)] = alpha / (alpha + beta).
    Scalar mode uses a learned sigmoid logit. Supporting both lets us test
    whether the Beta parameterization itself is useful or whether any learned
    scalar gate is enough.
    """
    if "conf_logit" in splats:
        return torch.sigmoid(splats["conf_logit"]).squeeze(-1)

    if "conf_alpha" not in splats or "conf_beta" not in splats:
        raise KeyError(
            "Expected either 'conf_logit' or both 'conf_alpha' and 'conf_beta' "
            "in the splat dictionary."
        )

    alpha = F.softplus(splats["conf_alpha"]) + 1e-6
    beta = F.softplus(splats["conf_beta"]) + 1e-6
    return (alpha / (alpha + beta)).squeeze(-1)


def beta_entropy(splats) -> Tensor:
    """Return the mean entropy of learned Beta confidence distributions."""
    if "conf_alpha" not in splats or "conf_beta" not in splats:
        raise KeyError("Beta entropy requires 'conf_alpha' and 'conf_beta'.")
    alpha = F.softplus(splats["conf_alpha"]) + 1e-6
    beta = F.softplus(splats["conf_beta"]) + 1e-6
    return torch.distributions.Beta(alpha, beta).entropy().mean()

def sample_gumbel(shape, eps=1e-8, device="cuda"):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)
