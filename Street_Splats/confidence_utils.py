import torch
from torch import Tensor
from typing import Tuple

def confidence_values(splats) -> Tensor:
    """
    Returns sigmoid(confs + conf_bias)  ∈  (0,1), shape [N].
    """
    conf_logits = splats["confs"].squeeze(-1) + splats["conf_bias"].squeeze(-1)
    return torch.sigmoid(conf_logits)
