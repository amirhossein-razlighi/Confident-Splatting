import os
from gsplat.rendering import rasterization

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0 8.6+PTX 9.0+PTX 8.6 8.0 7.5"

rasterization()