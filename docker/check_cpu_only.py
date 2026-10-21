"""Build-time check: fail if any CUDA/GPU package is installed in the image (LOG-016)."""

import sys
from importlib.metadata import distributions, version

GPU_PREFIXES = ("nvidia-", "nvidia_", "cuda-", "cuda_", "triton")

names = sorted({d.metadata["Name"].lower() for d in distributions()})
gpu = [n for n in names if n.startswith(GPU_PREFIXES)]
torch_version = version("torch")
if gpu or not torch_version.endswith("+cpu"):
    sys.exit(f"GPU packages in the image: {gpu}; torch {torch_version}")
print(f"CPU-only: torch {torch_version}, {len(names)} packages, none CUDA")
