import random
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path):
    if path is None:
        return None
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
