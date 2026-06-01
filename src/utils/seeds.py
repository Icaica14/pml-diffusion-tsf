"""Deterministic seeding.

One call seeds every RNG we might touch (Python, NumPy, and — if installed —
PyTorch). The seed is read from the config and logged into the run manifest, so
every result in `results/registry.csv` is reproducible (CONTRIBUTING.md: "Seed
set and logged").
"""

from __future__ import annotations

import os
import random
import warnings

import numpy as np


def set_seed(seed: int) -> int:
    """Seed Python, NumPy, and (if available) PyTorch. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Torch is part of the heavy/Colab group; seed it only if it imports cleanly so
    # the light local env (M0/M1, pure NumPy/statsmodels) has no hard torch dependency.
    # We catch *any* exception, not just ImportError: a torch that is installed but
    # broken — e.g. a wheel compiled against NumPy 1.x raising on `import torch` under
    # NumPy 2.x — must be treated exactly like an absent torch, so it can never take
    # down a CPU-only seasonal-naive/ARIMA run. On Colab (torch healthy) this is a no-op.
    # The warnings filter mutes the cosmetic "Failed to initialize NumPy: _ARRAY_API
    # not found" ABI notice such a mismatched-but-usable wheel prints on import.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001 — broken-but-present torch == absent torch here
        pass

    return seed
