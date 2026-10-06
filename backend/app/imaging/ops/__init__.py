"""Operation registry.

To add a new tool (for example a noise model or a filter):
  1. write the function in a module in this package,
  2. add its parameter schema in app/schemas.py,
  3. register it in OPERATIONS below (geometric=True if it changes the
     image size, so the alpha channel is transformed with it).
The /process and /export endpoints then accept it with no other changes.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import basic, filters, geometry
from .geometry import OperationError


@dataclass(frozen=True)
class OpSpec:
    fn: Callable[..., np.ndarray]
    geometric: bool = False


OPERATIONS: dict[str, OpSpec] = {
    "crop": OpSpec(geometry.crop, geometric=True),
    "resize": OpSpec(geometry.resize, geometric=True),
    "grayscale": OpSpec(basic.grayscale),
    "brightness": OpSpec(basic.brightness),
    "contrast": OpSpec(basic.contrast),
    "gaussian_blur": OpSpec(filters.gaussian_blur),
    "median": OpSpec(filters.median),
}


def run_pipeline(rgb: np.ndarray, alpha: np.ndarray | None, operations: list):
    """Apply operations in order, starting from the original image.

    Returns (rgb, alpha). Geometric operations are applied to the alpha
    channel too; intensity operations and filters leave it unchanged.
    """
    out = rgb.astype(np.float64)
    a = alpha.astype(np.float64) if alpha is not None else None
    for op in operations:
        spec = OPERATIONS[op.type]
        params = op.params.model_dump() if op.params is not None else {}
        if spec.geometric and a is not None:
            both = spec.fn(np.dstack([out, a]), **params)
            out, a = both[..., :3], both[..., 3]
        else:
            out = spec.fn(out, **params)
    return out, (np.clip(a, 0, 255).astype(np.uint8) if a is not None else None)


__all__ = ["OPERATIONS", "OperationError", "run_pipeline"]
