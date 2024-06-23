from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


def pad_sequences(
    samples: list[np.ndarray],
    max_shape: list[int] | None = None,
    min_shape: list[int] | None = None,
    pad_value: int | float | bool = 0,
    dtype: torch.dtype | type | None = None,
    backend: str = "torch",
) -> list[np.ndarray] | list[torch.Tensor]:
    """Pad sequences.

    Parameters
    ----------
    samples : list[np.ndarray]
        List of samples.
    max_shape : list[int] | None
        Maximum shape of the samples.
    pad_value : int | float | bool, optional
        Value to pad with, by default 0.
    """
    if max_shape is None:
        shapes = np.array([x.shape for x in samples])
        max_shape = list(shapes.max(axis=0))
    if min_shape is not None:
        assert len(min_shape) == len(max_shape)
        max_shape = [max(x, y) for x, y in zip(max_shape, min_shape)]
    if backend == "torch":
        assert dtype is None or isinstance(dtype, torch.dtype)
        return [
            F.pad(
                input=torch.tensor(x, dtype=dtype),
                value=pad_value,
                pad=[
                    p
                    for i in reversed(range(len(max_shape)))
                    for p in [0, max_shape[i] - x.shape[i]]
                ],
            )
            for x in samples
        ]
    elif backend == "numpy":
        assert dtype is None or isinstance(dtype, type)
        return [
            np.pad(
                array=np.array(x, dtype=dtype),
                pad_width=[
                    (0, max_shape[i] - x.shape[i]) for i in range(len(max_shape))
                ],
                mode="constant",
                constant_values=pad_value,
            )
            for x in samples
        ]
    else:
        raise ValueError(f"Unsupported backend: {backend}")


class CollateSequences:
    """Collate sequences.

    Parameters
    ----------
    max_shape : list[int] | None, optional
        Maximum shape of the samples, by default None.
    pad_value : int | float | bool, optional
        Value to pad with, by default 0.
    dim : int, optional
        Dimension to stack, by default 0.
    agg_func : Callable, optional
        Aggregation function, by default torch.stack.
    """

    def __init__(
        self,
        max_shape: list[int] | None = None,
        min_shape: list[int] | None = None,
        pad_value: int | float | bool = 0,
        dim: int = 0,
        agg_func: Callable = torch.stack,
    ):
        self.max_shape = max_shape
        self.min_shape = min_shape
        self.pad_value = pad_value
        self.dim = dim
        self.agg_func = agg_func

    def __call__(self, samples: list[np.ndarray]):
        return self.agg_func(
            pad_sequences(
                samples=samples,
                max_shape=self.max_shape,
                min_shape=self.min_shape,
                pad_value=self.pad_value,
            ),
            dim=self.dim,
        )
