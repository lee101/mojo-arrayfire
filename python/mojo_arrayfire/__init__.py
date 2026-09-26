"""mojo-arrayfire: ArrayFire metadata, index and marshalling arithmetic in Mojo.

Importable next to the real `arrayfire` package, which it never imports, so
both can be present at once. The real package needs the ArrayFire C library;
this one does not.
"""

from .core import (
    DTYPE_VALUES,
    NONE,
    NUMBER_DTYPE,
    assign_dims,
    assign_shape,
    broadcast_index,
    broadcast_shape,
    get_info,
    implicit_dtype,
    reorder2d,
    seq_batch,
    seq_from_int,
    slice_length,
    slice_lengths,
    strides4,
)

__all__ = [
    "seq_batch",
    "seq_from_int",
    "slice_length",
    "slice_lengths",
    "assign_dims",
    "assign_shape",
    "reorder2d",
    "broadcast_shape",
    "broadcast_index",
    "strides4",
    "implicit_dtype",
    "get_info",
    "DTYPE_VALUES",
    "NUMBER_DTYPE",
    "NONE",
]
__version__ = "0.1.0"
