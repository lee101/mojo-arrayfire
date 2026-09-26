"""Public surface of mojo_arrayfire: ArrayFire metadata and index arithmetic.

The real `arrayfire` package needs the ArrayFire C library, which is not
installed here, so this package is deliberately narrow: it ports the Python-side
arithmetic that ArrayFire's binding performs on every call (slice -> `Seq`,
assignment shapes, host buffer marshalling, broadcast shape and index maps) and
leaves every `af_*` device call to the real package.
"""

import numpy as np

from ._lib import (
    NONE,
    assign_dims,
    broadcast_index,
    broadcast_shape,
    reorder2d,
    seq_batch,
    seq_from_int,
    slice_lengths,
    strides4,
)

# ArrayFire's `Dtype` enum values, from arrayfire/library.py.
DTYPE_VALUES = {
    "f32": 0, "c32": 1, "f64": 2, "c64": 3, "b8": 4, "s32": 5, "u32": 6,
    "u8": 7, "s64": 8, "u64": 9, "s16": 10, "u16": 11, "f16": 12,
}

#: `arrayfire.util.number_dtype`: the dtype a bare Python number carries.
NUMBER_DTYPE = {"bool": "b8", "int": "s64", "float": "f64", "complex": "c64"}

__all__ = [
    "seq_batch",
    "seq_from_int",
    "slice_lengths",
    "assign_dims",
    "nested_offsets",
    "reorder2d",
    "broadcast_shape",
    "broadcast_index",
    "strides4",
    "implicit_dtype",
    "get_info",
    "nest",
    "DTYPE_VALUES",
    "NUMBER_DTYPE",
    "NONE",
]


def slice_length(key: slice, dim: int) -> int:
    """`_slice_to_length` for one slice, through the batched kernel."""
    if not isinstance(key, slice):
        raise TypeError(f"expected a slice, got {type(key).__name__}")
    return int(slice_lengths([key], [dim])[0])


def assign_shape(key, idims) -> np.ndarray:
    """The 4-dimensional shape an `a[key] = value` assignment would produce."""
    return assign_dims(key, idims)


def implicit_dtype(number, array_dtype: int) -> int:
    """`arrayfire.util.implicit_dtype`: promote a Python number against an array.

    Upstream takes and returns `Dtype` enum members, which are ints; this
    returns the same int so the two are directly comparable. `DTYPE_VALUES`
    maps a name onto the value.

    A `float` or `complex` literal keeps `f32`/`c32` rather than widening the
    array, which is the only special case upstream encodes.
    """
    if isinstance(number, bool):
        value = DTYPE_VALUES["b8"]
    elif isinstance(number, int):
        value = DTYPE_VALUES["s64"]
    elif isinstance(number, complex):
        value = DTYPE_VALUES["c64"]
    elif isinstance(number, float):
        value = DTYPE_VALUES["f64"]
    else:
        raise TypeError(f"not a Python number: {type(number).__name__}")
    narrow = (DTYPE_VALUES["f32"], DTYPE_VALUES["c32"])
    if value == DTYPE_VALUES["f64"] and array_dtype in narrow:
        return DTYPE_VALUES["f32"]
    if value == DTYPE_VALUES["c64"] and array_dtype in narrow:
        return DTYPE_VALUES["c32"]
    return value


def get_info(dims=None, buf_len: int = 0):
    """`arrayfire.array._get_info`: (numdims, idims) for a shape or a length.

    Four multiplications, so this stays in Python; there is no loop here worth
    compiling.
    """
    if dims:
        numdims = len(dims)
        idims = [1] * 4
        for i, dim in enumerate(dims):
            idims[i] = int(dim)
    elif buf_len != 0:
        idims = [int(buf_len), 1, 1, 1]
        numdims = 1
    else:
        raise RuntimeError("Invalid size")
    return numdims, idims
