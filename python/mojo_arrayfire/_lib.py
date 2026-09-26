"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-arrayfire.so"

#: sentinel for an absent `slice.start` / `slice.stop`; 0 means an absent step
NONE = -4611686018427387904




def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.af_seq_batch.argtypes = [ctypes.c_int64] * 5
    lib.af_slice_lengths.argtypes = [ctypes.c_int64] * 6
    lib.af_assign_dims.argtypes = [ctypes.c_int64] * 7
    lib.af_reorder2d.argtypes = [ctypes.c_int64] * 5
    lib.af_reorder2d.restype = None
    lib.af_broadcast_shape.argtypes = [ctypes.c_int64] * 4
    lib.af_broadcast_index.argtypes = [ctypes.c_int64] * 4
    lib.af_strides4.argtypes = [ctypes.c_int64] * 2
    for name in (
        "af_seq_batch", "af_slice_lengths", "af_assign_dims",
        "af_reorder2d", "af_broadcast_shape",
        "af_broadcast_index", "af_strides4",
    ):
        getattr(lib, name).restype = None
    return lib


lib = _load()


def _addr(a: np.ndarray) -> int:
    """Base address of a buffer for the C ABI.

    Every array handed to a kernel is bound to a local first. An array built
    inline in the call expression has no reference left once its address has
    been taken, and the buffer can be recycled before the kernel reads it.
    """
    return int(a.ctypes.data)


def _i64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def _triples(slices):
    """(starts, stops, steps) as three contiguous arrays for the kernel."""
    enc = [_encode_slice(s) for s in slices]
    return (
        _i64([e[0] for e in enc]),
        _i64([e[1] for e in enc]),
        _i64([e[2] for e in enc]),
    )


def _encode_slice(s: slice):
    if not isinstance(s, slice):
        raise TypeError(f"expected a slice, got {type(s).__name__}")
    return (
        NONE if s.start is None else int(s.start),
        NONE if s.stop is None else int(s.stop),
        0 if s.step is None else int(s.step),
    )


def seq_batch(slices) -> np.ndarray:
    """ArrayFire `Seq(begin, end, step)` triples, one per input slice.

    Port of `arrayfire.index.Seq.__init__`, batched: upstream builds one `Seq`
    per `__getitem__` axis in Python, this builds the whole index record in one
    pass. A plain integer index is upstream's `numbers.Number` branch and is
    `(index, index, 1)`; use `seq_from_int` for it.
    """
    n = len(slices)
    if n == 0:
        return np.empty((0, 3), dtype=np.int64)
    starts, stops, steps = _triples(slices)
    out = np.empty(3 * n, dtype=np.int64)
    lib.af_seq_batch(
        n, _addr(starts), _addr(stops), _addr(steps), _addr(out)
    )
    return out.reshape(n, 3)


def seq_from_int(index: int) -> np.ndarray:
    """The `Seq` upstream builds for an integer index: `(index, index, 1)`."""
    return np.array([[int(index), int(index), 1]], dtype=np.int64)


def slice_lengths(slices, dims) -> np.ndarray:
    """Port of `arrayfire.array._slice_to_length`, batched over one dim each."""
    n = len(slices)
    if n != len(dims):
        raise ValueError(f"{n} slices but {len(dims)} dimensions")
    if n == 0:
        return np.empty(0, dtype=np.int64)
    d = _i64(dims)
    starts, stops, steps = _triples(slices)
    out = np.empty(n, dtype=np.int64)
    lib.af_slice_lengths(
        n, _addr(d), _addr(starts), _addr(stops), _addr(steps), _addr(out)
    )
    return out


def assign_dims(keys, idims) -> np.ndarray:
    """Port of `arrayfire.array._get_assign_dims` for number and slice keys.

    Each key is a sequence of up to four entries, each an integer or a slice,
    and the result is the 4-dimensional shape of the assignment target. Missing
    trailing axes default to a full slice, as upstream does.
    """
    key = list(keys)
    if len(key) > 4:
        raise ValueError("af arrays have at most four dimensions")
    n = 4
    kinds = []
    flat = []
    flat_dims = []
    for axis in range(n):
        entry = key[axis] if axis < len(key) else slice(None)
        dim = int(idims[axis]) if axis < len(idims) else 1
        if isinstance(entry, slice):
            kinds.append(1)
            flat.extend(_encode_slice(entry))
        elif isinstance(entry, (int, np.integer)):
            kinds.append(0)
            flat.extend((0, 0, 0))
        else:
            raise TypeError(
                f"key {axis} is {type(entry).__name__}; af.Array handles need a "
                f"live device and are not supported here"
            )
        flat_dims.append(dim)
    out = np.empty(4, dtype=np.int64)
    kind_buf = _i64(kinds)
    dim_buf = _i64(flat_dims)
    starts = _i64(flat[0::3])
    stops = _i64(flat[1::3])
    steps = _i64(flat[2::3])
    lib.af_assign_dims(
        n, _addr(kind_buf), _addr(dim_buf), _addr(starts), _addr(stops),
        _addr(steps), _addr(out),
    )
    return out


def reorder2d(src: np.ndarray) -> np.ndarray:
    """Row-major 2-D host buffer to the column-major image af expects.

    This is `arrayfire.interop._cc_to_af_array` on the host: the shape is
    reversed and the axes are swapped, which as a flat permutation is
    `dst[i0 + i1*n0] = src[i1 + i0*n1]`. The returned array holds the same
    logical matrix in Fortran order, which is what
    `af_create_array(..., reversed_shape)` wants.
    """
    src = np.ascontiguousarray(src)
    if src.ndim != 2:
        raise ValueError(f"reorder2d needs a 2-D array, got shape {src.shape}")
    if src.dtype.kind not in "iuf":
        raise TypeError(f"reorder2d has no kernel for {src.dtype}")
    n0, n1 = src.shape
    flat = np.empty(src.size, dtype=src.dtype)
    lib.af_reorder2d(n0, n1, src.dtype.itemsize, _addr(src.reshape(-1)), _addr(flat))
    return flat.reshape((n0, n1), order="F")


def broadcast_shape(groups) -> np.ndarray:
    """Broadcast several operand shapes by ArrayFire's rule, batched.

    Each group is a sequence of operands and each operand a 4-tuple of extents.
    A dimension of extent 1 is broadcast against its partner; two dimensions
    above 1 must agree, otherwise the group raises.
    """
    groups = [tuple(tuple(int(x) for x in op) for op in g) for g in groups]
    if not groups:
        return np.empty((0, 4), dtype=np.int64)
    nops = len(groups[0])
    for g in groups:
        if len(g) != nops:
            raise ValueError("every group must have the same operand count")
        for op in g:
            if len(op) != 4:
                raise ValueError("each operand must be a 4-tuple of extents")
    n = len(groups)
    buf = _i64([v for g in groups for op in g for v in op])
    out = np.empty(4 * n, dtype=np.int64)
    lib.af_broadcast_shape(n, nops, _addr(buf), _addr(out))
    result = out.reshape(n, 4)
    bad = (result < 0).any(axis=1)
    if bad.any():
        idx = int(np.argmax(bad))
        raise ValueError(
            f"operands of group {idx} are not broadcastable: {groups[idx]}"
        )
    return result


def broadcast_index(out_dims, in_dims) -> np.ndarray:
    """Flat source offset for every position of a broadcast output.

    The inverse gather behind `af.broadcast`: for each of the `prod(out_dims)`
    output positions, the flat offset into the operand, with an operand axis of
    extent 1 contributing a zero stride so that it repeats.
    """
    out_dims = tuple(int(x) for x in out_dims)
    in_dims = tuple(int(x) for x in in_dims)
    if len(out_dims) != 4 or len(in_dims) != 4:
        raise ValueError("out_dims and in_dims must both be 4-tuples")
    if any(v < 1 for v in out_dims) or any(v < 1 for v in in_dims):
        raise ValueError("every extent must be at least 1")
    od = _i64(out_dims)
    idm = _i64(in_dims)
    total = 1
    for v in out_dims:
        total *= v
    out = np.empty(max(total, 1), dtype=np.int64)
    lib.af_broadcast_index(total, _addr(od), _addr(idm), _addr(out))
    return out[:total]


def strides4(dims) -> np.ndarray:
    """Column-major strides of a 4-dimensional extent, `dim4` style."""
    dims = tuple(int(x) for x in dims)
    if len(dims) != 4:
        raise ValueError("dims must be a 4-tuple of extents")
    d = _i64(dims)
    out = np.empty(4, dtype=np.int64)
    lib.af_strides4(_addr(d), _addr(out))
    return out
