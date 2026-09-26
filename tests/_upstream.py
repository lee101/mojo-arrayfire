"""Execute the real `arrayfire` Python code as the parity oracle.

`import arrayfire` fails in this environment: the package loads the ArrayFire C
library at import time and no backend library is installed. The Python-side
arithmetic this port covers does not need the C library at all, so the harness
below extracts the relevant class and function source from the installed
distribution and executes it verbatim, supplying only the handful of names those
definitions close over (`ctypes`, `numbers`, `math`, and the `Dtype` enum read
out of `library.py`).

That makes the oracle the upstream implementation itself, not a paraphrase of
it. If the installed source is missing, every test that needs it skips loudly
rather than falling back to a local copy.
"""

import ctypes
import functools
import math
import numbers
import pathlib
import re

import pytest

_SITE = pathlib.Path(
    "/nvme0n1-disk/mojo-toolchain/testvenv/lib/python3.13/site-packages"
)
_ARRAYFIRE = _SITE / "arrayfire"


def _require(path: pathlib.Path):
    if not path.exists():
        pytest.skip(
            f"arrayfire source not found at {path}; parity cannot be checked"
        )
    return path.read_text()


def _enum_type():
    """`_Enum_Type`: an int whose `.value` is the int, as library.py declares."""
    return type(
        "_Enum_Type",
        (int,),
        {"value": property(lambda self: int(self))},
    )


@functools.lru_cache(maxsize=None)
def _dtype():
    """The real `Dtype` enum, rebuilt from its source in library.py."""
    src = _require(_ARRAYFIRE / "library.py")
    match = re.search(r"^class Dtype\(_Enum\):.*?(?=^class )", src, re.S | re.M)
    assert match, "Dtype enum not found in arrayfire/library.py"
    _Enum = type("_Enum", (), {"__slots__": ()})
    _Enum_Type = _enum_type()
    ns = {"_Enum": _Enum, "_Enum_Type": _Enum_Type}
    exec(match.group(0), ns)
    return ns["Dtype"]


Dtype = _dtype()


@functools.lru_cache(maxsize=None)
def _slice_to_length():
    src = _require(_ARRAYFIRE / "array.py")
    match = re.search(
        r"^def _slice_to_length\(key, dim\):.*?(?=^def )", src, re.S | re.M
    )
    assert match, "_slice_to_length not found in arrayfire/array.py"
    ns = {}
    exec(match.group(0), ns)
    return ns["_slice_to_length"]


@functools.lru_cache(maxsize=None)
def _ctype_to_lists():
    src = _require(_ARRAYFIRE / "array.py")
    match = re.search(
        r"^def _ctype_to_lists\(.*?(?=^def )", src, re.S | re.M
    )
    assert match, "_ctype_to_lists not found in arrayfire/array.py"
    ns = {}
    exec(match.group(0), ns)
    return ns["_ctype_to_lists"]


@functools.lru_cache(maxsize=None)
def _get_info():
    src = _require(_ARRAYFIRE / "array.py")
    match = re.search(r"^def _get_info\(.*?(?=^def )", src, re.S | re.M)
    assert match, "_get_info not found in arrayfire/array.py"
    ns = {}
    exec(match.group(0), ns)
    return ns["_get_info"]


@functools.lru_cache(maxsize=None)
def _get_assign_dims():
    """`_get_assign_dims`, with stand-ins for the two names it closes over."""
    src = _require(_ARRAYFIRE / "array.py")
    match = re.search(
        r"^def _get_assign_dims\(.*?(?=^def )", src, re.S | re.M
    )
    assert match, "_get_assign_dims not found in arrayfire/array.py"

    class BaseArray:  # the type it isinstance-checks af.Array keys against
        pass

    class ParallelRange:
        pass

    ns = {
        "_is_number": lambda a: isinstance(a, numbers.Number),
        "BaseArray": BaseArray,
        "ParallelRange": ParallelRange,
        "_slice_to_length": _slice_to_length(),
    }
    exec(match.group(0), ns)
    return ns["_get_assign_dims"]


@functools.lru_cache(maxsize=None)
def _seq():
    """`arrayfire.index.Seq`, whose constructor is the slice mapping."""
    src = _require(_ARRAYFIRE / "index.py")
    match = re.search(
        r"^class Seq\(ct\.Structure\):.*?(?=^class ParallelRange)", src, re.S | re.M
    )
    assert match, "Seq not found in arrayfire/index.py"
    ns = {
        "ct": ctypes,
        "math": math,
        "numbers": numbers,
        "c_double_t": ctypes.c_double,
        "_is_number": lambda a: isinstance(a, numbers.Number),
    }
    exec(match.group(0), ns)
    return ns["Seq"]


@functools.lru_cache(maxsize=None)
def _implicit_dtype():
    """`arrayfire.util.implicit_dtype` and `number_dtype`, verbatim."""
    src = _require(_ARRAYFIRE / "util.py")
    out = {}
    ns = {"Dtype": Dtype, "to_dtype": {"f": Dtype.f32, "d": Dtype.f64,
                                        "i": Dtype.s32, "l": Dtype.s64,
                                        "b": Dtype.b8, "F": Dtype.c32,
                                        "D": Dtype.c64}}
    for name in ("number_dtype", "implicit_dtype"):
        match = re.search(rf"^def {name}\(.*?(?=^def )", src, re.S | re.M)
        assert match, f"{name} not found in arrayfire/util.py"
        exec(match.group(0), ns)
        out[name] = ns[name]
    return out


@functools.lru_cache(maxsize=None)
def seq_triple(s):
    """The (begin, end, step) upstream's `Seq` produces for one index."""
    q = _seq()(s)
    return (float(q.begin), float(q.end), float(q.step))
