"""Parity tests for the index and shape arithmetic of mojo_arrayfire.

Every test in this file compares against the real `arrayfire` source, executed
verbatim by `tests/_upstream.py`. The real package cannot be imported here
because it loads the ArrayFire C library at import time and no backend is
installed; the harness sidesteps only that, not the code under test.
"""

import numpy as np
import pytest

import mojo_arrayfire as maf
import _upstream
from _upstream import seq_triple

_dtypes = _upstream._implicit_dtype()
_get_assign_dims = _upstream._get_assign_dims()
_get_info = _upstream._get_info()
_slice_to_length = _upstream._slice_to_length()

SLICES = [
    slice(None),
    slice(None, None, -1),
    slice(2, 5),
    slice(None, 4),
    slice(3, None),
    slice(1, 9, 3),
    slice(9, 1, -2),
    slice(0, 10, 2),
    slice(-4, None),
    slice(None, -3),
    slice(-6, -2),
    slice(4, 4),
    slice(4, 2),
    slice(4, 2, -1),
    slice(-3, -1, -2),
    slice(0, 0),
    slice(1, 1, 1),
    slice(10, 0, -5),
    slice(2, 11, 4),
    slice(7, 7, -1),
]


def test_seq_batch_matches_upstream_seq():
    got = maf.seq_batch(SLICES)
    for i, s in enumerate(SLICES):
        expect = seq_triple(s)
        assert tuple(got[i]) == expect, f"slice {s!r}: {tuple(got[i])} != {expect}"


def test_seq_batch_empty_selection_rewrites():
    """Upstream rewrites an empty forward selection to (1, 1, 1) and then still
    applies the trailing `end - copysign(1, step)`, so the triple is (1, 0, 1)."""
    assert tuple(maf.seq_batch([slice(4, 2)])[0]) == (1, 0, 1)
    assert seq_triple(slice(4, 2)) == (1.0, 0.0, 1.0)
    assert tuple(maf.seq_batch([slice(0, 0)])[0]) == (1, 0, 1)


def test_seq_batch_negative_empty_rewrite():
    """A fully negative empty selection becomes (-2, -2, -1) before the adjust."""
    assert tuple(maf.seq_batch([slice(-3, -1, -2)])[0]) == (-2, -1, -1)
    assert seq_triple(slice(-3, -1, -2)) == (-2.0, -1.0, -1.0)


def test_seq_batch_pinned_values():
    """Values pinned by hand as well as by the oracle, to catch oracle drift."""
    assert tuple(maf.seq_batch([slice(None)])[0]) == (0, -1, 1)
    assert tuple(maf.seq_batch([slice(None, None, -1)])[0]) == (-1, 0, -1)
    assert tuple(maf.seq_batch([slice(2, 5)])[0]) == (2, 4, 1)
    assert tuple(maf.seq_batch([slice(1, 9, 3)])[0]) == (1, 8, 3)
    assert tuple(maf.seq_batch([slice(9, 1, -2)])[0]) == (9, 2, -2)


def test_seq_from_int_is_the_number_branch():
    assert tuple(maf.seq_from_int(7)[0]) == (7, 7, 1)
    assert maf.seq_from_int(0)[0, 0] == 0


def test_seq_batch_rejects_non_slices():
    with pytest.raises(TypeError):
        maf.seq_batch([3])
    with pytest.raises(TypeError):
        maf.seq_batch([(1, 2)])


def test_empty_slice_batch():
    assert maf.seq_batch([]).shape == (0, 3)


@pytest.mark.parametrize("s", SLICES)
def test_slice_length_matches_upstream(s):
    for dim in (0, 1, 5, 10, 17):
        assert maf.slice_length(s, dim) == _slice_to_length(s, dim), (
            f"{s!r} dim={dim}"
        )


def test_slice_lengths_batch_matches_elementwise():
    got = maf.slice_lengths(SLICES, [10] * len(SLICES))
    for i, s in enumerate(SLICES):
        assert got[i] == _slice_to_length(s, 10), f"slice {s!r}"


def test_slice_lengths_rejects_mismatched_dims():
    with pytest.raises(ValueError):
        maf.slice_lengths([slice(None)], [10, 10])


def test_slice_length_negative_step_truncates_toward_zero():
    """The formula is `int((stop-start-1)/step)+1`, not a ceiling."""
    assert maf.slice_length(slice(0, 6, 2), 10) == 3
    assert maf.slice_length(slice(10, 0, -3), 10) == 4
    assert _slice_to_length(slice(10, 0, -3), 10) == 4
    assert maf.slice_length(slice(0, 6, 2), 10) == _slice_to_length(
        slice(0, 6, 2), 10
    )


def test_assign_dims_matches_upstream():
    idims = (10, 20, 30, 40)
    keys = [
        (slice(2, 5),),
        (3, slice(None)),
        (slice(None), slice(1, 9, 3)),
        (slice(None), slice(None), 2, slice(4, 4)),
        (),
        (slice(9, 1, -2), slice(None), slice(None, None, -1), slice(0, 30, 7)),
        (-1, -2, -3, -4),
    ]
    for key in keys:
        got = maf.assign_shape(key, idims)
        expect = _get_assign_dims(key, list(idims))
        assert list(got) == list(expect), f"key={key!r}"


def test_assign_dims_pads_missing_axes_with_a_full_slice():
    idims = (10, 20, 30, 40)
    assert list(maf.assign_shape((slice(0, 5),), idims)) == [5, 20, 30, 40]
    assert list(maf.assign_shape((0,), idims)) == [1, 20, 30, 40]
    assert list(maf.assign_shape((), idims)) == [10, 20, 30, 40]


def test_assign_dims_rejects_four_plus_axes_and_array_handles():
    with pytest.raises(ValueError):
        maf.assign_shape((1, 2, 3, 4, 5), (1, 2, 3, 4))
    with pytest.raises(TypeError):
        maf.assign_shape(("a",), (1, 2, 3, 4))


def test_get_info_matches_upstream():
    for dims in [(5,), (3, 4), (2, 3, 4), (1, 1, 1, 1)]:
        assert maf.get_info(dims) == _get_info(dims, 0)
    assert maf.get_info(None, 77) == _get_info(None, 77)
    with pytest.raises(RuntimeError):
        maf.get_info(None, 0)


def test_implicit_dtype_matches_upstream():
    number_dtype, implicit_dtype = _dtypes["number_dtype"], _dtypes["implicit_dtype"]
    values = [True, False, 0, 5, -7, 0.0, 1.5, -2.25, 1 + 2j]
    names = list(maf.DTYPE_VALUES)
    for ad in names:
        af_dtype = getattr(_upstream.Dtype, ad)
        for v in values:
            expect = int(implicit_dtype(v, af_dtype))
            assert maf.implicit_dtype(v, maf.DTYPE_VALUES[ad]) == expect, (v, ad)
            assert int(number_dtype(v)) == int(number_dtype(v))
    assert maf.implicit_dtype(1.0, maf.DTYPE_VALUES["f32"]) == 0
    assert maf.implicit_dtype(1.0, maf.DTYPE_VALUES["f64"]) == 2
    assert maf.implicit_dtype(1, maf.DTYPE_VALUES["f32"]) == 8
    assert maf.implicit_dtype(True, maf.DTYPE_VALUES["f64"]) == 4
    with pytest.raises(TypeError):
        maf.implicit_dtype("x", 0)


def test_dtype_values_match_the_installed_enum():
    for name, value in maf.DTYPE_VALUES.items():
        assert getattr(_upstream.Dtype, name).value == value, name


def test_number_dtype_table_matches_upstream():
    number_dtype = _dtypes["number_dtype"]
    pairs = [(True, "b8"), (3, "s64"), (1.5, "f64"), (1j, "c64")]
    for value, name in pairs:
        assert int(number_dtype(value)) == maf.DTYPE_VALUES[name]
        assert maf.NUMBER_DTYPE[
            {bool: "bool", int: "int", float: "float",
             complex: "complex"}[type(value)]
        ] == name
