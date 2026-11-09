"""Parity tests for the index and shape arithmetic of mojo_arrayfire.

Every test in this file compares against the real `arrayfire` source, executed
verbatim by `tests/_upstream.py`. The real package cannot be imported here
because it loads the ArrayFire C library at import time and no backend is
installed; the harness sidesteps only that, not the code under test.
"""

import math

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


def test_seq_batch_tolerance_float_triples():
    """Seq fields are c_double upstream; compare with an explicit tolerance."""
    for s in SLICES:
        got = maf.seq_batch([s])[0].astype(np.float64)
        expect = np.array(seq_triple(s), dtype=np.float64)
        np.testing.assert_allclose(got, expect, rtol=1e-9, atol=1e-9)


def test_seq_batch_one_element():
    assert tuple(maf.seq_batch([slice(3, 4)])[0]) == (3, 3, 1)
    assert seq_triple(slice(3, 4)) == (3.0, 3.0, 1.0)
    assert maf.seq_from_int(0)[0].tolist() == [0, 0, 1]
    assert maf.seq_from_int(-7)[0].tolist() == [-7, -7, 1]


def test_seq_batch_duplicates():
    got = maf.seq_batch([slice(2, 5)] * 3)
    for row in got:
        assert tuple(row) == (2, 4, 1)
        assert tuple(row) == tuple(np.array(seq_triple(slice(2, 5))))


def test_seq_from_int_matches_upstream_seq():
    for v in (0, 1, 7, -7, True):
        got = maf.seq_from_int(v)[0].astype(np.float64)
        expect = np.array(seq_triple(v), dtype=np.float64)
        np.testing.assert_allclose(got, expect, rtol=1e-9, atol=1e-9)


def test_seq_from_int_matches_upstream_numeric_index():
    for v in (np.int64(5),):
        got = maf.seq_from_int(v)[0].astype(np.float64)
        expect = np.array(seq_triple(v), dtype=np.float64)
        np.testing.assert_allclose(got, expect, rtol=1e-9, atol=1e-9)
    # A fractional index is stored verbatim as c_double by upstream's Seq,
    # but the C dim_t index is the truncation; the port returns that C value.
    assert seq_triple(2.5) == (2.5, 2.5, 1.0)
    assert maf.seq_from_int(2.5)[0].tolist() == [2, 2, 1]
    assert math.isclose(seq_triple(2.5)[0], 2.5, rel_tol=1e-9, abs_tol=1e-9)


def test_seq_from_int_complex_and_nonfinite():
    with pytest.raises(TypeError):
        maf.seq_from_int(1 + 2j)
    with pytest.raises(TypeError):
        seq_triple(1 + 2j)
    with pytest.raises(ValueError):
        maf.seq_from_int(float("nan"))
    with pytest.raises(OverflowError):
        maf.seq_from_int(float("inf"))


def test_slice_lengths_duplicates_and_empty():
    got = maf.slice_lengths([slice(2, 5)] * 3, [10, 10, 10])
    for v in got:
        assert int(v) == _slice_to_length(slice(2, 5), 10) == 3
    assert maf.slice_lengths([], []).shape == (0,)
    assert maf.seq_batch([]).shape == (0, 3)


def test_slice_length_degenerate_dims():
    assert maf.slice_length(slice(None), 0) == _slice_to_length(slice(None), 0) == 0
    assert maf.slice_length(slice(None), -3) == _slice_to_length(slice(None), -3) == -3
    assert maf.slice_length(slice(0, 100), 10) == _slice_to_length(slice(0, 100), 10) == 100
    assert maf.slice_length(slice(-100, -90), 10) == _slice_to_length(slice(-100, -90), 10) == -10
    assert maf.slice_length(slice(-100, -90), 10) == -10


def test_slice_length_zero_step_raises():
    with pytest.raises(ZeroDivisionError):
        _slice_to_length(slice(0, 5, 0), 10)
    with pytest.raises(ZeroDivisionError):
        maf.slice_length(slice(0, 5, 0), 10)
    with pytest.raises(ZeroDivisionError):
        maf.slice_lengths([slice(0, 5, 0)], [10])
    with pytest.raises(ZeroDivisionError):
        maf.assign_shape((slice(0, 5, 0),), (10, 20, 30, 40))
    with pytest.raises(ZeroDivisionError):
        _get_assign_dims((slice(0, 5, 0),), [10, 20, 30, 40])


def test_assign_dims_single_non_tuple_keys():
    assert list(maf.assign_shape(3, (10, 20, 30, 40))) == _get_assign_dims(3, [10, 20, 30, 40]) == [1, 20, 30, 40]
    assert list(maf.assign_shape(slice(2, 5), (10, 20, 30, 40))) == _get_assign_dims(slice(2, 5), [10, 20, 30, 40]) == [3, 20, 30, 40]
    assert list(maf.assign_dims(3, (10, 20, 30, 40))) == list(maf.assign_shape(3, (10, 20, 30, 40)))
    assert list(maf.assign_dims(slice(2, 5), (10, 20, 30, 40))) == list(maf.assign_shape(slice(2, 5), (10, 20, 30, 40)))
    assert list(maf.assign_shape(True, (10, 20, 30, 40))) == _get_assign_dims(True, [10, 20, 30, 40]) == [1, 20, 30, 40]


def test_assign_dims_degenerate_shapes():
    assert list(maf.assign_shape((), ())) == [1, 1, 1, 1]
    assert _get_assign_dims((), []) == [1, 1, 1, 1]
    assert list(maf.assign_shape((slice(3, 4),), (10, 20, 30, 40))) == _get_assign_dims((slice(3, 4),), [10, 20, 30, 40]) == [1, 20, 30, 40]


def test_get_info_degenerate():
    assert maf.get_info([0, 3], 0) == _get_info([0, 3], 0) == (2, [0, 3, 1, 1])
    assert maf.get_info(None, -5) == _get_info(None, -5) == (1, [-5, 1, 1, 1])
    assert maf.get_info([10**18], 0) == _get_info([10**18], 0)
    assert maf.get_info([2.5], 0)[1][0] == 2
    assert _get_info([2.5], 0)[1][0] == 2.5
    with pytest.raises(RuntimeError):
        _get_info([], 0)
    with pytest.raises(RuntimeError):
        maf.get_info([], 0)
    with pytest.raises(IndexError):
        _get_info([1, 2, 3, 4, 5], 0)
    with pytest.raises(IndexError):
        maf.get_info([1, 2, 3, 4, 5], 0)


def test_implicit_dtype_numpy_scalars():
    number_dtype, implicit_dtype = _dtypes["number_dtype"], _dtypes["implicit_dtype"]
    # Upstream number_dtype dispatches on a.dtype.char for non-builtin
    # scalars; the harness to_dtype table maps 'l' to s64 and 'i' to s32.
    assert int(number_dtype(np.int64(3))) == maf.DTYPE_VALUES["s64"]
    assert int(number_dtype(np.int32(3))) == maf.DTYPE_VALUES["s32"]
    assert int(implicit_dtype(np.int64(3), _upstream.Dtype.f32)) == maf.DTYPE_VALUES["s64"]
    assert int(implicit_dtype(np.int32(3), _upstream.Dtype.f32)) == maf.DTYPE_VALUES["s32"]
    assert maf.implicit_dtype(np.int64(3), maf.DTYPE_VALUES["f32"]) == maf.DTYPE_VALUES["s64"]
    assert int(number_dtype(np.float64(1.5))) == maf.DTYPE_VALUES["f64"]
    assert maf.implicit_dtype(np.float64(1.5), maf.DTYPE_VALUES["f32"]) == int(
        implicit_dtype(np.float64(1.5), _upstream.Dtype.f32)
    ) == maf.DTYPE_VALUES["f32"]
    assert maf.implicit_dtype(float("nan"), maf.DTYPE_VALUES["f32"]) == 0
    assert int(implicit_dtype(float("nan"), _upstream.Dtype.f32)) == 0
    assert maf.implicit_dtype(float("inf"), maf.DTYPE_VALUES["f32"]) == 0
    assert math.isnan(float("nan"))
    assert math.isinf(float("inf"))
