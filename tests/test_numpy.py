import numpy
import pytest
from inline_snapshot import snapshot
from inline_snapshot._global_state import state
from inline_snapshot.testing import Example

from inline_snapshot_numpy import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_almost_equal_nulp,
    assert_array_equal,
    assert_array_max_ulp,
)
from inline_snapshot_numpy import snapshot as numpy_snapshot


def test_assert_array_equal():
    actual = numpy.array([[1, 2], [3, 4]])

    assert_array_equal(actual, numpy_snapshot(numpy.array([[1, 2], [3, 4]])))
    assert_array_equal(actual, actual)

    value = numpy_snapshot(numpy.array([[1, 2], [3, 4]]))
    assert_array_equal(actual, value)
    assert_array_equal(actual, value)


def test_assertion_helpers():
    assert_allclose(numpy.array([1.0, 2.0]), snapshot(numpy.array([1.0, 2.0 + 1e-8])))
    assert_array_almost_equal(numpy.array([1.0]), snapshot(numpy.array([1.0 + 1e-8])))
    assert_array_almost_equal_nulp(numpy.array([1.0]), snapshot(numpy.array([1.0])))
    assert_array_max_ulp(numpy.array([1.0]), snapshot(numpy.array([1.0])))


def test_failure_preserves_numpy_message():
    active = state().active
    state().active = False
    try:
        with pytest.raises(AssertionError, match="Arrays are not equal"):
            assert_array_equal(numpy.array([1, 2]), numpy.array([1, 3]))
    finally:
        state().active = active


def test_failure_preserves_non_assertion_error():
    active = state().active
    state().active = False
    try:
        with pytest.raises(TypeError):
            assert_array_equal(
                numpy.array([(1,)], dtype=[("value", "i1")]), "not an array"
            )
    finally:
        state().active = active


def test_create_and_fix_snapshot():
    example = Example(
        """\
from inline_snapshot import snapshot
from inline_snapshot_numpy import assert_array_equal
from numpy import array


def test_array():
    assert_array_equal(array([1, 2]), snapshot())
"""
    )
    example.run_inline(
        ["--inline-snapshot=create"],
        changed_files=snapshot(
            {
                "tests/test_something.py": """\
from inline_snapshot import snapshot
from inline_snapshot_numpy import assert_array_equal
from numpy import array


def test_array():
    assert_array_equal(array([1, 2]), snapshot(array([1, 2])))
"""
            }
        ),
    )


def test_create_implicit_snapshot_argument():
    Example(
        """\
from inline_snapshot_numpy import assert_array_equal
from numpy import array


def test_array():
    assert_array_equal(array([1, 2]))
"""
    ).run_inline(
        ["--inline-snapshot=create"],
        changed_files=snapshot(
            {
                "tests/test_something.py": """\
from inline_snapshot_numpy import assert_array_equal
from numpy import array


def test_array():
    assert_array_equal(array([1, 2]), desired=array([1, 2]))
"""
            }
        ),
    ).run_inline()


def test_dtype_is_preserved():
    example = Example(
        """\
from inline_snapshot import snapshot
from inline_snapshot_numpy import assert_array_equal
from numpy import array


def test_array():
    assert_array_equal(array([1, 2], dtype="float32"), snapshot())
"""
    )
    example.run_inline(
        ["--inline-snapshot=create"],
        changed_files=snapshot(
            {
                "tests/test_something.py": """\
from inline_snapshot import snapshot
from inline_snapshot_numpy import assert_array_equal
from numpy import array


def test_array():
    assert_array_equal(
        array([1, 2], dtype="float32"), snapshot(array([1.0, 2.0], dtype="<f4"))
    )
"""
            }
        ),
    )


def test_unusual_dtypes():
    structured = numpy.array([(1, 2.0)], dtype=[("x", "i1"), ("y", "f4")])
    assert_array_equal(structured, snapshot(structured.copy()))

    ragged = numpy.empty(2, dtype=object)
    ragged[:] = [[1], [1, 2]]
    assert_array_equal(ragged, snapshot(ragged.copy()))


def test_setup():
    from inline_snapshot_numpy import setup

    setup()
    numpy.testing.assert_array_equal(numpy.array([1]), snapshot(numpy.array([1])))
