import os
import platform
from unittest.mock import ANY

import numpy
import pytest
from inline_snapshot import Is
from inline_snapshot import snapshot as inline_snapshot
from inline_snapshot.testing import Example

from inline_snapshot_numpy import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_almost_equal_nulp,
    assert_array_equal,
    assert_array_max_ulp,
    snapshot,
)

requires_snapshot_updates = pytest.mark.skipif(
    platform.python_implementation() == "PyPy" and "PYTEST_XDIST_WORKER" in os.environ,
    reason="inline-snapshot disables snapshot updates with xdist on PyPy",
)


def test_snapshot_export():
    assert snapshot is inline_snapshot


def test_assert_array_equal():
    actual = numpy.array([[1, 2], [3, 4]])

    assert_array_equal(actual, snapshot(numpy.array([[1, 2], [3, 4]])))
    assert_array_equal(actual, actual)

    value = snapshot(numpy.array([[1, 2], [3, 4]]))
    assert_array_equal(actual, value)
    assert_array_equal(actual, value)


def test_assertion_helpers():
    assert_allclose(numpy.array([1.0, 2.0]), snapshot(numpy.array([1.0, 2.0 + 1e-8])))
    assert_array_almost_equal(numpy.array([1.0]), snapshot(numpy.array([1.0 + 1e-8])))
    assert_array_almost_equal_nulp(numpy.array([1.0]), snapshot(numpy.array([1.0])))
    assert_array_max_ulp(numpy.array([1.0]), snapshot(numpy.array([1.0])))


def test_failure_preserves_numpy_message():
    Example(
        """\
import numpy
import pytest
from inline_snapshot_numpy import assert_array_equal


def test_failure():
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        assert_array_equal(numpy.array([1, 2]), numpy.array([1, 3]))
"""
    ).run_inline(["--inline-snapshot=disable"], reported_categories=set())


def test_failure_preserves_non_assertion_error():
    Example(
        """\
import numpy
import pytest
from inline_snapshot_numpy import assert_array_max_ulp


def test_failure():
    with pytest.raises(TypeError, match="non-numeric array"):
        assert_array_max_ulp(numpy.array(["a"]), numpy.array(["b"]))
"""
    ).run_inline(["--inline-snapshot=disable"], reported_categories=set())


@requires_snapshot_updates
def test_numpy_external_format():
    Example(
        """\
from inline_snapshot import external
from inline_snapshot_numpy import assert_array_equal
from numpy import array

value = 3
insert_values = [1, 3]
delete_values = [0, 1, 2, 3]
dtype_values = [1]
dtype = "int32"


def test_array():
    assert_array_equal(array([[1, 2], [value, 4]], dtype="float32"), external())


def test_insert():
    assert_array_equal(array(insert_values), external())


def test_delete():
    assert_array_equal(array(delete_values), external())


def test_dtype():
    assert_array_equal(array(dtype_values, dtype=dtype), external())
"""
    ).run_inline(
        ["--inline-snapshot=create"], changed_files=Is(ANY)
    ).run_inline().replace(
        "value = 3", "value = 30"
    ).replace(
        "insert_values = [1, 3]", "insert_values = [0, 1, 2, 3]"
    ).replace(
        "delete_values = [0, 1, 2, 3]", "delete_values = [1, 3]"
    ).replace(
        "dtype_values = [1]", "dtype_values = [1, 2]"
    ).replace(
        'dtype = "int32"', 'dtype = "int64"'
    ).run_inline(
        ["--inline-snapshot=report"],
        report=snapshot(
            """\
FAIL: some snapshots in this test have incorrect values.
If you just created this value with --inline-snapshot=create, the value is now \n\
created and you can ignore this message.
FAIL: some snapshots in this test have incorrect values.
If you just created this value with --inline-snapshot=create, the value is now \n\
created and you can ignore this message.
FAIL: some snapshots in this test have incorrect values.
If you just created this value with --inline-snapshot=create, the value is now \n\
created and you can ignore this message.
FAIL: some snapshots in this test have incorrect values.
If you just created this value with --inline-snapshot=create, the value is now \n\
created and you can ignore this message.


═══════════════════════════════ inline-snapshot ════════════════════════════════
+--------------- uuid:e3e70682-c209-4cac-a29f-6fbed82c07cd.npy ----------------+
| ~ (1, 0): 3. -> 30.                                                          |
+------------------------------------------------------------------------------+
+--------------- uuid:f728b4fa-4248-4e3a-8a5d-2f346baa9455.npy ----------------+
| - shape: (2,)                                                                |
| + shape: (4,)                                                                |
| + (0,): 0                                                                    |
| + (2,): 2                                                                    |
+------------------------------------------------------------------------------+
+--------------- uuid:eb1167b3-67a9-4378-bc65-c1e582e2e662.npy ----------------+
| - shape: (4,)                                                                |
| + shape: (2,)                                                                |
| - (0,): 0                                                                    |
| - (2,): 2                                                                    |
+------------------------------------------------------------------------------+
+--------------- uuid:f7c1bd87-4da5-4709-9471-3d60c8a70639.npy ----------------+
| - shape: (1,)                                                                |
| + shape: (2,)                                                                |
| - dtype: <i4                                                                 |
| + dtype: <i8                                                                 |
| + (1,): 2                                                                    |
+------------------------------------------------------------------------------+
These changes are not applied.
Use --inline-snapshot=fix to apply them, or use the interactive mode with \n\
--inline-snapshot=review

"""
        ),
        reported_categories={"fix"},
    )


@requires_snapshot_updates
def test_external_ndarray_snapshot():
    example = Example(
        """\
from inline_snapshot import external
from numpy import array


def test_array():
    assert external() == array([1])
"""
    ).run_inline(
        ["--inline-snapshot=create"],
        changed_files={
            ".inline-snapshot/files_using_external.txt": """\
# this file is generated by inline-snapshot and requires no manual edits (https://15r10nk.github.io/inline-snapshot/latest/external/external/#cleaning-up-old-externals)
tests/test_something.py
""",
            "tests/__inline_snapshot__/test_something/test_array/e3e70682-c209-4cac-a29f-6fbed82c07cd.npy": Is(
                ANY
            ),
            "tests/test_something.py": """\
from inline_snapshot import external
from numpy import array


def test_array():
    assert external("uuid:e3e70682-c209-4cac-a29f-6fbed82c07cd.npy") == array([1])
""",
        },
    )
    assert ".npy" in example.read_text("tests/test_something.py")
    example.run_inline().run_inline(
        ["--inline-snapshot=disable"], reported_categories=set()
    )


@requires_snapshot_updates
def test_external_snapshot():
    example = Example(
        """\
from inline_snapshot import external
from inline_snapshot_numpy import assert_array_almost_equal
from numpy import array

value = 3.01


def test_array():
    assert_array_almost_equal(array([1, 2, value]), external())
"""
    ).run_inline(["--inline-snapshot=create"], changed_files=Is(ANY))
    assert ".npy" in example.read_text("tests/test_something.py")
    example.run_inline().replace(
        "value = 3.01", "value = 3.01000001"
    ).run_inline().run_inline(["--inline-snapshot=disable"], reported_categories=set())


@requires_snapshot_updates
def test_create_and_fix_snapshot():
    example = Example(
        """\
from inline_snapshot import snapshot
from inline_snapshot_numpy import assert_array_equal
from numpy import array

value = 2


def test_array():
    assert_array_equal(array([1, value]), snapshot())
"""
    )
    example = example.run_inline(
        ["--inline-snapshot=create"],
        changed_files=snapshot(
            {
                "tests/test_something.py": """\
from inline_snapshot import snapshot
from inline_snapshot_numpy import assert_array_equal
from numpy import array

value = 2


def test_array():
    assert_array_equal(array([1, value]), snapshot(array([1, 2])))
"""
            }
        ),
    )
    example.run_inline().replace("value = 2", "value = 3").run_inline(
        ["--inline-snapshot=fix"], changed_files=Is(ANY)
    ).run_inline().run_inline(["--inline-snapshot=disable"], reported_categories=set())


@requires_snapshot_updates
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


@requires_snapshot_updates
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
    ).run_inline().run_inline(["--inline-snapshot=disable"], reported_categories=set())


def test_unusual_dtypes():
    structured = numpy.array([(1, 2.0)], dtype=[("x", "i1"), ("y", "f4")])
    assert_array_equal(structured, snapshot(structured.copy()))

    ragged = numpy.empty(2, dtype=object)
    ragged[:] = [[1], [1, 2]]
    assert_array_equal(ragged, snapshot(ragged.copy()))


def test_setup():
    Example(
        {
            "conftest.py": """\
from inline_snapshot_numpy import setup


setup()
""",
            "tests/test_something.py": """\
import numpy
from inline_snapshot import snapshot


def test_array():
    numpy.testing.assert_array_equal(numpy.array([1]), snapshot(numpy.array([1])))
""",
        }
    ).run_pytest()
