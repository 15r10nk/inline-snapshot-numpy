from functools import wraps
from typing import Optional

import numpy
import numpy.testing
from inline_snapshot import snapshot_arg
from inline_snapshot._snapshot.generic_value import GenericValue
from inline_snapshot._types import SnapshotRefBase

from ._array import Array


def _make_assert_equal(assert_equal):
    @wraps(assert_equal)
    def result(actual, desired=..., *args, **kwargs):
        error: Optional[Exception] = None

        def compare(left, right):
            nonlocal error
            try:
                assert_equal(left, right, *args, **kwargs)
            except Exception as exception:
                error = exception
                return False
            return True

        if not isinstance(desired, (GenericValue, SnapshotRefBase)):
            desired = Array(desired, compare)
            desired = snapshot_arg(desired)

        if not (Array(actual, compare) == desired):
            if error is not None:
                raise error

    return result


assert_allclose = _make_assert_equal(numpy.testing.assert_allclose)
assert_array_almost_equal = _make_assert_equal(numpy.testing.assert_array_almost_equal)
assert_array_almost_equal_nulp = _make_assert_equal(
    numpy.testing.assert_array_almost_equal_nulp
)
assert_array_equal = _make_assert_equal(numpy.testing.assert_array_equal)
assert_array_max_ulp = _make_assert_equal(numpy.testing.assert_array_max_ulp)


def setup():
    """Make NumPy's testing helpers accept inline snapshots."""

    numpy.testing.assert_allclose = assert_allclose
    numpy.testing.assert_array_almost_equal = assert_array_almost_equal
    numpy.testing.assert_array_almost_equal_nulp = assert_array_almost_equal_nulp
    numpy.testing.assert_array_equal = assert_array_equal
    numpy.testing.assert_array_max_ulp = assert_array_max_ulp
