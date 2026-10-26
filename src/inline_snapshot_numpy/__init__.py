from functools import wraps
from typing import Optional

import numpy
import numpy.testing
from inline_snapshot import snapshot, snapshot_arg
from inline_snapshot._snapshot.generic_value import GenericValue
from inline_snapshot.plugin import ImportFrom, customize

__version__ = "0.1.0"

__all__ = (
    "assert_allclose",
    "assert_array_almost_equal",
    "assert_array_almost_equal_nulp",
    "assert_array_equal",
    "assert_array_max_ulp",
    "setup",
    "snapshot",
)


class Array:
    # Make NumPy defer ndarray comparisons to this wrapper.
    __array_priority__ = float("inf")

    def __init__(self, value, compare):
        self.value = value
        self.compare = compare

    def __eq__(self, other):
        if isinstance(other, GenericValue):
            return NotImplemented
        if isinstance(other, Array):
            other = other.value
        return self.compare(self.value, other)


@customize
def customize_array(value, builder):
    if not isinstance(value, Array):
        return None

    array = numpy.asarray(value.value)
    values = array.tolist()

    try:
        inferred_dtype = numpy.asarray(values).dtype
    except (TypeError, ValueError):  # pragma: no cover - defensive for object arrays
        inferred_dtype = None

    kwargs = {}
    if inferred_dtype != array.dtype:
        kwargs["dtype"] = (
            array.dtype.descr if array.dtype.names is not None else array.dtype.str
        )

    array_function = builder.create_code(
        "array", imports=[ImportFrom("numpy", "array")]
    )
    return builder.create_call(array_function, [values], kwargs)


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

        if not isinstance(desired, GenericValue):
            desired = Array(desired, compare)
        desired = snapshot_arg(desired)

        if not (Array(actual, compare) == desired):
            assert error is not None
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
