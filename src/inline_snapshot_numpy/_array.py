import numpy
from inline_snapshot._snapshot.generic_value import GenericValue
from inline_snapshot._types import SnapshotRefBase
from inline_snapshot.plugin import ImportFrom, customize


class Array:
    # Make NumPy defer ndarray comparisons to this wrapper.
    __array_priority__ = float("inf")

    def __init__(self, value, compare):
        self.value = value
        self.compare = compare

    def __eq__(self, other):
        if isinstance(other, (GenericValue, SnapshotRefBase)):
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
