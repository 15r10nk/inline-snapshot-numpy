# inline-snapshot-numpy

[NumPy](https://numpy.org/) integration for
[inline-snapshot](https://github.com/15r10nk/inline-snapshot).

## Installation

```bash
pip install inline-snapshot-numpy
```

## Usage

The package provides snapshot-aware versions of NumPy's array assertions:

```python
from inline_snapshot_numpy import assert_array_equal
from numpy import array


def test_array():
    actual = array([[1, 2], [3, 4]])
    assert_array_equal(actual)
```

Create the missing snapshot with:

```bash
pytest --inline-snapshot=create
```

The test will become:

```python
def test_array():
    actual = array([[1, 2], [3, 4]])
    assert_array_equal(actual, desired=array([[1, 2], [3, 4]]))
```

The following helpers are available: `assert_array_equal`, `assert_allclose`,
`assert_array_almost_equal`, `assert_array_almost_equal_nulp`,
and `assert_array_max_ulp`.

The expected array is managed through `snapshot_arg()`, so no explicit
`snapshot()` call or import is needed. You can still pass an explicit
`snapshot(...)` for compatibility.

Alternatively, call `setup()` from `conftest.py` to patch the corresponding
functions in `numpy.testing`:

```python
from inline_snapshot_numpy import setup

setup()
```

## License

Distributed under the terms of the MIT license.
