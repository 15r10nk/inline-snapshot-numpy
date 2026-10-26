from difflib import unified_diff
from pathlib import Path
from typing import Union

import numpy
from inline_snapshot import Format, register_format
from rich.syntax import Syntax

from ._array import Array


def _array_text(value: numpy.ndarray) -> str:
    values = numpy.array2string(value, separator=", ")
    return f"shape: {value.shape}\ndtype: {value.dtype.str}\nvalues:\n{values}"


@register_format
class NumpyFormat(Format[numpy.ndarray]):
    """Store NumPy arrays as non-pickled ``.npy`` files."""

    suffix = ".npy"

    def is_format_for(self, value: object) -> bool:
        return isinstance(value, (Array, numpy.ndarray))

    def encode(self, value: Union[numpy.ndarray, Array], path: Path) -> None:
        if isinstance(value, Array):
            value = numpy.asarray(value.value)
        numpy.save(path, value, allow_pickle=False)

    def decode(self, path: Path) -> numpy.ndarray:
        return numpy.load(path, allow_pickle=False)

    def rich_show(self, path: Path) -> str:
        return _array_text(self.decode(path))

    def rich_diff(self, original: Path, new: Path) -> Syntax:
        text = "\n".join(
            unified_diff(
                _array_text(self.decode(original)).splitlines(),
                _array_text(self.decode(new)).splitlines(),
                lineterm="",
            )
        )
        return Syntax(text, "diff", theme="ansi_light", word_wrap=True)
