from difflib import SequenceMatcher
from pathlib import Path
from typing import Union

import numpy
from inline_snapshot import Format, register_format
from rich.text import Text

from ._array import Array


def _array_text(value: numpy.ndarray) -> str:
    values = numpy.array2string(value, separator=", ")
    return f"shape: {value.shape}\ndtype: {value.dtype.str}\nvalues:\n{values}"


def _value_text(value: numpy.generic) -> str:
    return numpy.array2string(numpy.asarray(value), separator=", ", threshold=numpy.inf)


def _index_text(index: int, shape: tuple[int, ...]) -> str:
    return repr(tuple(int(part) for part in numpy.unravel_index(index, shape)))


def _array_diff(original: numpy.ndarray, new: numpy.ndarray) -> Text:
    lines = Text()

    def add_line(*parts: tuple[str, str]) -> None:
        if lines:
            lines.append("\n")
        for text, style in parts:
            lines.append(text, style)

    if original.shape != new.shape:
        add_line((f"- shape: {original.shape}", "red"))
        add_line((f"+ shape: {new.shape}", "green"))
    if original.dtype != new.dtype:
        add_line((f"- dtype: {original.dtype.str}", "red"))
        add_line((f"+ dtype: {new.dtype.str}", "green"))

    original_values = [_value_text(value) for value in original.flat]
    new_values = [_value_text(value) for value in new.flat]
    for operation, old_start, old_end, new_start, new_end in SequenceMatcher(
        None, original_values, new_values, autojunk=False
    ).get_opcodes():
        if operation == "equal":
            continue
        if operation == "replace":
            changed = min(old_end - old_start, new_end - new_start)
            for offset in range(changed):
                old_index = old_start + offset
                new_index = new_start + offset
                old_location = _index_text(old_index, original.shape)
                new_location = _index_text(new_index, new.shape)
                location = (
                    old_location
                    if old_location == new_location
                    else f"{old_location} -> {new_location}"
                )
                add_line(
                    (f"~ {location}: ", "yellow"),
                    (original_values[old_index], "red"),
                    (" -> ", "yellow"),
                    (new_values[new_index], "green"),
                )
            old_start += changed
            new_start += changed
        for index in range(old_start, old_end):
            add_line(
                (
                    f"- {_index_text(index, original.shape)}: {original_values[index]}",
                    "red",
                )
            )
        for index in range(new_start, new_end):
            add_line(
                (f"+ {_index_text(index, new.shape)}: {new_values[index]}", "green")
            )

    return lines


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

    def rich_diff(self, original: Path, new: Path) -> Text:
        return _array_diff(self.decode(original), self.decode(new))
