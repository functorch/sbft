import typing as t
from pathlib import Path

def load(
    f: t.Union[Path, list[str], t.IO[str]],
    _dict: type[t.Any] = ...,
) -> dict[str, t.Any]: ...
