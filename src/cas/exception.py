from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parser import Cursor


class InvalidExpressionError(Exception):
    pass


class ParseError(Exception):
    def __init__(self, source: str, start: 'Cursor', end: 'Cursor', description: str):
        super().__init__(source, start, end, description)

    def __str__(self) -> str:
        source = self.args[0]
        start = self.args[1]
        end = self.args[2]
        description = self.args[3]

        return f'Parse error at {start}: {description}\n' \
               f'{source}\n' \
               f'{" " * start}{"~" * max(1, end - start + 1)}'
