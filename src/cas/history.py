from dataclasses import dataclass
from typing import Dict, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from .expression import *

__all__ = [
    'History',
    'HistoryItem'
]


@dataclass
class HistoryItem:
    command: 'BaseExpression'
    result: 'BaseExpression'


class History:
    def __init__(self, items: Mapping[int, HistoryItem]):
        self.items: Dict[int, HistoryItem] = dict(items)

    def __len__(self) -> int:
        """proxy method for self.items"""
        return len(self.items)

    def __contains__(self, item):
        """proxy method for self.items"""
        return item in self.items

    def append(self, item: HistoryItem):
        """proxy method for self.items"""
        self.items[self.next_number] = item

    def resolve(self, ref: 'HistoryRef') -> 'BaseExpression':
        # precondition
        assert ref.item in self

        return self.items[ref.item].result

    @property
    def next_number(self) -> int:
        return max(self.items.keys(), default=0) + 1
