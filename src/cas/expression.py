import enum
import operator
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from typing import Union, TypeVar, Optional, Callable

from .exception import *
from .history import *

__all__ = [
    'DisplayOptions',
    'BinaryOperation',
    'BaseExpression',
    'Literal',
    'Symbol',
    'BinaryExpr',
    'HistoryRef',
    'ExpandExpression',
]


@dataclass
class DisplayOptions:
    spaces: bool
    """whether to put spaces around the operator / between operands"""
    operator: bool
    """whether to display the operator itself"""


class BinaryOperation(enum.Enum):
    Add = '+'
    Sub = '-'
    Mul = '*'
    Div = '/'
    Pow = '^'

    @property
    def display_options(self) -> DisplayOptions:
        if self is BinaryOperation.Pow:
            return DisplayOptions(False, True)
        if self is BinaryOperation.Mul:
            return DisplayOptions(False, False)
        return DisplayOptions(True, True)

    @property
    def need_parens(self) -> bool:
        return self is BinaryOperation.Add or self is BinaryOperation.Sub

    @property
    def operator(self) -> Callable[['BaseExpression', 'BaseExpression'], 'BaseExpression']:
        if self is BinaryOperation.Add:
            return operator.add
        if self is BinaryOperation.Sub:
            return operator.sub
        # TODO

    def __call__(self, lhs: 'BaseExpression', rhs: 'BaseExpression') -> 'BaseExpression':
        return self.operator(lhs, rhs)


T = TypeVar('T')


class BaseExpression(metaclass=ABCMeta):
    def expand(self) -> 'BaseExpression':
        return self

    @abstractmethod
    def clone(self: T) -> 'T':
        raise NotImplemented


class Literal(BaseExpression):
    def __init__(self, literal: Union[Fraction, int, float]):
        self.literal = literal

    def __str__(self) -> str:
        return str(self.literal)

    def __add__(self, other) -> BaseExpression:
        if isinstance(other, Literal):
            return Literal(self.literal + other.literal)
        return BinaryExpr(self, BinaryOperation.Add, other)

    def __eq__(self, other: object):
        if not isinstance(other, Literal):
            return NotImplemented
        return self.literal == other.literal

    def __hash__(self):
        return hash(self.literal)

    def clone(self: 'Literal') -> 'Literal':
        return Literal(self.literal)


class Symbol(BaseExpression):
    def __init__(self, symbol: str):
        self.symbol = symbol

    def __str__(self) -> str:
        return self.symbol

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Symbol):
            return False
        return self.symbol == other.symbol

    def __hash__(self) -> int:
        return hash(self.symbol)

    def clone(self: 'Symbol') -> 'Symbol':
        return Symbol(self.symbol)


class BinaryExpr(BaseExpression):
    def __init__(self, lhs: BaseExpression, op: BinaryOperation, rhs: BaseExpression, parens: Optional[bool] = None):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs
        self.parens = op.need_parens if parens is None else parens

    def __str__(self) -> str:
        options = self.op.display_options

        space = " " if options.spaces else ""
        op = self.op.value if options.operator else ""
        l, r = "()" if self.parens else ("", "")

        if options.spaces and not options.operator:
            # only one space
            return f'{l}{self.lhs} {self.rhs}{r}'
        else:
            return f'{l}{self.lhs}{space}{op}{space}{self.rhs}{r}'

    def clone(self: 'BinaryExpr',
              lhs: Optional[BaseExpression] = None,
              rhs: Optional[BaseExpression] = None) -> 'BinaryExpr':
        lhs = lhs if lhs is not None else self.lhs.clone()
        rhs = rhs if rhs is not None else self.rhs.clone()
        return BinaryExpr(lhs, self.op, rhs, self.parens)


class HistoryRef(BaseExpression):
    def __init__(self, item: int):
        self.item = item

    def __str__(self) -> str:
        return f'[{self.item}]'

    def resolve(self, history: 'History') -> 'BaseExpression':
        if self.item not in history:
            raise InvalidExpressionError()
        return history.resolve(self)

    def clone(self: 'HistoryRef') -> 'HistoryRef':
        return HistoryRef(self.item)


class ExpandExpression(BaseExpression):
    def __init__(self, inner: BaseExpression):
        self.inner = inner

    def __str__(self) -> str:
        return f'expand {self.inner}'

    def clone(self: 'ExpandExpression', inner: Optional[BaseExpression] = None) -> 'ExpandExpression':
        inner = inner if inner is not None else self.inner.clone()
        return ExpandExpression(inner)
