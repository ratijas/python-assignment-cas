import atexit
import enum
import operator
import readline
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Union, TypeVar, Optional, Callable

from .exception import *
from .expression import *
from .history import *
from .parser import *


class Repl:
    def __init__(self):
        self.history: History = History({})
        # let user work comfortably with REPL in terminal
        # noinspection PyBroadException
        try:
            readline.read_history_file()
        except Exception:
            pass
        atexit.register(readline.write_history_file)

    def block(self):
        try:
            while True:
                history_number = self.history.next_number
                raw = input(f'[{history_number}]>>> ')
                if not raw.strip():
                    continue
                try:
                    expr = self.parse(raw)
                    print(f'[debug] parsed: {expr}')
                    result = self.eval(expr)
                    self.history.append(HistoryItem(expr, result))
                    print(f'{history_number}: {result}')

                except ParseError as e:
                    print(f'err: {e}')

                except InvalidExpressionError:
                    print(f'err: invalid expression')

        except (KeyboardInterrupt, EOFError):
            pass

    @staticmethod
    def parse(raw: str) -> 'BaseExpression':
        """try parse an expression.

        :param raw: raw user input string
        :return: parsed expression
        :raise InvalidExpressionError: if unable to parse
        """
        # import re
        # raw = raw.strip()
        try:
            # match = re.match(r"^\[(\d+)\]$", raw)
            # if match is not None:
            #     return HistoryRef(int(match[1]))
            #
            # match = re.match(r"^\d+$", raw)
            # if match is not None:
            #     return Literal(int(raw))
            #
            # match = re.match(r"^\d+\.\d+$", raw)
            # if match is not None:
            #     return Literal(float(raw))

            if raw == "yahaha":
                return BinaryExpr(Literal(42), BinaryOperation.Add, Literal(37), True)

            return build_ast(raw)

        except ValueError:
            raise InvalidExpressionError()

    def eval(self, expression: BaseExpression) -> BaseExpression:
        """try evaluate an expression.

        :param expression: input expression
        :return: result expression
        :raise InvalidExpressionError: if unable to compute
        """
        passes: List[Pass] = [
            ConstantsFolding(),
            HistoryExpansion(self.history)
        ]

        # loop through passes, applying them all one by one.
        # then, if any of them resulted in expression change,
        # re-run all passes; and so on until no pass mutates
        # the expression.

        # if isinstance(expression, ExpandExpression):
        #     expression = expression.inner

        affected = True
        while affected:
            affected = False
            for p in passes:
                result = p.run(expression)
                if result.affected:
                    affected = True
                    expression = result.new

        return expression


@dataclass
class PassResult:
    old: BaseExpression
    new: Optional[BaseExpression]

    @property
    def affected(self):
        return (self.new is not None) and (self.new is not self.old)


class Pass(metaclass=ABCMeta):
    @abstractmethod
    def run(self, expression: BaseExpression) -> PassResult:
        pass

    # @final
    def walk(self, expression: BaseExpression) -> PassResult:
        """Recursive passes may redefine `step` method and call `walk` on their expressions.

        Walk occurs in the depth-first order, i.e. root nodes are visited last.

        For a particular expression in the expression tree,
        if step function returns some new expression,
        then it will replace the current node.
        """
        # preserve original expression, introduce mutable variable.
        e = expression

        # visit children nodes
        if isinstance(e, (Literal, HistoryRef, Symbol)):
            pass

        elif isinstance(e, ExpandExpression):
            inner = self.walk(e.inner)
            if inner.affected:
                e = e.clone(inner=inner.new)

        elif isinstance(e, BinaryExpr):
            lhs = self.walk(e.lhs)
            rhs = self.walk(e.rhs)

            if any((lhs.affected, rhs.affected)):
                e = e.clone(lhs=lhs.new, rhs=rhs.new)

        # visit root node
        result = self.step(e)
        if result is not None:
            e = result

        # nothing new when `e` is still the same object as the `expression`.
        new: Optional[BaseExpression] = None if e is expression else e
        return PassResult(expression, new)

    def step(self, expression: BaseExpression) -> Optional[BaseExpression]:
        return None


class ConstantsFolding(Pass):

    def run(self, e: BaseExpression) -> PassResult:
        return self.walk(e)

    def step(self, e: BaseExpression) -> Optional[BaseExpression]:
        if isinstance(e, BinaryExpr):
            if all(isinstance(expr, Literal) for expr in (e.lhs, e.rhs)):
                return e.op(e.lhs, e.rhs)


class HistoryExpansion(Pass):

    def __init__(self, history: History):
        self.history = history

    def run(self, e: BaseExpression) -> PassResult:
        return self.walk(e)

    def step(self, expression: BaseExpression) -> Optional[BaseExpression]:
        if isinstance(expression, HistoryRef):
            return expression.resolve(self.history)


def main():
    repl = Repl()
    repl.block()


if __name__ == '__main__':
    main()
