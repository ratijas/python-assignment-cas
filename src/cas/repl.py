import atexit
import readline
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional

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
        try:
            return build_expr(raw)

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
            CompoundSorting(),
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
    def affected(self) -> bool:
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

        elif isinstance(e, CompoundExpression):
            results = [self.walk(expr) for expr in e.inner]

            if any(inner.affected for inner in results):
                e = e.clone(r.new for r in results)

        # visit root node
        result = self.step(e)
        if result is not None:
            e = result

        # nothing new when `e` is still the same object as the `expression`.
        new: Optional[BaseExpression] = None if e is expression else e
        return PassResult(expression, new)

    def step(self, expression: BaseExpression) -> Optional[BaseExpression]:
        return None


class CompoundSorting(Pass):

    def run(self, e: BaseExpression) -> PassResult:
        return self.walk(e)

    def step(self, e: BaseExpression) -> Optional[BaseExpression]:
        if isinstance(e, CompoundExpression):
            new = sorted(e.inner, key=str)
            if e.inner != new:
                return e.clone(new)


class ConstantsFolding(Pass):

    def run(self, e: BaseExpression) -> PassResult:
        return self.walk(e)

    def step(self, e: BaseExpression) -> Optional[BaseExpression]:
        if isinstance(e, BinaryExpr):
            if all(isinstance(expr, Literal) for expr in (e.lhs, e.rhs)):
                return e.op(e.lhs, e.rhs)
        if isinstance(e, CompoundExpression):
            literals = [i for i, ex in enumerate(e.inner) if isinstance(ex, Literal)]
            if len(literals) > 1:
                factor = reduce(BinaryOperation.Mul, (e.inner[i] for i in literals))
                # remove literals
                e = e.clone()
                e.inner = [ex.clone()
                           for i, ex in enumerate(e.inner)
                           if i not in literals]
                # prepend factor, if it would make sense
                if factor != Literal(1):
                    e.inner.insert(0, factor)
                return e


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
