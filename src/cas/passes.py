from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from typing import Optional, Sequence, Dict, List

from .expression import *
from .history import *

__all__ = [
    'PassResult',
    'Pass',
    'CompoundSorting',
    'ConstantsFolding',
    'HistoryExpansion',
    'evaluate',
]


@dataclass
class PassResult:
    old: BaseExpression
    new: Optional[BaseExpression]

    @property
    def any(self) -> BaseExpression:
        return self.new if self.new is not None else self.old

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
        if isinstance(e, BinaryExpr) and \
                isinstance(e.lhs, Literal) and isinstance(e.rhs, Literal):
            return e.op(e.lhs, e.rhs)

        if isinstance(e, CompoundExpression):
            literals = [i for i, ex in enumerate(e.inner) if isinstance(ex, Literal)]
            if len(literals) > 1:
                factor = reduce(BinaryOperation.Mul, (e.inner[i] for i in literals))
                # remove literals
                e = CompoundExpression(ex.clone()
                                       for i, ex in enumerate(e.inner)
                                       if i not in literals)
                # prepend factor, if it would make sense
                if factor != Literal(1) or len(e.inner) == 0:
                    e.inner.insert(0, factor)
                return e


class FactorsFolding(Pass):

    def run(self, e: BaseExpression) -> PassResult:
        return self.walk(e)

    def step(self, e: BaseExpression) -> Optional[BaseExpression]:
        #        Binary {* | /}
        #        /          \
        # M*compounds <op> N*compounds
        if isinstance(e, BinaryExpr) and e.op.is_add_sub and \
                isinstance(e.lhs, CompoundExpression) and isinstance(e.rhs, CompoundExpression) and \
                FactorsFolding.starts_with_literal(e.lhs) and FactorsFolding.starts_with_literal(e.rhs) and \
                self.same_factors(e.lhs, e.rhs):
            factor = e.op.operator(e.lhs.inner[0], e.rhs.inner[0])
            common = e.lhs.inner[1:]
            return CompoundExpression([factor, *common])

        # Compound xxyyy -> x^2y^3
        if isinstance(e, CompoundExpression):
            factors: Dict[BaseExpression, CompoundExpression] = defaultdict(lambda: CompoundExpression(()))
            for factor in e.inner:
                print(f'factor: {factor} {factor!r}')
                if isinstance(factor, BinaryExpr) and factor.op is BinaryOperation.Pow:
                    factors[factor.lhs].inner.append(factor.rhs)
                else:
                    for index, f in enumerate(factors[factor].inner):
                        if isinstance(f, Literal):
                            factors[factor].inner[index] = f + Literal(1)
                            break
                    else:
                        factors[factor].inner.insert(0, Literal(1))

            components: List[BaseExpression] = []
            for factor, power in factors.items():
                # print(f"BBBBBBBBB {factor}, {power}")
                # power = ConstantsFolding().run(power).old
                # print(f"ANYYY {power}")
                if isinstance(power, CompoundExpression) and len(power.inner) == 1:
                    power = power.inner[0]

                    if power == Literal(1):
                        components.append(factor)
                        continue

                components.append(BinaryExpr(factor, BinaryOperation.Pow, power))

            result = CompoundExpression(components)
            if e != result:
                return result

    @staticmethod
    def starts_with_literal(e: CompoundExpression) -> bool:
        return len(e.inner) > 0 and isinstance(e.inner[0], Literal)

    @staticmethod
    def same_factors(lhs: CompoundExpression, rhs: CompoundExpression) -> bool:
        return CompoundExpression(lhs.inner[1:]) == CompoundExpression(rhs.inner[1:])


class HistoryExpansion(Pass):

    def __init__(self, history: History):
        self.history = history

    def run(self, e: BaseExpression) -> PassResult:
        return self.walk(e)

    def step(self, expression: BaseExpression) -> Optional[BaseExpression]:
        if isinstance(expression, HistoryRef):
            if expression.item < 0:
                expression.item = self.history.next_number + expression.item

            return expression.resolve(self.history)


class Expanding(Pass):

    def run(self, e: BaseExpression) -> PassResult:
        if isinstance(e, ExpandExpression):
            result = self.walk(e.inner)
            return PassResult(e, result.new)
        return PassResult(e, None)

    def step(self, e: BaseExpression) -> Optional[BaseExpression]:
        if isinstance(e, Literal) and isinstance(e.literal, Fraction):
            return Literal(float(e.literal))

        if isinstance(e, BinaryExpr) and e.op.is_mul_div:
            if isinstance(e.lhs, BinaryExpr) and e.lhs.op.is_add_sub:
                # e == (lhs.l {+ | -} lhs.r) {* | /} rhs -> ((lhs.l {* | /} rhs) {+ | -} (lhs.r {* | /} rhs))
                new_lhs = BinaryExpr(e.lhs.lhs, e.op, e.rhs, False)
                new_rhs = BinaryExpr(e.lhs.rhs, e.op, e.rhs, False)
                return BinaryExpr(new_lhs, e.lhs.op, new_rhs, True)
            if isinstance(e.rhs, BinaryExpr) and e.rhs.parens is not False:
                # e == lhs {* | /} (rhs.l {+ | -} rhs.r)
                new_lhs = BinaryExpr(e.lhs, e.op, e.rhs.lhs, False)
                new_rhs = BinaryExpr(e.lhs, e.op, e.rhs.rhs, False)
                return BinaryExpr(new_lhs, e.rhs.op, new_rhs, True)


class RemoveOuterParens(Pass):

    def run(self, expression: BaseExpression) -> PassResult:
        if isinstance(expression, BinaryExpr) and expression.parens is not False:
            return PassResult(expression, expression.clone(parens=False))
        return PassResult(expression, None)


class WrapPowerRhsInParens(Pass):

    def run(self, e: BaseExpression) -> PassResult:
        return self.walk(e)

    def step(self, e: BaseExpression) -> Optional[BaseExpression]:
        if isinstance(e, BinaryExpr) and e.op is BinaryOperation.Pow and len(str(e.rhs)) > 1:
            if isinstance(e.rhs, BinaryExpr) and e.rhs.parens is not True:
                return e.clone(rhs=e.rhs.clone(parens=True))
            if isinstance(e.rhs, CompoundExpression) and e.rhs.parens is not True:
                return e.clone(rhs=e.rhs.clone(parens=True))


def evaluate(history: History, expression: BaseExpression, passes: Optional[Sequence[Pass]] = None) -> BaseExpression:
    """try evaluate an expression.

    :param history: REPL history
    :param expression: input expression
    :param passes: enabled passes
    :return: result expression, defaults to all
    :raise InvalidExpressionError: if unable to compute
    """
    passes: Sequence[Pass] = [
        ConstantsFolding(),
        CompoundSorting(),
        HistoryExpansion(history),
        FactorsFolding(),
        Expanding(),
        # cosmetics
        RemoveOuterParens(),
        WrapPowerRhsInParens(),
    ] if passes is None else passes

    # loop through passes, applying them all one by one.
    # then, if any of them resulted in expression change,
    # re-run all passes; and so on until no pass mutates
    # the expression.

    affected = True
    while affected:
        affected = False
        for p in passes:
            result = p.run(expression)
            if result.affected:
                affected = True
                expression = result.new

    return expression
