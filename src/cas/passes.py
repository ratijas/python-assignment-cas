from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from typing import Dict, List, MutableSequence, Optional, Sequence, Tuple

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
                e = e.clone([(r.new
                              if r.new is not None
                              else r.old.clone())
                             for r in results])

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
                e = e.clone([ex.clone()
                             for i, ex in enumerate(e.inner)
                             if i not in literals])
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

        if isinstance(e, BinaryExpr) and e.op.is_mul_div:
            power = {BinaryOperation.Mul: 1, BinaryOperation.Div: -1}[e.op]
            # compounds <op> symbol
            if isinstance(e.lhs, CompoundExpression) and isinstance(e.rhs, (Symbol, Literal)):
                return e.lhs.clone([
                    *e.lhs.inner,
                    BinaryExpr(e.rhs, BinaryOperation.Pow, Literal(power), False),
                ])
            # make sure each of compounds gets copy of power
            # symbol / compounds -> symbol * compounds^-1
            if isinstance(e.lhs, (Symbol, Literal)) and isinstance(e.rhs, CompoundExpression):
                return e.rhs.clone([
                    e.lhs,
                    *[BinaryExpr(rhs, BinaryOperation.Pow, Literal(power), False)
                      for rhs in e.rhs.inner],
                ])
            # symbol * literal -> compound
            if isinstance(e.lhs, (Symbol, Literal)) and isinstance(e.rhs, (Symbol, Literal)):
                if isinstance(e.rhs, Symbol) and e.op is BinaryOperation.Div:
                    return
                return CompoundExpression([
                    e.lhs,
                    BinaryExpr(e.rhs, BinaryOperation.Pow, Literal(power), False),
                ])
            # compound * compound -> OneBigCompound
            if isinstance(e.lhs, CompoundExpression) and isinstance(e.rhs, CompoundExpression):
                return CompoundExpression([
                    *e.lhs.inner,
                    *[BinaryExpr(rhs, BinaryOperation.Pow, Literal(power), False)
                      for rhs in e.rhs.inner]
                ])

        # Compound xxyyy -> x^2y^3
        if isinstance(e, CompoundExpression):
            components = self.collect_compound_components_by_base(e)
            result = self.assemble_compound_from_components(components)

            if e != result:
                return result

    @staticmethod
    def starts_with_literal(e: CompoundExpression) -> bool:
        return len(e.inner) > 0 and isinstance(e.inner[0], Literal)

    @staticmethod
    def same_factors(lhs: CompoundExpression, rhs: CompoundExpression) -> bool:
        return CompoundExpression(lhs.inner[1:]) == CompoundExpression(rhs.inner[1:])

    @classmethod
    def collect_compound_components_by_base(cls, e: CompoundExpression) -> Dict[BaseExpression, List[BaseExpression]]:
        # factors are added to each other in the end
        components: Dict[BaseExpression, List[BaseExpression]] = defaultdict(list)
        for expr in e.inner:

            # component := base ^ power
            if isinstance(expr, BinaryExpr) and expr.op is BinaryOperation.Pow:
                base, power = expr.lhs, expr.rhs
                components[base].append(power)

            # component := expr
            else:
                base, power = expr, Literal(1)
                # get and increment by one
                lst = components[base]
                idx, literal = cls.get_or_insert_literal(lst)
                lst[idx] = literal + power

        return components

    @classmethod
    def get_or_insert_literal(cls, components: MutableSequence[BaseExpression],
                              default: Literal = Literal(0)) -> Tuple[int, Literal]:
        for idx, expr in enumerate(components):
            if isinstance(expr, Literal):
                return idx, expr
        else:
            components.append(default)
            return len(components) - 1, default

    @classmethod
    def assemble_compound_from_components(cls,
                                          components: Dict[BaseExpression, List[BaseExpression]]) -> CompoundExpression:
        expressions: List[BaseExpression] = []

        for base in sorted(components.keys(), key=str):
            powers = components[base]
            result = cls.assemble_base_with_power(base, powers)
            if result != Literal(1):
                expressions.append(result)

        if len(expressions) == 0:
            expressions.append(Literal(1))

        return CompoundExpression(expressions)

    @classmethod
    def assemble_base_with_power(cls, base: BaseExpression,
                                 powers: Sequence[BaseExpression]) -> Optional[BaseExpression]:
        assert len(powers) > 0

        if len(powers) == 1:
            power = powers[0]

            # a ^ 0 == 1.
            if power == Literal(0):
                return Literal(1)

            # a ^ 1 == a
            elif power == Literal(1):
                return base

            # a ^ b == ?
            else:
                return BinaryExpr(base, BinaryOperation.Pow, power)

        # a ^ (b + c + ...)
        else:
            power = reduce(cls.combine_sum, powers)
            return BinaryExpr(base, BinaryOperation.Pow, power)

    @classmethod
    def combine_sum(cls, lhs: BaseExpression, rhs: BaseExpression) -> BaseExpression:
        return BinaryExpr(lhs, BinaryOperation.Add, rhs, False)


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
            power = e.rhs
            if isinstance(power, BinaryExpr) and power.parens is not True:
                return e.clone(rhs=power.clone(parens=True))
            if isinstance(power, CompoundExpression) and power.parens is not True:
                return e.clone(rhs=power.clone(parens=True))


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
