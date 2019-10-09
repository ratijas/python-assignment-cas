import abc
import enum
from dataclasses import dataclass
from fractions import Fraction
from typing import Generic, TypeVar, Iterable, Optional, List, Sequence, Callable, MutableSequence, Any, Union, Set

from .exception import *
from .expression import *

__all__ = [
    'TokenType',
    'Token',
    'Cursor',
    'tokenize',
    'build_ast',
    'build_expr',
    'AstNodeType',
    'AstNode',
    'AstRaw',
    'AstExpand',
    'AstAtom',
    'AstParen',
    'AstBinaryExpr',
    'AstCompound',
    'Replace',
]


###############################################################################
#                                  Tokenizer                                  #
###############################################################################


class TokenType(enum.Enum):
    LParen = '('
    RParen = ')'
    LBracket = '['
    RBracket = ']'
    Expand = 'expand'
    Literal = 'literal'
    """value is a Literal instance"""
    Symbol = 'symbol'
    """value is a str with length 1"""
    Operator = 'operator'
    """value is a BinaryOperation enum instance representing particular operator"""


T = TypeVar('T')
Cursor = int


@dataclass
class Token(Generic[T]):
    tty: TokenType
    start: int
    end: int
    raw: str
    value: T


EXPAND = 'expand'
UNARY_SIGNS = "+-"
OPERATORS = '+-*/^'
PARENS = '()'
BRACKETS = '[]'
PARENS_AND_BRACKETS = {
    '(': TokenType.LParen,
    ')': TokenType.RParen,
    '[': TokenType.LBracket,
    ']': TokenType.RBracket,
}


def tokenize(s: str) -> Iterable[Token[Any]]:
    cursor: Cursor = 0
    cursor = skip_spaces(cursor, s)
    cursor, token = try_parse_expand(cursor, s)
    if token:
        yield token
    cursor = skip_spaces(cursor, s)
    while cursor < len(s):
        # preview first and next characters
        ch = s[cursor]
        cch = s[cursor + 1] if cursor + 1 < len(s) else "\0"

        if ch.isdigit() or (ch in UNARY_SIGNS and cch.isdigit()):
            cursor, token = parse_literal(cursor, s)
            yield token

        elif ch.isalpha():
            cursor, token = parse_symbol(cursor, s)
            yield token

        elif ch in OPERATORS:
            cursor, token = parse_operator(cursor, s)
            yield token

        elif ch in PARENS_AND_BRACKETS:
            cursor, token = parse_parens_brackets(cursor, s)
            yield token

        else:
            raise ParseError(s, cursor, cursor, 'unexpected token')

        cursor = skip_spaces(cursor, s)


def try_parse_expand(cursor: Cursor, s: str) -> (Cursor, Optional[Token[None]]):
    if s.startswith(EXPAND, cursor):
        start = cursor
        cursor += len(EXPAND)
        end = cursor - 1
        check_next(cursor, s, EXPAND)
        cursor = skip_spaces(cursor, s)
        return cursor, Token(TokenType.Expand, start, end, EXPAND, None)
    return cursor, None


def skip_spaces(cursor: Cursor, s: str) -> Cursor:
    """return pointer at first non-whitespace character."""
    while cursor < len(s) and s[cursor].isspace():
        cursor += 1
    return cursor


def check_next(cursor: Cursor, s: str, description: str) -> None:
    if cursor >= len(s):
        return
    ch = s[cursor]
    if ch.isspace():
        return
    if ch in "()[]+-*/^.":
        return
    raise ParseError(s, cursor, cursor, description)


def parse_int(cursor: Cursor, s: str) -> (Cursor, Token[int]):
    start = cursor
    # sign
    if cursor < len(s) and s[cursor] in UNARY_SIGNS:
        cursor += 1
    # digits
    while cursor < len(s) and s[cursor].isdigit():
        cursor += 1
    sub = s[start:cursor]
    if len(sub) > 0:
        return cursor, Token(TokenType.Literal, start, cursor - 1, sub, int(sub, 10))

    raise ParseError(s, start, cursor, 'integer')


def parse_literal(cursor: Cursor, s: str) -> (Cursor, Token[Literal]):
    # cursor = skip_spaces(cursor, s)
    start = cursor
    cursor, pre = parse_int(cursor, s)

    if s.startswith("/", cursor) or s.startswith(".", cursor):
        ch = s[cursor]
        cursor += 1
        cursor, post = parse_int(cursor, s)
        end = cursor - 1
        raw = s[start:cursor]

        # fraction
        if ch == "/":
            literal = Literal(Fraction(pre.value, post.value))

        # decimal
        else:
            literal = Literal(float(raw))

        return cursor, Token(TokenType.Literal, start, end, raw, literal)

    end = cursor - 1

    return cursor, Token(TokenType.Literal, start, end, pre.raw, Literal(pre.value))


def parse_symbol(cursor: Cursor, s: str) -> (Cursor, Token[Symbol]):
    symbol = s[cursor]
    return cursor + 1, Token(TokenType.Symbol, cursor, cursor, symbol, Symbol(symbol))


def parse_operator(cursor: Cursor, s: str) -> (Cursor, Token[BinaryOperation]):
    assert cursor < len(s)
    assert s[cursor] in OPERATORS

    op = s[cursor]
    return cursor + 1, Token(TokenType.Operator, cursor, cursor, op, BinaryOperation(op))


def parse_parens_brackets(cursor: Cursor, s: str) -> (Cursor, Token[None]):
    assert cursor < len(s)
    assert s[cursor] in PARENS_AND_BRACKETS

    raw = s[cursor]
    tty = PARENS_AND_BRACKETS[raw]
    return cursor + 1, Token(tty, cursor, cursor, raw, None)


###############################################################################
#                                     AST                                     #
###############################################################################

class AstNodeType(enum.Enum):
    Raw = enum.auto()
    Expand = enum.auto()
    Atom = enum.auto()
    Paren = enum.auto()
    BinaryExpr = enum.auto()
    Compound = enum.auto()


@dataclass
class AstNode(Generic[T], metaclass=abc.ABCMeta):
    value: T
    start: int
    end: int
    raw: str

    @property
    @abc.abstractmethod
    def ty(self) -> AstNodeType:
        pass

    # @abc.abstractmethod
    def into_expr(self, source: str) -> BaseExpression:
        raise ParseError(source, self.start, self.end,
                         f'AST node of type {type(self)} cannot be converted into expession')


@dataclass
class AstRaw(AstNode[Token]):
    """Transparent wrapper for token.

    All raw nodes must be replaced by the end of parsing process.
    """

    @classmethod
    def from_token(cls, token: Token) -> 'AstRaw':
        return cls(token, token.start, token.end, token.raw)

    @property
    def ty(self) -> AstNodeType:
        return AstNodeType.Raw


@dataclass
class AstExpand(AstNode[BaseExpression]):
    """Expand command with attached expression"""

    @property
    def ty(self) -> AstNodeType:
        return AstNodeType.Expand

    def into_expr(self, source: str) -> BaseExpression:
        return ExpandExpression(self.value)


@dataclass
class AstAtom(AstNode[Union[Literal, Symbol, HistoryRef]]):
    """Either literal, symbol or history ref"""

    @property
    def ty(self) -> AstNodeType:
        return AstNodeType.Atom

    def into_expr(self, source: str) -> BaseExpression:
        return self.value


@dataclass
class AstParen(AstNode[AstNode]):
    """Parens are the building blocks of any expression"""

    @property
    def ty(self) -> AstNodeType:
        return AstNodeType.Paren

    def into_expr(self, source: str) -> BaseExpression:
        return self.value.into_expr(source)


@dataclass
class AstBinaryExpr(AstNode[BinaryExpr]):
    """Building binary expressions is the second task after resolving parens.

    Those AstBinaryExpr, which are direct descendants of AstParen, should be simplified
    into just AstBinaryExpr with underlying expression's `parens` property set to True."""

    @property
    def ty(self) -> AstNodeType:
        return AstNodeType.BinaryExpr

    def into_expr(self, source: str) -> BaseExpression:
        return self.value


@dataclass
class AstCompound(AstNode[List[BaseExpression]]):
    """Multiplication written in-line without operator, e.g.: '2xy'.

    Such expression is a special case of binary multiplication.
    """

    @property
    def ty(self) -> AstNodeType:
        return AstNodeType.Compound

    def into_expr(self, source: str) -> BaseExpression:
        return CompoundExpression(self.value)


def build_expr(source: str) -> BaseExpression:
    return build_ast(source).into_expr(source)


def build_ast(source: str) -> AstNode:
    tokens = list(tokenize(source))
    nodes = list(map(AstRaw.from_token, tokens))

    n_start = 0
    n_end = len(nodes) - 1 if len(nodes) > n_start else n_start
    return build_with_reducers(source, nodes, n_start, n_end)


def build_with_reducers(source: str, nodes: MutableSequence[AstNode], n_start: Cursor, n_end: Cursor) -> AstNode:
    """Reduce all subsequent nodes from `n_start` into single AST node."""
    reducers = [
        ExpandReducer(),
        LiteralsReducer(),
        HistoryReducer(),
        ParensReducer(),
        BinaryReducer({BinaryOperation.Pow}),
        CompoundReducer(),
        BinaryReducer({BinaryOperation.Mul, BinaryOperation.Div}),
        BinaryReducer({BinaryOperation.Add, BinaryOperation.Sub}),
        RedundantParensReducer(),
        ParensToBinaryExprReducer(),
        TopLevelParensReducer(),
    ]
    # not the most efficient algorithm, but should work.
    # Quiet similar to the one used at REPL evaluation stage.
    changed = True  # kind of do-while loop in C
    while changed:
        changed = False
        for reducer in reducers:
            # apply all replacements suggested by reducer
            while True:  # while ((replace := ...) is not None):
                replace = reducer.reduce(source, nodes, n_start, n_end)
                if replace is None: break

                replace.apply(nodes)
                # shift `n_end`
                n_end += replace.diff
                # indicate that something has changed
                changed = True

    if n_end < n_start:
        at = nodes[n_start - 1].end + 1 if (n_start - 1) in range(len(nodes)) else 0
        to = nodes[n_end + 1].start - 1 if (n_end + 1) in range(len(nodes)) else 0
        raise ParseError(source, at, to, 'no content')

    if n_end > n_start:
        raise ParseError(source, nodes[n_start + 1].start, len(source) - 1, 'leftovers')

    return nodes[n_start]


@dataclass
class Replace:
    start: Cursor
    end: Cursor
    target: Sequence[AstNode]

    @property
    def diff(self) -> int:
        old = self.end - self.start + 1
        new = len(self.target)
        return new - old

    @classmethod
    def one(cls, at: Cursor, target: AstNode) -> 'Replace':
        return cls(at, at, [target])

    def apply(self, nodes: MutableSequence[AstNode]):
        """Modify nodes list in-place."""
        nodes[self.start:self.end + 1] = self.target


def filter_raw_node_token_type(tty: TokenType) -> Callable[[AstNode], bool]:
    return lambda node: isinstance(node, AstRaw) and node.value.tty == tty


filter_raw_node_left_paren = filter_raw_node_token_type(TokenType.LParen)
filter_raw_node_right_paren = filter_raw_node_token_type(TokenType.RParen)


class Reducer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reduce(self, source: str, nodes: Sequence[AstNode], n_start: Cursor, n_end: Cursor) -> Optional[Replace]:
        """Called once for each replacement until no more replacements can be made.

        Parameters `n_start` & `n_end` are absolute positions of nodes in `nodes` list.
        """


class ExpandReducer(Reducer):
    def reduce(self, source: str, nodes: Sequence[AstNode], n_start: Cursor, n_end: Cursor) -> Optional[Replace]:
        if len(nodes) > n_start:
            expand = nodes[n_start]
            if expand.ty == AstNodeType.Raw and expand.value.tty == TokenType.Expand:
                node = build_with_reducers(source, list(nodes), n_start + 1, n_end)
                expr = node.into_expr(source)
                s_start, s_end = expand.start, nodes[-1].end
                target = AstExpand(expr, s_start, s_end, source[s_start:s_end + 1])
                return Replace(n_start, n_end, [target])


class LiteralsReducer(Reducer):
    """Reduce AstRaw literals and symbols nodes to AstAtom nodes"""

    def reduce(self, source: str, nodes: Sequence[AstNode], n_start: Cursor, n_end: Cursor) -> Optional[Replace]:
        literal_filter = filter_raw_node_token_type(TokenType.Literal)
        symbol_filter = filter_raw_node_token_type(TokenType.Symbol)

        for i in range(n_start, n_end + 1):
            node = nodes[i]
            if literal_filter(node) or symbol_filter(node):
                token = node.value  # type: Token[Union[Literal, Symbol]]
                return Replace.one(i, AstAtom(token.value, token.start, token.end, token.raw))


class HistoryReducer(Reducer):
    """Reduce sequence of {LBracket <HistoryItem> RBracket} to {History}

    where <HistoryItem> is either Literal[int] or token "last".
    """

    def reduce(self, source: str, nodes: Sequence[AstNode], n_start: Cursor, n_end: Cursor) -> Optional[Replace]:
        filter_lbr = filter_raw_node_token_type(TokenType.LBracket)
        filter_rbr = filter_raw_node_token_type(TokenType.RBracket)
        filter_atom_literal = lambda n: \
            isinstance(n, AstAtom) and \
            isinstance(n.value, Literal)

        for i in range(n_start, n_end + 1):
            lbr = nodes[i]
            if filter_lbr(lbr):
                # token "last"
                # костыль mode on
                if i + 5 <= n_end:
                    atoms = nodes[i + 1:i + 5]
                    rbr = nodes[i + 5]

                    inner_start = atoms[0].start
                    inner_end = atoms[-1].end
                    if source[inner_start:inner_end + 1] == "last":
                        history = HistoryRef(-1)
                        start = lbr.start
                        end = rbr.end
                        target = AstAtom(history, start, end, source[start:end + 1])
                        return Replace(i, i + 5, [target])

                # Literal[int]
                if i + 2 > n_end:
                    raise ParseError(source, lbr.start, lbr.end,
                                     'opened history reference without matching bracket')

                atom, rbr = nodes[i + 1], nodes[i + 2]

                if not filter_atom_literal(atom) or not isinstance(atom.value.literal, int):
                    raise ParseError(source, atom.start, atom.end,
                                     'history index must be either an integer literal or token "last"')

                if not filter_rbr(rbr):
                    raise ParseError(source, rbr.start, rbr.end, 'expected "]"')

                history = HistoryRef(atom.value.literal)
                start, end = lbr.start, rbr.end
                target = AstAtom(history, start, end, source[start:end + 1])
                return Replace(i, i + 2, [target])


class ParensReducer(Reducer):
    """Recursively parse parens"""

    def reduce(self, source: str, nodes: Sequence[AstNode], n_start: Cursor, n_end: Cursor) -> Optional[Replace]:
        # do not use recursion, instead replace
        # create replacer for the inner-most parens.
        stack: List[Cursor] = []
        for i in range(n_start, n_end + 1):
            node = nodes[i]
            if filter_raw_node_left_paren(node):
                stack.append(i)

            elif filter_raw_node_right_paren(node):
                p_start = stack.pop()
                p_end = i
                child = build_with_reducers(source, list(nodes), p_start + 1, p_end - 1)
                s_start = nodes[p_start].start
                s_end = nodes[p_end].end
                target = AstParen(child, s_start, s_end, source[s_start:s_end + 1])
                return Replace(p_start, p_end, [target])

        return None


class BinaryReducer(Reducer):
    """Reduce binary expression like {<lhs> BinaryOperator <rhs>} into AstBinaryExpr"""

    def __init__(self, operators: Set[BinaryOperation]):
        """Create BinaryReducer for a set of operators of same priority."""
        super().__init__()

        self.operators = operators

    def reduce(self, source: str, nodes: Sequence[AstNode], n_start: Cursor, n_end: Cursor) -> Optional[Replace]:
        for i in range(n_start, n_end + 1):
            node = nodes[i]
            if self.filter(node):
                op: BinaryOperation = node.value.value
                n_lhs, n_rhs = i - 1, i + 1
                if n_lhs < n_start or n_rhs > n_end:
                    raise ParseError(source, node.start, node.end,
                                     'binary operation missing argument(s)')
                lhs, rhs = nodes[n_lhs], nodes[n_rhs]
                s_start, s_end = lhs.start, rhs.end
                lhs_expr = lhs.into_expr(source)  # potentially fail-able operations
                rhs_expr = rhs.into_expr(source)  # on separate lines
                target = AstBinaryExpr(BinaryExpr(lhs_expr, op, rhs_expr), lhs.start, rhs.end,
                                       source[s_start:s_end + 1])
                return Replace(n_lhs, n_rhs, [target])

    def filter(self, node: AstNode) -> bool:
        if not isinstance(node, AstRaw): return False
        token = node.value
        if token.tty != TokenType.Operator: return False
        operator: BinaryOperation = token.value
        return operator in self.operators


class CompoundReducer(Reducer):
    """Reduce subsequent literals and symbols into one AstCompound object.

    Reducing { Literal Symbol+ } into Compound([literal, *symbols]).
    """

    def reduce(self, source: str, nodes: Sequence[AstNode], n_start: Cursor, n_end: Cursor) -> Optional[Replace]:
        for i in range(n_start, n_end + 1):
            node = nodes[i]
            if CompoundReducer.filter_node(node):
                j = i
                while j + 1 <= n_end:
                    if CompoundReducer.is_literal(nodes[j]) and CompoundReducer.is_literal(nodes[j + 1]):
                        raise ParseError(source, nodes[j].start, nodes[j + 1].end,
                                         'compound can not contain two literals in a row')
                    if CompoundReducer.filter_node(nodes[j + 1]):
                        j += 1
                    else:
                        break
                if j != i:
                    c_start, c_end = nodes[i], nodes[j]
                    s_start, s_end = c_start.start, c_end.end
                    raw = source[s_start:s_end + 1]
                    expr_list = [n.into_expr(source) for n in nodes[i:j + 1]]
                    target = AstCompound(expr_list, s_start, s_end, raw)
                    return Replace(i, j, [target])

    @staticmethod
    def filter_node(node: AstNode) -> bool:
        """Is it suitable node for compound multiplication?"""
        if isinstance(node, AstAtom):
            return CompoundReducer.filter_expr(node.value)
        if isinstance(node, AstBinaryExpr):
            expr = node.value
            if expr.op is BinaryOperation.Pow:
                return CompoundReducer.filter_expr(expr.lhs) and \
                       CompoundReducer.filter_expr(expr.rhs)

    @staticmethod
    def filter_expr(expr: BaseExpression) -> bool:
        return isinstance(expr, (Literal, Symbol))

    @staticmethod
    def is_literal(node: AstNode):
        return isinstance(node, AstAtom) and isinstance(node.value, Literal)


class RedundantParensReducer(Reducer):
    """Reduce double-stacked parens into single layer."""

    def reduce(self, source: str, nodes: Sequence[AstNode], n_start: Cursor, n_end: Cursor) -> Optional[Replace]:
        for i in range(n_start, n_end + 1):
            node = nodes[i]
            if isinstance(node, AstParen) and isinstance(node.value, AstParen):
                return Replace(i, i, [node.value])


class TopLevelParensReducer(Reducer):
    """Reduce top-level binary expression's parens into nothing."""

    def reduce(self, source: str, nodes: Sequence[AstNode], n_start: Cursor, n_end: Cursor) -> Optional[Replace]:

        if n_start == 0:
            node = nodes[0]
            if isinstance(node, AstExpand):
                expr = node.value
                if isinstance(expr, BinaryExpr) and expr.parens is not False:  # None or True
                    expr = expr.clone(parens=False)
                    target = AstExpand(expr, node.start, node.end, node.raw)
                    return Replace(0, 0, [target])

            elif isinstance(node, AstBinaryExpr) and node.value.parens is not False:  # None or True
                expr = node.value.clone(parens=False)
                target = AstBinaryExpr(expr, node.start, node.end, node.raw)
                return Replace(0, 0, [target])


class ParensToBinaryExprReducer(Reducer):
    """Reduce { AstParen(AstBinaryExpr(...)) } into { AstBinaryExpr(parens=True) } """

    def reduce(self, source: str, nodes: Sequence[AstNode], n_start: Cursor, n_end: Cursor) -> Optional[Replace]:
        for i in range(n_start, n_end + 1):
            node = nodes[i]
            if isinstance(node, AstParen) and isinstance(node.value, AstBinaryExpr):
                child = node.value
                expr = child.value.clone(parens=True)
                target = AstBinaryExpr(expr, child.start, child.end, child.raw)
                return Replace(i, i, [target])
