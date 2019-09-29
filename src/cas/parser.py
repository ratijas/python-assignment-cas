import abc
import enum
from dataclasses import dataclass
from fractions import Fraction
from typing import Generic, TypeVar, Iterable, Optional, List, Sequence, Callable, MutableSequence, Any

from .expression import *
from .exception import *

__all__ = [
    'TokenType',
    'Token',
    'Cursor',
    'tokenize',
    'build_ast',
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
    """value is a char representing particular operator"""


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


# def parse_history(cursor: Cursor, s: str) -> (Cursor, Token[int]):
#     start = cursor
#     if s[cursor] == '[':
#         cursor = skip_spaces(cursor + 1, s)
#         cursor, t = parse_int(cursor, s)
#         check_next(cursor, s, 'integer')
#         cursor = skip_spaces(cursor, s)
#         if s[cursor] == ']':
#             end = cursor
#             cursor += 1
#             sub = s[start:cursor]
#
#             return cursor, Token(TokenType.History, start, end, sub, t.value)


# verified
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

def build_ast(source: str) -> BaseExpression:
    tokens = list(tokenize(source))

    if find_one(tokens, PatternTokenType([TokenType.Expand]), 0) == 0:
        index = find_one(tokens, PatternTokenType([TokenType.Expand]), 1)
        if index is not None:
            t = tokens[index]
            raise ParseError(source, t.start, t.end, '"expand" may only appear at the beginning')

        return ExpandExpression(parse_tokens(source, tokens, 1))

    return parse_tokens(source, tokens, 0)


def parse_tokens(source: str, tokens: List[Token], start: int = 0) -> BaseExpression:
    # replace {"expand" expr...} with ExpandExpression object
    # for i in find_all(tokens, PatternTokenType([TokenType.Expand]), start):
    #     inner = parse_tokens(source, tokens, i + 1)
    #     return ExpandExpression(inner)

    # replace sequence of {LBracket Literal[int] RBracket} with {History}
    # TODO: AST node
    # replace_all(PatternTokenType([TokenType.LBracket, TokenType.Literal, TokenType.RBracket]),
    #             history_replacer,
    #             source, tokens, start)

    # top priority are parens: ( expr )
    # later they will be removed by yet another filter
    replace_all(PatternCombinator([
        PatternTokenType([TokenType.LParen]),
        PatternUntil(PatternTokenType([TokenType.RParen]))
    ]),
        parens_replacer,
        source, tokens, start)

    # for i in find_all(tokens, PatternTokenType([TokenType.LBracket, TokenType.Literal, TokenType.RBracket]), start):
    #     # LBracket at index i
    #     lbr, n, rbr = tokens[i:i + 2]
    #     start, end = lbr.start, rbr.end
    #     raw = source[start:end]
    #     value = n.value
    #     tokens[i:i + 2] = [Token(TokenType.History, start, end, raw, value)]

    # for i in find_all(tokens, PatternTokenType([TokenType.Literal, TokenType.Symbol]), start):
    #     pass
    return tokens


# def history_replacer(source: str, tokens: Sequence[Token]) -> Sequence[Token]:
#     assert len(tokens) == 3
#
#     lbr, n, rbr = tokens[0], tokens[1], tokens[2]
#
#     if not isinstance(n.value, Literal) or not isinstance(n.value.literal, int):
#         raise ParseError(source, n.start, n.end, 'history index must be integer literal')
#
#     start, end = lbr.start, rbr.end
#     raw = source[start:end + 1]
#     return [Token(TokenType.History, start, end, raw, n.value.literal)]


class IPattern(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def match(self, tokens: Sequence[Token], start: int) -> Optional[int]:
        """Note that 0 (zero) is a valid match length"""
        return None

    @abc.abstractmethod
    def min_len(self) -> int:
        return 0


class PatternTokenType(IPattern):

    def __init__(self, types: Sequence[TokenType]) -> None:
        super().__init__()

        self.types = types

    def match(self, tokens: Sequence[Token], start: int) -> Optional[int]:
        if start + self.min_len() > len(tokens):
            return 0

        if all(token.tty == tty
               for token, tty in zip(tokens[start:], self.types)):
            return len(self.types)

        return None

    def min_len(self) -> int:
        return len(self.types)


class PatternCombinator(IPattern):

    def __init__(self, patterns: Sequence[IPattern]) -> None:
        super().__init__()

        self.patterns = patterns

    def match(self, tokens: Sequence[Token], start: int) -> Optional[int]:
        length = 0
        for p in self.patterns:
            match = p.match(tokens, start)
            if match is None:
                return 0
            start += match
            length += match
        return length

    def min_len(self) -> int:
        return sum(p.min_len() for p in self.patterns)


class PatternUntil(IPattern):
    """Up until and including sub-pattern."""

    def __init__(self, pattern: IPattern):
        super().__init__()

        self.pattern = pattern

    def match(self, tokens: Sequence[Token], start: int) -> Optional[int]:
        if start + self.min_len() > len(tokens):
            return 0

        for i in range(start, len(tokens)):
            match = self.pattern.match(tokens, i)
            if match is not None:
                return i - start + match

    def min_len(self) -> int:
        return self.pattern.min_len()


# def match(self, tokens: List[Token]) -> Match:
#     pass


# class Rule:
#     pattern: Pattern
#     substitution: Pattern


# def match(tokens: List[Token], pattern: Pattern)

def find_one(tokens: Sequence[Token], pattern: IPattern, start: int) -> Optional[int]:
    for i in range(start, len(tokens)):
        if pattern.match(tokens, i):
            return i


# def find_all(tokens: Sequence[Token], pattern: IPattern, start: int) -> Iterable[int]:
#     while True:
#         index = find_one(tokens, pattern, start)
#         if index is None:
#             return
#
#         yield index
#         # it is reasonable to assume that tokens list will be changed between runs
#         break

#
# for i in range(start, len(tokens)):
#
#     if pattern.match(tokens, i):
#         yield i
#         # it is reasonable to assume that tokens list will be changed between runs
#         break
# else:
#     # if no matches found during the run over whole sequence of tokens
#     return
#


def replace_all(pattern: IPattern,
                replacer: Callable[[str, Sequence[Token]], Sequence[Token]],
                source: str,
                tokens: MutableSequence[Token],
                start: int = 0):
    index = start
    while index + pattern.min_len() <= len(tokens):
        match = pattern.match(tokens, index)
        if match:
            tokens[index:index + match] = replacer(source, tokens[index:index + match])

        index += 1


def parens_replacer(source: str, tokens: Sequence[Token]) -> Sequence[Token]:
    assert len(tokens) >= 3  # parent are the first and the last tokens
    tokens = tokens[1:-1]
    # TODO: build actual AST
    return tokens


if __name__ == '__main__':
    # for t in build_ast('  expand 42/-7 + (+3^ 100500.228xy) [3]'):
    #     print(t)
    #
    # print(build_ast('expand [3]'))
    # # print(_t)

    _p = PatternCombinator([
        PatternTokenType([TokenType.LParen]),
        PatternUntil(PatternTokenType([TokenType.RParen]))
    ])

    _s = '4x * (2x - 1)'
    _ts = list(tokenize(_s))

    print(_p.match(_ts, 0))

    replace_all(_p,
                parens_replacer,
                _s,
                _ts)

    for _t in _ts:
        print(_t)
