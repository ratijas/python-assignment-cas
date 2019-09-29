import unittest

from cas.exception import *
from cas.expression import *
from cas.parser import *


class TokenizerTestCase(unittest.TestCase):

    def assertRaisesParseError(self, tokenizer):
        """Parse Error shortcut mixin"""
        self.assertRaises(ParseError, list, tokenizer)

    def test_tokens_equality(self):
        self.assertEqual(Token(TokenType.LBracket, 0, 0, '[', None),
                         Token(TokenType.LBracket, 0, 0, '[', None))
        self.assertNotEqual(Token(TokenType.Literal, 0, 1, '42', Literal(42)),
                            Token(TokenType.Literal, 0, 3, '42.0', Literal(42.0)))
        self.assertEqual(Symbol('x'), Symbol('x'))
        self.assertEqual(BinaryOperation.Add, BinaryOperation.Add)

    def test_literals(self):
        self.assertRaisesParseError(tokenize('6.'))
        self.assertRaisesParseError(tokenize('12/'))
        self.assertRaisesParseError(tokenize('1.2.3'))

    def test_history(self):
        self.assertSequenceEqual(list(tokenize('[42]')), [
            Token(TokenType.LBracket, 0, 0, '[', None),
            Token(TokenType.Literal, 1, 2, '42', Literal(42)),
            Token(TokenType.RBracket, 3, 3, ']', None),
        ])

    def test_parent(self):
        self.assertSequenceEqual(list(tokenize('(2 + 3)')), [
            Token(TokenType.LParen, 0, 0, '(', None),
            Token(TokenType.Literal, 1, 1, '2', Literal(2)),
            Token(TokenType.Operator, 3, 3, '+', BinaryOperation.Add),
            Token(TokenType.Literal, 5, 5, '3', Literal(3)),
            Token(TokenType.RParen, 6, 6, ')', None),
        ])

    def test_compound(self):
        self.assertSequenceEqual(list(tokenize('2xy^3z')), [
            Token(TokenType.Literal, start=0, end=0, raw='2', value=Literal(2)),
            Token(TokenType.Symbol, start=1, end=1, raw='x', value=Symbol('x')),
            Token(TokenType.Symbol, start=2, end=2, raw='y', value=Symbol('y')),
            Token(TokenType.Operator, start=3, end=3, raw='^', value=BinaryOperation.Pow),
            Token(TokenType.Literal, start=4, end=4, raw='3', value=Literal(3)),
            Token(TokenType.Symbol, start=5, end=5, raw='z', value=Symbol('z')),
        ])


class ReplaceTestCase(unittest.TestCase):

    def test_diff_size(self):
        # remove first node
        self.assertEqual(Replace(0, 0, []).diff, -1)

        # replace first node with another node
        replace = Replace(0, 0, [AstRaw.from_token(Token(TokenType.Literal, 0, 0, '42', Literal(42)))])
        self.assertEqual(replace.diff, 0)

        # 0 1 2 3 4 5
        #     ^^^^^^^
        # 4 nodes in total
        self.assertEqual(Replace(2, 5, []).diff, -4)

    def test_apply(self):
        nodes = [
            AstRaw.from_token(Token(TokenType.LParen, 0, 0, '(', None)),
            AstRaw.from_token(Token(TokenType.LBracket, 1, 1, '[', None)),
            AstRaw.from_token(Token(TokenType.Literal, 2, 3, '42', Literal(42))),
            AstRaw.from_token(Token(TokenType.RBracket, 4, 4, ']', None)),
            AstRaw.from_token(Token(TokenType.RParen, 5, 5, ')', None)),
        ]
        history = AstAtom(HistoryRef(42), 1, 4, '[42]')
        replace = Replace(1, 3, [history])
        replace.apply(nodes)
        self.assertEqual(len(nodes), 3)
        self.assertSequenceEqual(nodes, [
            AstRaw.from_token(Token(TokenType.LParen, 0, 0, '(', None)),
            history,
            AstRaw.from_token(Token(TokenType.RParen, 5, 5, ')', None)),
        ])


if __name__ == '__main__':
    unittest.main()
