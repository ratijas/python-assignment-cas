import atexit
import readline

from .exception import *
from .expression import *
from .history import *
from .parser import *
from .passes import *


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
        return evaluate(self.history, expression)


def main():
    repl = Repl()
    repl.block()


if __name__ == '__main__':
    main()
