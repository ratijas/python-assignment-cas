# Assignment 6 'Computer Algebra Systems'

## Short description

Basically, this is a lightweight clone of `sympy`, supporting basic math operations and some terms folding/factoring functionality. More over, some forks (to my knowledge) after [this](https://github.com/ratijas/python-assignment-cas/commit/4092483de0ca2fcdec38c7a10993543b1ce66239) commit implemented the most interesting part of the system using `eval()` with `sympy` symbols in local context.

This CAS does not aim to be feature-complete. Instead it is more of a PoC and university assignment employing lexer, parser, expression tree and symbolic algebra.

## Syntax

Upon running as a module, program enters REPL where it reads expressions, evaluates and simplifies them, optionally expands parens, and prints back the results.

Supported operators (ordered by descending priority) are:

 - `()` parens
 - `^` power
 - `*` multiplication, `/` division
 - `+` add, `-` subtract
 - `expand` as a prefix to the whole expression

Supported operands are:

 - literals:
   + Python int as `[+-]?digits`
   + Python float as `[+-]?digits.digits`
   + [Fraction](https://docs.python.org/3/library/fractions.html) as `int/int` without spaces around `/`
 - reference to the previous result as `[N]`, `[-N]` or special syntax `[last]`
 - symbols: any single letter
 - compounds: sequence of multiple symbols optionally prefixed by a literal. For example: `4x`, `2x^3`, `x y z`.
 
## How to run

`$ cd src && python -m cas`

### Run all tests

`$ cd src && python -m cas.tests`

## Credits

 Done by ivan tkachenko ([@ratijas](https://t.me/ratijas)), Innopolis student on BS16-SE track.

Except as otherwise noted (below and/or in individual files), code is
licensed under the Apache License, Version 2.0 <LICENSE-APACHE> or
<http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
<LICENSE-MIT> or <http://opensource.org/licenses/MIT>, at your option.
