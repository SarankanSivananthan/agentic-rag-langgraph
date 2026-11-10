"""Calculation tools: arithmetic and date differences.

Language models are unreliable at multi-digit arithmetic and at counting the
years between two dates, and the questions that need it ("How old was X
when ...?") are exactly the ones where an agent should reach for a tool
instead of guessing. Both functions are pure and raise `CalculatorError` with
a message the agent can read and recover from; they never execute arbitrary
code (expressions are parsed with `ast` and only a small whitelist of nodes
is evaluated).
"""
from __future__ import annotations

import ast
import operator
from datetime import date, datetime

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 100


class CalculatorError(ValueError):
    """The input could not be evaluated; the message says why."""


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError(f"exponent too large (limit is {MAX_EXPONENT})")
        try:
            return _BINARY_OPERATORS[type(node.op)](left, right)
        except ZeroDivisionError:
            raise CalculatorError("division by zero") from None
        except OverflowError:
            raise CalculatorError("result too large") from None
    raise CalculatorError("only numbers and + - * / // % ** with parentheses are allowed")


def _format_number(value: int | float) -> str:
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise CalculatorError("result is not a finite number")
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return format(value, ".10g")
    return str(value)


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression such as ``(1865 - 1809) / 2``."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(f"expression too long (limit is {MAX_EXPRESSION_LENGTH} characters)")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise CalculatorError(f"could not parse {expression!r} as an arithmetic expression") from None
    return _format_number(_evaluate(tree.body))


# --- dates ---------------------------------------------------------------

# Tried in order; the flag says whether the format pins down a full day.
# Incomplete dates ("1809", "February 1809") assume the 1st.
_DATE_FORMATS = (
    ("%Y-%m-%d", True), ("%B %d, %Y", True), ("%B %d %Y", True), ("%d %B %Y", True),
    ("%b %d, %Y", True), ("%d %b %Y", True),
    ("%Y-%m", False), ("%B %Y", False), ("%Y", False),
)


def _parse(text: str) -> tuple[date, bool]:
    """The date, and whether the text gave a complete day (not just a month or year)."""
    cleaned = " ".join(text.replace(",", ", ").split()).strip()
    for fmt, complete in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date(), complete
        except ValueError:
            continue
    raise CalculatorError(
        f"could not read {text!r} as a date; use e.g. '1809-02-12', 'February 12, 1809' or '1809'"
    )


def parse_date(text: str) -> date:
    return _parse(text)[0]


def date_difference(start: str, end: str, unit: str = "years") -> str:
    """Time elapsed from ``start`` to ``end``, in whole ``years`` or in ``days``.

    Whole years count like an age: a birthday not yet reached on the end date
    does not count. Dates without a month or day assume the 1st of the month
    or year, which can shift a whole-year result by one; the answer says so.
    """
    if unit not in ("years", "days"):
        raise CalculatorError("unit must be 'years' or 'days'")
    (start_date, start_complete), (end_date, end_complete) = _parse(start), _parse(end)
    if end_date < start_date:
        raise CalculatorError(f"{end!r} is before {start!r}; swap them to get a positive difference")

    if unit == "days":
        result = (end_date - start_date).days
    else:
        result = end_date.year - start_date.year - (
            (end_date.month, end_date.day) < (start_date.month, start_date.day)
        )

    note = "" if start_complete and end_complete else " (assuming the 1st for incomplete dates)"
    return f"{result} {unit}{note}"

