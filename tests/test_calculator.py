import pytest

from agentic_rag.calculator import CalculatorError, calculate, date_difference, parse_date


@pytest.mark.parametrize("expression, expected", [
    ("1865 - 1809", "56"),
    ("2 + 3 * 4", "14"),
    ("(2 + 3) * 4", "20"),
    ("-5 + 2", "-3"),
    ("2 ** 10", "1024"),
    ("17 // 5", "3"),
    ("17 % 5", "2"),
    ("7 / 2", "3.5"),
    ("10 / 4 * 2", "5"),
    ("1 / 3", "0.3333333333"),
    ("  12 * 12  ", "144"),
])
def test_arithmetic(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo hi')",
    "abs(-3)",
    "x + 1",
    "'a' * 3",
    "(1).real",
    "[1, 2][0]",
    "lambda: 1",
    "1 if True else 2",
    "",
    "2 +",
])
def test_anything_beyond_plain_arithmetic_is_rejected(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)


def test_division_by_zero_is_reported():
    with pytest.raises(CalculatorError, match="division by zero"):
        calculate("5 / 0")


def test_huge_exponents_are_refused_instead_of_hanging():
    with pytest.raises(CalculatorError, match="exponent too large"):
        calculate("9 ** 9 ** 9")
    with pytest.raises(CalculatorError, match="too long"):
        calculate("1+" * 200 + "1")


@pytest.mark.parametrize("text, expected", [
    ("1809-02-12", (1809, 2, 12)),
    ("February 12, 1809", (1809, 2, 12)),
    ("February 12 1809", (1809, 2, 12)),
    ("12 February 1809", (1809, 2, 12)),
    ("Feb 12, 1809", (1809, 2, 12)),
    ("February 1809", (1809, 2, 1)),
    ("1809", (1809, 1, 1)),
])
def test_dates_are_read_in_the_formats_wikipedia_uses(text, expected):
    parsed = parse_date(text)
    assert (parsed.year, parsed.month, parsed.day) == expected


def test_unreadable_dates_are_reported():
    with pytest.raises(CalculatorError, match="could not read"):
        parse_date("sometime in spring")


def test_whole_years_count_like_an_age():
    # Lincoln: born February 12, 1809, elected November 6, 1860, died April 15, 1865.
    assert date_difference("February 12, 1809", "November 6, 1860") == "51 years"
    assert date_difference("February 12, 1809", "April 15, 1865") == "56 years"
    # Birthday not yet reached on the end date: one year fewer.
    assert date_difference("February 12, 1809", "January 1, 1860") == "50 years"
    # Exactly on the birthday it counts.
    assert date_difference("February 12, 1809", "February 12, 1860") == "51 years"


def test_days_between_dates():
    assert date_difference("1865-04-14", "1865-04-15", unit="days") == "1 days"
    assert date_difference("2000-01-01", "2001-01-01", unit="days") == "366 days"  # 2000 is a leap year


def test_incomplete_dates_are_flagged_in_the_answer():
    assert date_difference("1809", "1865") == "56 years (assuming the 1st for incomplete dates)"
    assert "assuming" not in date_difference("1809-02-12", "1865-04-15")


def test_reversed_dates_and_bad_units_are_reported():
    with pytest.raises(CalculatorError, match="before"):
        date_difference("1865", "1809")
    with pytest.raises(CalculatorError, match="unit"):
        date_difference("1809", "1865", unit="months")
