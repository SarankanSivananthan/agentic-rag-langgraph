"""Builds the 'calculation' category: questions that need the calculator on top
of retrieval — an age, or a number of days or years between two dates stated
in the corpus.

These are the cases where a language model most often slips: counting whole
years correctly depends on whether a birthday has been reached, which is easy
to get wrong without a tool. Every input date comes from a cited passage, and
each gold answer is computed by the calculator tool AND cross-checked here with
an independent datetime implementation; a disagreement stops the build.

Questions whose answer would depend on assuming the 1st for an incomplete date
(e.g. "1745") are deliberately left out: the whole-year result there is
ambiguous, which would test the question, not the system.

Run from the project root: python scripts/build_calculation_questions.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_rag.calculator import calculate, date_difference, parse_date  # noqa: E402

# (question, [(passage id, start date text), (passage id, end date text)], unit)
# The date texts are exactly as written in the passages.
DATE_QUESTIONS = [
    # age at death
    ("How old was Michael Faraday when he died?", [(74, "September 22, 1791"), (74, "August 25, 1867")], "years"),
    ("How old was Anders Celsius when he died?", [(141, "November 27, 1701"), (141, "April 25, 1744")], "years"),
    ("How old was Blaise Pascal when he died?", [(205, "June 19 1623"), (205, "August 19 1662")], "years"),
    ("How old was Amedeo Avogadro when he died?", [(725, "August 9, 1776"), (725, "July 9, 1856")], "years"),
    ("How old was Woodrow Wilson when he died?", [(2129, "December 28, 1856"), (2129, "February 3, 1924")], "years"),
    ("How old was Alessandro Volta when he died?", [(2250, "February 18, 1745"), (2250, "March 5, 1827")], "years"),
    ("How old was Abraham Lincoln when he died?", [(278, "February 12, 1809"), (278, "April 15, 1865")], "years"),
    # age at an event described in another passage
    ("How old was Abraham Lincoln when he was elected President?", [(278, "February 12, 1809"), (319, "November 6, 1860")], "years"),
    ("How old was Abraham Lincoln when he married Mary Todd?", [(278, "February 12, 1809"), (296, "November 4 1842")], "years"),
    ("How old was Millard Fillmore when he married Abigail Powers?", [(148, "January 7, 1800"), (151, "February 26, 1826")], "years"),
    ("How old was Millard Fillmore when he married Caroline McIntosh?", [(148, "January 7, 1800"), (185, "February 10, 1858")], "years"),
    ("How old was Grover Cleveland when he married Frances Folsom?", [(473, "March 18 1837"), (496, "June 2, 1886")], "years"),
    ("How old was Ulysses S. Grant when he married Julia Dent?", [(1768, "April 27, 1822"), (1775, "August 22, 1848")], "years"),
    ("How old was Gerald Ford when he became President?", [(1992, "July 14, 1913"), (2041, "August 9, 1974")], "years"),
    ("How old was Gerald Ford when he married Elizabeth Bloomer Warren?", [(1992, "July 14, 1913"), (2022, "October 15, 1948")], "years"),
    ("How old was James Monroe when he married Elizabeth Kortright?", [(2506, "April 28, 1758"), (2507, "February 16, 1786")], "years"),
    ("How old was Michael Faraday when he married Sarah Barnard?", [(74, "September 22, 1791"), (85, "June 2, 1821")], "years"),
    ("How old was Calvin Coolidge when President Harding died?", [(399, "July 4 1872"), (433, "August 2 1923")], "years"),
    ("How old was Calvin Coolidge when he was inaugurated as Governor of Massachusetts?", [(399, "July 4 1872"), (424, "January 1 1919")], "years"),
    # durations
    ("For how many years was Abraham Lincoln President?", [(278, "March 4, 1861"), (278, "April 15, 1865")], "years"),
    ("For how many days did James Monroe hold both the Secretary of State and Secretary of War posts?", [(2512, "October 1, 1814"), (2512, "February 28, 1815")], "days"),
    ("How many days passed between Abraham Lincoln's election and his first inauguration?", [(319, "November 6, 1860"), (321, "March 4, 1861")], "days"),
    ("How many years had Abraham Lincoln been married to Mary Todd when he died?", [(296, "November 4 1842"), (278, "April 15, 1865")], "years"),
]

# Plain arithmetic on figures stated in the corpus: (question, passage, quote, expression)
ARITHMETIC_QUESTIONS = [
    ("By how many kilograms do the heaviest and lightest typical wolf weights differ?", 1021,
     "between 32 and 62 kilograms", "62 - 32"),
]


def sentence_with(text: str, needle: str) -> str:
    start = text.index(needle)
    previous_end = text.rfind(". ", 0, start)
    left = 0 if previous_end < 0 else previous_end + 2
    right = text.find(". ", start + len(needle))
    return text[left: right + 1 if right >= 0 else len(text)].strip()


def independent_difference(start: str, end: str, unit: str) -> int:
    """A second, deliberately separate implementation, for cross-checking."""
    a, b = parse_date(start), parse_date(end)
    if unit == "days":
        return (b - a).days
    years = b.year - a.year
    if (b.month, b.day) < (a.month, a.day):
        years -= 1
    return years


def main():
    passages = pd.read_parquet("data/passages.parquet")["passage"]
    rows = []

    for question, facts, unit in DATE_QUESTIONS:
        (pid_start, start), (pid_end, end) = facts
        evidence = []
        for pid, date_text in facts:
            quote = sentence_with(passages[pid], date_text)
            if not any(e["passage_id"] == pid and e["quote"] == quote for e in evidence):
                evidence.append({"passage_id": pid, "quote": quote})

        tool_result = date_difference(start, end, unit)  # e.g. "51 years"
        assert "assuming" not in tool_result, (question, "incomplete date")
        answer = tool_result.split()[0]
        assert int(answer) == independent_difference(start, end, unit), (question, tool_result)
        rows.append({
            "question": question, "answer": answer, "answer_type": "number",
            "expected_tools": ["retrieve", "calculator"], "evidence": evidence,
            "computation": {"tool": "date_difference", "args": [start, end, unit]},
        })

    for question, pid, needle, expression in ARITHMETIC_QUESTIONS:
        rows.append({
            "question": question, "answer": calculate(expression), "answer_type": "number",
            "expected_tools": ["retrieve", "calculator"],
            "evidence": [{"passage_id": pid, "quote": sentence_with(passages[pid], needle)}],
            "computation": {"tool": "calculate", "args": [expression]},
        })

    with open("data/eval/calculation.jsonl", "w") as f:
        for i, row in enumerate(rows, start=1):
            record = {"id": f"calculation-{i:03d}", "category": "calculation", **row,
                      "source": "written by hand from the cited passages"}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(rows)} calculation questions written")
    for i, r in enumerate(rows, start=1):
        print(f"  {i:2d}. {r['answer']:>4}  {r['question']}")


if __name__ == "__main__":
    main()
