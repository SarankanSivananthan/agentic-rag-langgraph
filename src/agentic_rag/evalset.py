"""The hand-built evaluation set.

One JSON line per question, grouped in one file per category under
`data/eval/`. Every question records the tools a correct solution uses, so
tool choice can be scored separately from answer quality, and the evidence
(passage id plus a verbatim quote) that its answer rests on, so the set can
be checked against the corpus mechanically.

Categories:
  lookup        one retrieval is enough
  calculation   needs the calculator on top of a retrieval
  multi_step    needs several retrievals chained together
  unanswerable  the corpus does not contain the answer: abstaining is correct
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "data" / "eval"

CATEGORIES = ("lookup", "calculation", "multi_step", "unanswerable")
TOOLS = ("retrieve", "calculator")
ANSWER_TYPES = ("text", "number")


@dataclass(frozen=True)
class Evidence:
    passage_id: int
    quote: str  # verbatim substring of the passage


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    answer: str | None  # None for unanswerable questions
    expected_tools: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    source: str
    computation: dict | None = None  # how a calculation question's answer follows from its evidence
    answer_type: str = "text"  # "number": judged on the number in the answer, not on its wording


def _parse(record: dict) -> Question:
    if record["category"] not in CATEGORIES:
        raise ValueError(f"{record['id']}: unknown category {record['category']!r}")
    unknown = set(record["expected_tools"]) - set(TOOLS)
    if unknown:
        raise ValueError(f"{record['id']}: unknown tools {sorted(unknown)}")
    answer_type = record.get("answer_type", "text")
    if answer_type not in ANSWER_TYPES:
        raise ValueError(f"{record['id']}: unknown answer type {answer_type!r}")
    return Question(
        id=record["id"],
        category=record["category"],
        question=record["question"],
        answer=record["answer"],
        expected_tools=tuple(record["expected_tools"]),
        evidence=tuple(Evidence(e["passage_id"], e["quote"]) for e in record.get("evidence", [])),
        source=record["source"],
        computation=record.get("computation"),
        answer_type=answer_type,
    )


def load_questions(eval_dir: Path = EVAL_DIR, categories: tuple[str, ...] | None = None) -> list[Question]:
    questions = []
    for path in sorted(Path(eval_dir).glob("*.jsonl")):
        with open(path) as f:
            questions += [_parse(json.loads(line)) for line in f if line.strip()]
    if categories is not None:
        questions = [q for q in questions if q.category in categories]
    return questions
