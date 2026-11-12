"""Run a system over the evaluation questions and keep one record per question.

A "system" is any function from a question string to a `RunResult` — the
baseline and the agent both fit — so they go through exactly the same loop and
the same scoring. Every question is recorded with its answer, its score, its
tool trace and its cost, so results can be audited line by line, not just read
off a summary.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from agentic_rag.evalset import Question
from agentic_rag.metrics import is_grounded, score_answer, score_tool_use
from agentic_rag.runs import RunResult

System = Callable[[str], RunResult]


@dataclass
class QuestionRecord:
    id: str
    category: str
    subtype: str | None
    question: str
    gold: str | None
    prediction: str
    correct: float
    exact_match: float | None
    f1: float | None
    abstained: bool
    required_tools_used: bool
    enough_retrievals: bool
    needless_tools: list[str]
    trace: list[dict]
    latency_s: float
    llm_calls: int
    input_tokens: int
    output_tokens: int
    error: str | None
    step_limit_hit: bool
    grounded: bool | None = None  # the facts the answer depends on were in the retrieved text (None: not applicable)

    def faithful_score(self) -> float:
        """1.0 if correct AND grounded in what was retrieved; for unanswerable questions, a correct abstention."""
        if self.category == "unanswerable":
            return self.correct
        return float(self.correct >= 0.5 and bool(self.grounded))


def evaluate_question(system: System, question: Question) -> QuestionRecord:
    result = system(question.question)
    answer = score_answer(question, result.answer)
    tools = score_tool_use(question, result.trace)
    return QuestionRecord(
        id=question.id, category=question.category, subtype=question.subtype,
        question=question.question, gold=question.answer, prediction=result.answer,
        correct=answer.correct, exact_match=answer.exact_match, f1=answer.f1, abstained=answer.abstained,
        required_tools_used=tools.required_tools_used, enough_retrievals=tools.enough_retrievals,
        needless_tools=list(tools.needless_tools),
        trace=[{"name": call.name, "args": call.args, "output": call.output} for call in result.trace],
        latency_s=result.latency_s, llm_calls=result.llm_calls,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        error=result.error, step_limit_hit=result.step_limit_hit,
        grounded=is_grounded(question, result.trace),
    )


def evaluate(
    system: System, questions: list[Question], on_record: Callable[[QuestionRecord], None] | None = None
) -> list[QuestionRecord]:
    """Score `system` on every question, in order. `on_record` sees each record as soon as it exists."""
    records = []
    for question in questions:
        record = evaluate_question(system, question)
        records.append(record)
        if on_record:
            on_record(record)
    return records


def write_records(records: list[QuestionRecord], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_records(path: Path) -> list[QuestionRecord]:
    with open(path) as f:
        return [QuestionRecord(**json.loads(line)) for line in f if line.strip()]
