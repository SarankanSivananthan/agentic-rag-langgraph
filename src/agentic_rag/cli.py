"""Command-line entry point: run the baseline or the agent on the evaluation set.

    python scripts/run_eval.py --system baseline --provider openai --model <model>
    python scripts/run_eval.py --system agent    --provider openai --model <model>

Each question's record is written to disk the moment it is scored, and an
interrupted run resumes where it stopped (`--resume`), because a full run makes
hundreds of model calls and a rate limit or a dropped connection should not
cost all of them.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from agentic_rag.agent import DEFAULT_MAX_STEPS, build_agent, run_agent
from agentic_rag.baseline import run_baseline
from agentic_rag.evaluate import QuestionRecord, System, evaluate_question, read_records, write_records
from agentic_rag.evalset import CATEGORIES, Question, load_questions
from agentic_rag.llm import PROVIDERS, get_chat_model
from agentic_rag.report import summarize, summary_markdown
from agentic_rag.retriever import Retriever
from agentic_rag.tools import make_tools

SYSTEMS = ("baseline", "agent")
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def build_system(name: str, llm: BaseChatModel, retriever: Retriever, max_steps: int = DEFAULT_MAX_STEPS) -> System:
    if name == "baseline":
        return lambda question: run_baseline(llm, retriever, question)
    if name == "agent":
        graph = build_agent(llm, make_tools(retriever), max_steps=max_steps)
        return lambda question: run_agent(graph, question, max_steps=max_steps)
    raise ValueError(f"unknown system {name!r}; expected one of {SYSTEMS}")


def select_questions(
    questions: list[Question], categories: tuple[str, ...] | None = None, limit_per_category: int | None = None
) -> list[Question]:
    """Filter by category and/or keep the first N of each, for a cheap smoke test."""
    selected, taken = [], {}
    for question in questions:
        if categories and question.category not in categories:
            continue
        if limit_per_category is not None and taken.get(question.category, 0) >= limit_per_category:
            continue
        taken[question.category] = taken.get(question.category, 0) + 1
        selected.append(question)
    return selected


def default_output_path(system: str, provider: str, model: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
    return RESULTS_DIR / f"{system}_{provider}_{safe_model}.jsonl"


def run_system(
    system: System, questions: list[Question], out_path: Path | None = None, resume: bool = False,
    progress: Callable[[str], None] = print,
) -> list[QuestionRecord]:
    """Score `system` on `questions`, saving each record as soon as it exists."""
    done: dict[str, QuestionRecord] = {}
    if out_path and resume and Path(out_path).exists():
        done = {record.id: record for record in read_records(out_path)}
    elif out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("")

    for number, question in enumerate(questions, start=1):
        if question.id in done:
            progress(f"[{number}/{len(questions)}] {question.id} already done, skipped")
            continue
        record = evaluate_question(system, question)
        done[question.id] = record
        if out_path:
            with open(out_path, "a") as f:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        flag = "ERROR " + record.error if record.error else f"correct={record.correct:.0f}"
        progress(f"[{number}/{len(questions)}] {question.id} {flag} ({record.latency_s:.1f}s)")

    records = [done[q.id] for q in questions]
    if out_path:
        write_records(records, out_path)  # same records, in question order
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the baseline or the agent on the question set.")
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--model", default=None, help="model name (defaults to a placeholder, see llm.py)")
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=None)
    parser.add_argument("--limit-per-category", type=int, default=None, help="smoke test: only the first N of each category")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--resume", action="store_true", help="skip questions already in the output file")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    llm = get_chat_model(args.provider, args.model)
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "model"
    out_path = args.out or default_output_path(args.system, args.provider, model_name)

    questions = select_questions(load_questions(), tuple(args.categories) if args.categories else None, args.limit_per_category)
    system = build_system(args.system, llm, Retriever.from_passages(), args.max_steps)
    print(f"{args.system} / {args.provider} / {model_name}: {len(questions)} questions -> {out_path}")

    records = run_system(system, questions, out_path, resume=args.resume)
    summary = summary_markdown(summarize(records))
    out_path.with_suffix(".md").write_text(summary)
    print("\n" + summary)
