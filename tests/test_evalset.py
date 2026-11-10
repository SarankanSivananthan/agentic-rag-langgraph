"""Checks on the evaluation set itself: a malformed question, or a quote that no
longer matches the corpus, fails here instead of silently skewing a result."""
import json
from collections import Counter

import pytest

from agentic_rag.evalset import CATEGORIES, EVAL_DIR, TOOLS, Question, load_questions
from agentic_rag.retriever import load_passages

QUESTIONS = load_questions()
PASSAGES = load_passages()["passage"]


def by_category(category):
    return [q for q in QUESTIONS if q.category == category]


def test_ids_are_unique():
    ids = [q.id for q in QUESTIONS]
    assert not [i for i, n in Counter(ids).items() if n > 1]


def test_every_question_is_well_formed():
    for q in QUESTIONS:
        assert q.question.strip().endswith("?"), q.id
        assert q.category in CATEGORIES, q.id
        assert q.id.startswith(q.category.replace("_", "-")), q.id
        assert q.expected_tools and set(q.expected_tools) <= set(TOOLS), q.id
        assert q.source, q.id


def test_every_answerable_question_cites_evidence_that_is_in_the_corpus():
    for q in QUESTIONS:
        if q.category == "unanswerable":
            continue
        assert q.evidence, f"{q.id} has no evidence"
        for e in q.evidence:
            assert e.quote in PASSAGES[e.passage_id], f"{q.id}: quote is not verbatim in passage {e.passage_id}"


def test_lookup_answers_appear_in_their_evidence():
    lookups = by_category("lookup")
    assert len(lookups) == 30
    for q in lookups:
        assert q.expected_tools == ("retrieve",), q.id
        assert q.answer.lower() in " ".join(e.quote for e in q.evidence).lower(), q.id


def test_lookup_questions_are_distinct():
    questions = [q.question.lower() for q in by_category("lookup")]
    assert len(set(questions)) == len(questions)


def test_review_log_accounts_for_every_lookup_question():
    import csv

    with open(EVAL_DIR / "lookup_review.csv") as f:
        review = list(csv.DictReader(f))
    kept = {row["source_id"] for row in review if row["decision"] == "keep"}
    cited = {q.source.rsplit(" ", 1)[1] for q in by_category("lookup")}
    assert kept == cited
    assert all(row["note"] for row in review if row["decision"] == "reject")


def test_loader_rejects_unknown_tools_and_categories(tmp_path):
    bad_tool = {"id": "lookup-1", "category": "lookup", "question": "Q?", "answer": "A",
                "expected_tools": ["web_search"], "source": "x"}
    (tmp_path / "bad.jsonl").write_text(json.dumps(bad_tool) + "\n")
    with pytest.raises(ValueError, match="unknown tools"):
        load_questions(tmp_path)

    bad_category = dict(bad_tool, expected_tools=["retrieve"], category="trivia")
    (tmp_path / "bad.jsonl").write_text(json.dumps(bad_category) + "\n")
    with pytest.raises(ValueError, match="unknown category"):
        load_questions(tmp_path)


def test_loader_can_filter_by_category():
    assert {q.category for q in load_questions(categories=("lookup",))} == {"lookup"}
    assert isinstance(QUESTIONS[0], Question)
