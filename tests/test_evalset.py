"""Checks on the evaluation set itself: a malformed question, or a quote that no
longer matches the corpus, fails here instead of silently skewing a result."""
import json
import re
from collections import Counter

from agentic_rag.calculator import calculate, date_difference, parse_date

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


# --- calculation ----------------------------------------------------------


def test_calculation_questions_need_both_tools_and_have_numeric_answers():
    calculations = by_category("calculation")
    assert len(calculations) == 24
    for q in calculations:
        assert q.expected_tools == ("retrieve", "calculator"), q.id
        assert q.answer_type == "number" and q.answer.isdigit(), q.id


def test_calculation_answers_follow_from_their_computation():
    for q in by_category("calculation"):
        tool, args = q.computation["tool"], q.computation["args"]
        result = date_difference(*args) if tool == "date_difference" else calculate(*args)
        assert result.split()[0] == q.answer, q.id


def test_calculation_inputs_are_stated_in_the_cited_evidence():
    for q in by_category("calculation"):
        evidence = " ".join(e.quote for e in q.evidence)
        if q.computation["tool"] == "date_difference":
            start, end, _ = q.computation["args"]
            assert start in evidence and end in evidence, q.id
        else:
            assert "32 and 62" in evidence, q.id  # the figures behind the wolf-weight arithmetic


def test_date_answers_match_an_independent_computation():
    for q in by_category("calculation"):
        if q.computation["tool"] != "date_difference":
            continue
        start, end, unit = q.computation["args"]
        a, b = parse_date(start), parse_date(end)
        expected = (b - a).days if unit == "days" else b.year - a.year - ((b.month, b.day) < (a.month, a.day))
        assert int(q.answer) == expected, q.id


def test_subtracting_the_years_is_not_enough_for_many_calculation_questions():
    # The reason these questions need a tool: a model that just subtracts the
    # two years is wrong whenever the birthday or anniversary is not reached.
    wrong_by_subtraction = [
        q for q in by_category("calculation")
        if q.computation["tool"] == "date_difference" and q.computation["args"][2] == "years"
        and parse_date(q.computation["args"][1]).year - parse_date(q.computation["args"][0]).year != int(q.answer)
    ]
    assert len(wrong_by_subtraction) >= 8


# --- multi-step -----------------------------------------------------------


def test_multi_step_questions_are_two_retrievals_in_two_different_passages():
    multi = by_category("multi_step")
    assert len(multi) == 24
    for q in multi:
        assert q.expected_tools == ("retrieve",), q.id
        assert len(q.evidence) == 2, q.id
        assert q.evidence[0].passage_id != q.evidence[1].passage_id, q.id


def test_the_first_step_names_the_entity_but_does_not_contain_the_answer():
    for q in by_category("multi_step"):
        first = q.evidence[0]
        assert q.chain["descriptor"] in first.quote, q.id
        assert q.chain["entity"].split()[-1] in PASSAGES[first.passage_id], q.id
        assert q.answer.lower() not in PASSAGES[first.passage_id].lower(), q.id


def test_the_second_step_has_the_answer_but_not_the_descriptor():
    # No lexical shortcut: the passage holding the answer never repeats the
    # question's own description of the entity ("thirteenth President").
    for q in by_category("multi_step"):
        second = q.evidence[1]
        assert q.answer.lower() in second.quote.lower(), q.id
        assert q.chain["descriptor"].lower() not in PASSAGES[second.passage_id].lower(), q.id


def test_multi_step_questions_cover_a_variety_of_entities():
    entities = {q.chain["entity"] for q in by_category("multi_step")}
    assert len(entities) >= 10


# --- unanswerable ---------------------------------------------------------


def test_unanswerable_questions_have_no_answer_and_no_evidence():
    unanswerable = by_category("unanswerable")
    assert len(unanswerable) == 20
    assert Counter(q.subtype for q in unanswerable) == {"absent_fact": 12, "out_of_domain": 8}
    for q in unanswerable:
        assert q.answer is None and not q.evidence, q.id
        assert q.expected_tools == ("retrieve",), q.id


def test_no_passage_matches_an_absence_pattern():
    # The corpus is re-scanned on every run, so a changed corpus cannot quietly
    # turn an unanswerable question into an answerable one.
    for q in by_category("unanswerable"):
        assert q.absence_patterns, q.id
        for pattern in q.absence_patterns:
            regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)
            matches = [pid for pid, text in PASSAGES.items() if regex.search(text)]
            assert not matches, f"{q.id}: {pattern!r} matches passages {matches[:5]}"


def test_absent_fact_questions_are_about_subjects_the_corpus_covers():
    for q in by_category("unanswerable"):
        if q.subtype == "absent_fact":
            assert PASSAGES.str.contains(q.topic, case=False, regex=False).any(), q.id
        else:
            assert q.topic is None, q.id


def test_the_full_set_has_the_planned_size_and_balance():
    assert Counter(q.category for q in QUESTIONS) == {
        "lookup": 30, "calculation": 24, "multi_step": 24, "unanswerable": 20,
    }
