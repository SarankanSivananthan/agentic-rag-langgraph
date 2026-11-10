import pytest

from agentic_rag.evalset import load_questions
from agentic_rag.metrics import (
    ToolCall,
    bootstrap_ci,
    contains_match,
    exact_match,
    extract_numbers,
    f1_score,
    is_abstention,
    normalize_answer,
    number_match,
    score_answer,
    score_tool_use,
)

QUESTIONS = {q.id: q for q in load_questions()}
LOOKUP = QUESTIONS["lookup-002"]  # Where was Millard Fillmore born? -> Summerhill, New York
CALCULATION = QUESTIONS["calculation-008"]  # How old was Lincoln when elected President? -> 51
MULTI_STEP = QUESTIONS["multi-step-001"]
UNANSWERABLE = QUESTIONS["unanswerable-001"]


def test_normalisation_ignores_case_punctuation_and_articles():
    assert normalize_answer("The Federalist Party.") == "federalist party"
    assert exact_match("The Federalist Party.", "federalist party") == 1
    assert exact_match("yes", "Yes.") == 1
    assert exact_match("diets", "different diets") == 0


def test_f1_gives_partial_credit():
    assert f1_score("diets", "different diets") == pytest.approx(2 / 3)
    assert f1_score("", "x") == 0 and f1_score("", "") == 1
    assert f1_score("completely different", "unrelated words") == 0


@pytest.mark.parametrize("prediction, gold, expected", [
    ("He was born in Summerhill, New York.", "Summerhill, New York", 1),
    ("summerhill new york", "Summerhill, New York", 1),
    ("Buffalo, New York", "Summerhill, New York", 0),
    ("The elephant is large", "ant", 0),  # word boundaries, not substrings
    ("18140 people", "1814", 0),
    ("", "anything", 0),
])
def test_contains_match(prediction, gold, expected):
    assert contains_match(prediction, gold) == expected


@pytest.mark.parametrize("text, numbers", [
    ("51", [51.0]),
    ("He was 51 years old.", [51.0]),
    ("1,865,908 votes", [1865908.0]),
    ("about 3.5 meters", [3.5]),
    ("between 17 and 20", [17.0, 20.0]),
    ("no numbers here", []),
    ("x2 and v3", []),  # digits glued to letters are not standalone numbers
])
def test_extract_numbers(text, numbers):
    assert extract_numbers(text) == numbers


@pytest.mark.parametrize("prediction, gold, expected", [
    ("51", "51", 1),
    ("He was 51 years old when he was elected.", "51", 1),
    ("Aged 51 (born in 1809)", "51", 1),  # a supporting figure is tolerated
    ("He was 15.", "51", 0),
    ("He was 151.", "51", 0),  # not a substring match
    ("I could not tell.", "51", 0),
])
def test_number_match(prediction, gold, expected):
    assert number_match(prediction, gold) == expected


def test_number_match_needs_a_single_number_as_gold():
    with pytest.raises(ValueError):
        number_match("51", "between 17 and 20")


@pytest.mark.parametrize("answer, expected", [
    ("I don't know.", True),
    ("I don’t know", True),
    ("I could not find this in the knowledge base.", True),
    ("The context does not mention it.", True),
    ("There is no information about that.", True),
    ("Insufficient information to answer.", True),
    ("Montevideo", False),
    ("Lincoln was born in Hardin County, Kentucky.", False),
    ("", False),
])
def test_is_abstention(answer, expected):
    assert is_abstention(answer) == expected


def test_score_a_text_answer():
    right = score_answer(LOOKUP, "Fillmore was born in Summerhill, New York.")
    assert (right.correct, right.abstained) == (1.0, False)
    assert right.exact_match == 0 and 0 < right.f1 < 1  # EM would have called this wrong
    assert score_answer(LOOKUP, "Buffalo").correct == 0


def test_score_a_numeric_answer_on_the_number():
    assert score_answer(CALCULATION, "Lincoln was 51 years old.").correct == 1
    assert score_answer(CALCULATION, "Lincoln was 52 years old.").correct == 0


def test_abstaining_on_an_answerable_question_is_wrong():
    score = score_answer(LOOKUP, "I don't know.")
    assert (score.correct, score.abstained) == (0.0, True)


def test_unanswerable_questions_reward_abstention_and_punish_an_answer():
    assert score_answer(UNANSWERABLE, "The knowledge base does not contain this.").correct == 1
    answered = score_answer(UNANSWERABLE, "The capital of Ghana is Accra.")
    assert answered.correct == 0
    assert answered.exact_match is None and answered.f1 is None


def test_tool_use_for_each_kind_of_question():
    retrieve = ToolCall("retrieve", {"query": "q"})
    calculator = ToolCall("calculator", {"expression": "1 + 1"})

    ok = score_tool_use(LOOKUP, [retrieve])
    assert ok.required_tools_used and ok.enough_retrievals and ok.needless_tools == ()

    needless = score_tool_use(LOOKUP, [retrieve, calculator])
    assert needless.needless_tools == ("calculator",) and needless.num_calls == 2

    assert not score_tool_use(CALCULATION, [retrieve]).required_tools_used
    assert score_tool_use(CALCULATION, [retrieve, calculator]).required_tools_used

    assert not score_tool_use(MULTI_STEP, [retrieve]).enough_retrievals
    assert score_tool_use(MULTI_STEP, [retrieve, retrieve]).enough_retrievals

    assert not score_tool_use(UNANSWERABLE, []).required_tools_used


def test_bootstrap_interval_brackets_the_mean_and_is_reproducible():
    values = [1] * 60 + [0] * 40
    low, high = bootstrap_ci(values)
    assert low < 0.6 < high and (low, high) == bootstrap_ci(values)
    assert 0.4 < low and high < 0.8
