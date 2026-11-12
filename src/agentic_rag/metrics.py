"""How answers and tool use are scored.

Answer metrics are the SQuAD-style pair used in rag-wikipedia, so numbers from
the agent and from the fixed pipeline are directly comparable with it:

  exact match   the normalised prediction equals the normalised gold answer
  token F1      word overlap, giving partial credit EM would score as zero

On top of that, agent evaluation needs three more things:

  number match      numeric answers are judged on the number, not the wording
                    ("51", "51 years old" and "He was 51" are all right)
  abstention        for unanswerable questions, did the system decline?
  tool use          from a trace, were the right tools called, and no needless ones?

Every function here is pure and has no model in it, so the scoring itself is
tested separately from the systems it scores.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

import numpy as np

from agentic_rag.evalset import Question

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation)


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_match(prediction: str, gold: str) -> float:
    """1.0 if the normalised gold answer appears in the normalised prediction, on word boundaries.

    An agent answers in sentences ("He was born in Summerhill, New York"), which
    exact match would score as wrong against "Summerhill, New York". This is the
    headline correctness criterion for text answers; EM and F1 are reported
    next to it. It can credit an answer that happens to mention the gold among
    other things, which is why a concise final answer matters.
    """
    gold_norm = normalize_answer(gold)
    if not gold_norm:
        return 0.0
    return float(f" {gold_norm} " in f" {normalize_answer(prediction)} ")


# --- numbers ---------------------------------------------------------------

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def extract_numbers(text: str) -> list[float]:
    """Every number in the text, with thousands separators removed ("1,865" -> 1865)."""
    return [float(match.replace(",", "")) for match in _NUMBER.findall(str(text))]


def number_match(prediction: str, gold: str) -> float:
    """1.0 if the gold number appears among the numbers in the prediction.

    Lenient on purpose: it does not penalise surrounding words or a supporting
    figure ("51, born in 1809"). It would credit an answer that lists many
    numbers, which a concise final answer does not do; that leniency is why
    exact match and F1 are reported alongside it.
    """
    gold_numbers = extract_numbers(gold)
    if len(gold_numbers) != 1:
        raise ValueError(f"gold answer {gold!r} is not a single number")
    return float(gold_numbers[0] in extract_numbers(prediction))


# --- abstention ------------------------------------------------------------

_ABSTENTION_MARKERS = (
    "i don't know", "i do not know", "i dont know",
    "cannot find", "can't find", "could not find", "couldn't find", "unable to find",
    "no information", "not mentioned", "not stated", "not specified", "not provided",
    "does not contain", "doesn't contain", "do not contain", "don't contain",
    "does not say", "doesn't say", "doesn't mention", "does not mention",
    "not in the knowledge base", "not in the context", "not in the provided",
    "not found in", "not available in", "unable to answer", "cannot answer", "can't answer",
    "insufficient information",
)


def is_abstention(answer: str) -> bool:
    """Does the answer decline to answer?"""
    text = str(answer).lower().replace("’", "'")
    return any(marker in text for marker in _ABSTENTION_MARKERS)


# --- tool use --------------------------------------------------------------


# The calculator is exposed to the model as two simple tools; for scoring they
# are one kind of tool, because a question either needs calculation or does not.
TOOL_FAMILY = {"retrieve": "retrieve", "calculator": "calculator", "date_difference": "calculator"}


@dataclass(frozen=True)
class ToolCall:
    name: str  # "retrieve", "calculator" or "date_difference"
    args: dict

    @property
    def family(self) -> str:
        return TOOL_FAMILY.get(self.name, self.name)


@dataclass(frozen=True)
class ToolScore:
    required_tools_used: bool  # every kind of tool the question needs was called at least once
    enough_retrievals: bool  # a two-step question needs at least two retrievals
    needless_tools: tuple[str, ...]  # tools called that the question does not need
    num_calls: int


def minimum_retrievals(question: Question) -> int:
    return 2 if question.category == "multi_step" else 1


def score_tool_use(question: Question, trace: list[ToolCall]) -> ToolScore:
    used = Counter(call.family for call in trace)
    expected = set(question.expected_tools)
    return ToolScore(
        required_tools_used=expected <= set(used),
        enough_retrievals=used["retrieve"] >= minimum_retrievals(question),
        needless_tools=tuple(sorted(set(used) - expected)),
        num_calls=len(trace),
    )


# --- one question ----------------------------------------------------------


@dataclass(frozen=True)
class AnswerScore:
    correct: float  # 1.0 / 0.0: right answer (gold contained / number matched), or abstaining on an unanswerable question
    exact_match: float | None  # None where it does not apply (unanswerable)
    f1: float | None
    abstained: bool


def score_answer(question: Question, prediction: str) -> AnswerScore:
    """Score one final answer. For unanswerable questions, abstaining is the correct answer."""
    abstained = is_abstention(prediction)
    if question.category == "unanswerable":
        return AnswerScore(correct=float(abstained), exact_match=None, f1=None, abstained=abstained)

    em, f1 = exact_match(prediction, question.answer), f1_score(prediction, question.answer)
    if question.answer_type == "number":
        correct = 0.0 if abstained else number_match(prediction, question.answer)
    else:
        correct = 0.0 if abstained else contains_match(prediction, question.answer)
    return AnswerScore(correct=correct, exact_match=em, f1=f1, abstained=abstained)


def bootstrap_ci(values, n_resamples: int = 2000, seed: int = 42) -> tuple[float, float]:
    """95% percentile-bootstrap confidence interval for the mean of `values`."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
