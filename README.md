# Agentic RAG with LangGraph

An agent that decides *how* to answer a question — look it up in a Wikipedia knowledge base, compute something, chain two lookups, or admit the answer isn't there — built on top of the retrieval pipeline from [rag-wikipedia](https://github.com/SarankanSivananthan/rag-wikipedia), and evaluated the same way: measured, with intervals, against a plain fixed-pipeline baseline on the same questions.

> **Status: in progress.** This README is updated as each piece lands; nothing below is claimed until it is in the repo with its results.

## Goal

A tool-using agent is easy to demo and hard to evaluate. The question this project asks is the unglamorous one: **does the agent actually beat the fixed retrieve-then-generate pipeline, and where?** A plausible outcome is "no gain on simple lookups, a real gain on multi-step and arithmetic questions" — the evaluation is there to find out, not to confirm.

## Plan

- [x] Retriever over the Wikipedia passages (same chunking and embeddings as rag-wikipedia; `src/agentic_rag/retriever.py`)
- [x] Calculator tools (arithmetic and date differences; `src/agentic_rag/calculator.py`)
- [ ] Hand-built evaluation set, labelled with the tools each question needs (`data/eval/`, loader in `src/agentic_rag/evalset.py`):
  - [x] lookup (30)
  - [x] calculation (24)
  - [x] multi-step (24)
  - [x] unanswerable (20)
- [ ] Fixed-pipeline baseline (plain RAG) scored on that set
- [ ] LangGraph agent: routing, retrieval, relevance check with query rewriting, step limit
- [ ] Evaluation: tool-choice accuracy, answer exact match / F1, correct abstention, steps, latency, cost — each with 95% intervals
- [ ] Results and honest limitations in this README

## Evaluation design

The evaluation set is small and written by hand, so its intervals will be wide; that is stated next to every number rather than hidden. Every question records which tools a correct solution uses, so tool choice can be scored separately from answer quality. Agent and baseline are scored on identical questions with the same metrics (SQuAD-style exact match and token F1, as in rag-wikipedia).

## Data

`data/passages.parquet` is the knowledge base: the 3,200 Wikipedia passages of [`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) (CC-BY 3.0), the same file used by rag-wikipedia. The retriever's index is built from it on first use (a couple of minutes on CPU) and cached in `data/faiss_index/`, which is not versioned.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest
```

The retriever reproduces rag-wikipedia's: on the 498 questions whose answer can be checked by string match, the answer text appears in the top-4 chunks for 46.6% of them, the figure measured there for 250-character chunks.

## The evaluation set

Questions are stored one JSON line each in `data/eval/<category>.jsonl`, with the tools a correct solution needs and the evidence (passage id plus a verbatim quote) the answer rests on. `tests/test_evalset.py` checks the set against the corpus: every quote must be an exact substring of its passage, and every lookup answer must appear in its evidence.

**Lookup (30 questions).** Drawn from the 918 questions of rag-mini-wikipedia's test set — not chosen by whether a system gets them right, which would make any comparison meaningless. Procedure (`scripts/build_lookup_questions.py`): keep questions with a checkable answer found verbatim in 1-5 passages, shuffle with a fixed seed, and review in that order, keeping the first 30 that are well posed. All 67 reviewed questions have a recorded decision and reason in [`data/eval/lookup_review.csv`](data/eval/lookup_review.csv). Over half were rejected: fragments ("What centred in Sierra?"), pronouns with no referent ("When did he die?"), ambiguous names ("What was Adams' political party?"), and errors in the source data (two questions giving the same answer for different years; a heart attack attributed to Calvin Coolidge Jr. that was his father's). Gold answers are verbatim spans of the evidence, shortened by hand where the dataset gave a whole sentence.

**Calculation (24 questions).** An age, or a number of days or years between two dates stated in the corpus, plus one plain subtraction ("How old was Michael Faraday when he married Sarah Barnard?", "How many days passed between Abraham Lincoln's election and his first inauguration?"). These are written by hand from the cited passages (`scripts/build_calculation_questions.py`) and need the retriever *and* the calculator. Counting whole years depends on whether a birthday has been reached, which is where models slip without a tool: 8 of the 21 year-based questions are not answered by subtracting the two years. Each gold answer is computed by the calculator tool and cross-checked in the build script and again in the tests with an independent datetime computation; each input date must appear verbatim in the cited evidence. Questions whose result would depend on assuming the 1st for an incomplete date were left out, since the whole-year answer there is ambiguous. Numeric answers are marked `answer_type: number` so they can be judged on the number rather than the wording.

**Multi-step (24 questions).** Questions where the second lookup depends on the first: "Where was the thirteenth President of the United States born?" has to find out who that was before it can look up a birthplace (`scripts/build_multistep_questions.py`). The structure is enforced at build time and again in the tests, so each question really is two-step: the first passage states the descriptor ("thirteenth President") and does not contain the answer, and the second passage contains the answer and does *not* contain the descriptor, so a single search on the question's own words has no lexical shortcut to it. Each question also records the entity the first step resolves to, so a trace can be checked on the intermediate hop as well as on the final answer. Most chains go through U.S. presidents, because the corpus states ordinals ("thirteenth President") in one passage and biographical facts in others; a few use scientists (Avogadro, Faraday). That is a narrower spread than I would like, and it is a limitation of this set.

**Unanswerable (20 questions).** The right behaviour is to search, find nothing relevant and say so, not to answer from the model's own memory (`scripts/build_unanswerable_questions.py`). Twelve are *absent facts* about subjects the corpus does cover ("Who was James Monroe's Vice President?", "What is the capital of Ghana?") — the hard case, because related passages exist and a model that knows the answer is tempted to give it. Eight are *out of domain* ("Who painted the Mona Lisa?"). Absence is checked, not assumed: each question carries regular expressions that would match a passage stating the answer, and the build stops, and the tests fail, if any pattern matches any of the 3,200 passages. Most of my first guesses failed that check (Wilson's second wife, Webster as Fillmore's Secretary of State, Singapore's currency, the number of electoral votes Grant won are all in the corpus), which is why the patterns are broad and why this was not left to memory. A pattern check cannot prove that no passage paraphrases an answer; it is backed by reading the passages on each topic, and that residual risk remains.

### Composition and limits

| Category | Questions | Tools a correct solution uses |
|---|---|---|
| lookup | 30 | retriever |
| calculation | 24 | retriever + calculator |
| multi-step | 24 | retriever, twice |
| unanswerable | 20 | retriever, then abstain |
| **Total** | **98** | |

With 98 questions, per-category results rest on 20-30 questions each, so confidence intervals will be wide (several points either way) and are reported as such. The set is also uneven: calculation and multi-step questions lean on a handful of subjects (U.S. presidents, a few scientists). It is meant to expose differences between a fixed pipeline and an agent, not to give a precise absolute score.

## How answers are scored

Defined in `src/agentic_rag/metrics.py`, with no model in it, and tested on its own (`tests/test_metrics.py`). The answer metrics are the SQuAD-style pair from rag-wikipedia, so agent and fixed pipeline are comparable with it:

- **Correct** — the headline per-question criterion. For text answers, the normalised gold answer must appear in the answer on word boundaries (an agent replies in sentences, which exact match would score as wrong against a short gold span). For numeric answers, the gold number must be among the numbers in the answer. For unanswerable questions, abstaining *is* the correct answer. Abstaining on an answerable question is wrong.
- **Exact match and token F1** — reported next to it, as in rag-wikipedia.
- **Tool use**, from a trace of tool calls: whether every kind of tool the question needs was called, whether a two-step question made at least two retrievals, and which tools were called needlessly.
- **95% bootstrap intervals** on every rate.

Known leniencies, stated rather than hidden: "contained in the answer" can credit a verbose answer that mentions the gold among other things, and an answer that says "not in the knowledge base, but I believe it is Accra" counts as an abstention on an unanswerable question even though it leaks a guess. Both are why the final answers are kept short and why EM/F1 are shown alongside.
