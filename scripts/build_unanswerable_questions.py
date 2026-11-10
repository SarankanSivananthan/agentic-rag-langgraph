"""Builds the 'unanswerable' category: questions whose answer is NOT in the
corpus, where the right behaviour is to search, find nothing relevant and say
so — not to answer from the model's own memory.

Two kinds:
  absent_fact    about a subject the corpus does cover, but asking for a fact it
                 does not state (the hard case: related passages exist, and a
                 model that knows the answer can be tempted to give it)
  out_of_domain  about something the corpus never mentions

Absence is checked, not assumed. Each question carries regular expressions that
would match a passage stating the answer (or naming the thing asked about); the
build stops if any pattern matches any of the 3,200 passages. Many obvious
candidates failed this check while the set was being written (Wilson's second
wife, Webster as Fillmore's Secretary of State, the dollar as Singapore's
currency and others are all in the corpus), which is why the patterns are broad.
A pattern check cannot prove that no passage paraphrases the answer; it is
backed by reading the passages on each topic, and that residual risk is stated
in the README.

Run from the project root: python scripts/build_unanswerable_questions.py
"""
from __future__ import annotations

import json
import re

import pandas as pd

# (question, subtype, [patterns that must match no passage])
QUESTIONS = [
    ("What is the capital of Ghana?", "absent_fact",
     [r"Accra", r"Ghana.{0,250}\bcapital\b", r"\bcapital\b.{0,120}Ghana"]),
    ("Which species is the largest beetle in the world?", "absent_fact",
     [r"Goliath", r"Titanus", r"Hercules beetle", r"largest (species of )?beetle", r"heaviest beetle"]),
    ("How long do polar bears live in the wild?", "absent_fact",
     [r"polar bears?.{0,500}(lifespan|life expectancy|live (for )?(up to |to )?\d+|\d+ years in the wild)"]),
    ("How long is the gestation period of a polar bear?", "absent_fact",
     [r"polar bear.{0,400}gestation", r"gestation.{0,300}polar bear"]),
    ("What pets did Calvin Coolidge keep in the White House?", "absent_fact",
     [r"Coolidge.{0,300}\b(raccoon|Rebecca|pet|pets|dog|dogs|cat|cats)\b"]),
    ("Who was James Monroe's Vice President?", "absent_fact",
     [r"Tompkins", r"Monroe.{0,200}[Vv]ice [Pp]resident", r"[Vv]ice [Pp]resident.{0,200}Monroe"]),
    ("Which political party did Kwame Nkrumah found?", "absent_fact",
     [r"Convention People", r"Nkrumah.{0,300}party"]),
    ("How many eggs does a duck typically lay in a clutch?", "absent_fact",
     [r"duck.{0,300}\b(clutch|lay)\b.{0,100}eggs"]),
    ("Who was the first European to explore the coast of what is now Uruguay?", "absent_fact",
     [r"Sol[ií]s", r"(first|earliest) European.{0,120}Uruguay", r"Uruguay.{0,200}(explor|navigator|conquistador)"]),
    ("Who was Grover Cleveland's Vice President during his first term?", "absent_fact",
     [r"Hendricks", r"Stevenson"]),
    ("How long do otters live?", "absent_fact",
     [r"otters?.{0,300}(lifespan|live (for )?(up to |to )?\d+|\d+ years in the wild)"]),
    ("How many seats are there in the Canadian House of Commons?", "absent_fact",
     [r"House of Commons.{0,200}\b(308|338|301)\b", r"\b(308|338) (seats|members)"]),
    ("Who painted the Mona Lisa?", "out_of_domain", [r"Mona Lisa"]),
    ("What is the capital of Australia?", "out_of_domain", [r"Canberra"]),
    ("Who wrote Pride and Prejudice?", "out_of_domain", [r"Austen", r"Pride and Prejudice"]),
    ("What is the boiling point of ethanol?", "out_of_domain", [r"ethanol", r"ethyl alcohol"]),
    ("What is the chemical formula of table salt?", "out_of_domain", [r"NaCl", r"sodium chloride"]),
    ("What is the speed of light in a vacuum?", "out_of_domain", [r"speed of light", r"299[, ]?792"]),
    ("Which country won the 1998 FIFA World Cup?", "out_of_domain", [r"World Cup.{0,60}1998", r"1998.{0,60}World Cup"]),
    ("Who discovered penicillin?", "out_of_domain", [r"penicillin"]),
]


# The subject of each absent_fact question: the corpus must cover it, otherwise
# the question would really be out_of_domain.
TOPICS = {
    "What is the capital of Ghana?": "Ghana",
    "Which species is the largest beetle in the world?": "beetle",
    "How long do polar bears live in the wild?": "polar bear",
    "How long is the gestation period of a polar bear?": "polar bear",
    "What pets did Calvin Coolidge keep in the White House?": "Coolidge",
    "Who was James Monroe's Vice President?": "Monroe",
    "Which political party did Kwame Nkrumah found?": "Nkrumah",
    "How many eggs does a duck typically lay in a clutch?": "duck",
    "Who was the first European to explore the coast of what is now Uruguay?": "Uruguay",
    "Who was Grover Cleveland's Vice President during his first term?": "Cleveland",
    "How long do otters live?": "otter",
    "How many seats are there in the Canadian House of Commons?": "House of Commons",
}


def main():
    passages = pd.read_parquet("data/passages.parquet")["passage"]
    rows = []
    for question, subtype, patterns in QUESTIONS:
        for pattern in patterns:
            regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)
            matches = [pid for pid, text in passages.items() if regex.search(text)]
            assert not matches, f"{question!r}: pattern {pattern!r} matches passages {matches[:5]}"
        topic = TOPICS.get(question)
        assert (topic is None) == (subtype == "out_of_domain"), question
        if topic:
            assert passages.str.contains(topic, case=False, regex=False).any(), f"{question!r}: topic {topic!r} is not in the corpus"
        rows.append({
            "question": question, "answer": None, "expected_tools": ["retrieve"],
            "subtype": subtype, "topic": topic, "absence_patterns": patterns,
        })

    with open("data/eval/unanswerable.jsonl", "w") as f:
        for i, row in enumerate(rows, start=1):
            f.write(json.dumps({"id": f"unanswerable-{i:03d}", "category": "unanswerable", **row,
                                "evidence": [], "source": "written by hand; absence checked against all passages"},
                               ensure_ascii=False) + "\n")
    kinds = pd.Series([r["subtype"] for r in rows]).value_counts().to_dict()
    print(f"{len(rows)} unanswerable questions written: {kinds}")


if __name__ == "__main__":
    main()
