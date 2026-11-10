"""Builds the 'multi_step' category: questions that need two retrievals, the
second depending on the first. "Where was the thirteenth President born?" has
to find out who that was (one passage) before looking up a birthplace (another).

The structure is checked at build time, so each question really is two-step:
  - the first passage states the descriptor ("thirteenth President") and does
    NOT contain the answer;
  - the second passage contains the answer and does NOT contain the descriptor,
    so a single search on the question's own words has no lexical shortcut to it.

Each question records the entity the first step resolves to, so a trace can be
checked for the intermediate hop as well as the final answer.

Run from the project root: python scripts/build_multistep_questions.py
"""
from __future__ import annotations

import json

import pandas as pd

# (question, answer, entity found by step 1,
#  (first passage id, descriptor), (second passage id, descriptor that must be absent from it))
# The descriptor is verbatim text of the first passage; the answer is a verbatim span of the second.
QUESTIONS = [
    ("In which county was the sixteenth President of the United States born?", "Hardin County", "Abraham Lincoln", (278, "sixteenth President"), 282),
    ("Who did the sixteenth President of the United States marry in 1842?", "Mary Todd", "Abraham Lincoln", (278, "sixteenth President"), 296),
    ("In what year was the first photograph of the sixteenth President of the United States taken?", "1846", "Abraham Lincoln", (278, "sixteenth President"), 299),
    ("Where was the thirteenth President of the United States born?", "Summerhill, New York", "Millard Fillmore", (148, "thirteenth President"), 149),
    ("Who did the thirteenth President of the United States marry in 1826?", "Abigail Powers", "Millard Fillmore", (148, "thirteenth President"), 151),
    ("Who did the thirteenth President of the United States marry in 1858?", "Caroline McIntosh", "Millard Fillmore", (148, "thirteenth President"), 185),
    ("Which college did the thirtieth President of the United States attend?", "Amherst College", "Calvin Coolidge", (399, "thirtieth President"), 405),
    ("Where is the thirtieth President of the United States buried?", "Notch Cemetery", "Calvin Coolidge", (399, "thirtieth President"), 459),
    ("In which town was the President who succeeded to the office on Warren G. Harding's death born?", "Plymouth", "Calvin Coolidge", (399, "Warren G. Harding"), 402),
    ("Which university did the thirty-eighth President of the United States attend as an undergraduate?", "University of Michigan", "Gerald Ford", (1992, "thirty-eighth President"), 2003),
    ("Which positions did the thirty-eighth President of the United States play on his university football team?", "center and linebacker", "Gerald Ford", (1992, "thirty-eighth President"), 2003),
    ("Who did the President who assumed office after Richard Nixon's resignation marry in 1948?", "Elizabeth Bloomer Warren", "Gerald Ford", (2041, "Nixon resigned"), 2022),
    ("Where was the twenty-eighth President of the United States born?", "Staunton, Virginia", "Woodrow Wilson", (2129, "twenty-eighth President"), 2132),
    ("Where is the twenty-eighth President of the United States buried?", "Washington National Cathedral", "Woodrow Wilson", (2129, "twenty-eighth President"), 2224),
    ("Who did the fifth President of the United States marry in 1786?", "Elizabeth Kortright", "James Monroe", (2506, "fifth President"), 2507),
    ("On what date did the fifth President of the United States deliver the doctrine named after him to Congress?", "December 2, 1823", "James Monroe", (2506, "fifth President"), 2516),
    ("Who did the twenty-second President of the United States marry in the White House?", "Frances Cornelia Folsom", "Grover Cleveland", (473, "twenty-second"), 496),
    ("On what date did the eighteenth President of the United States resign from the Army?", "July 31, 1854", "Ulysses S. Grant", (1768, "eighteenth President"), 1778),
    ("Who did the eighteenth President of the United States marry in 1848?", "Julia Boggs Dent", "Ulysses S. Grant", (1768, "eighteenth President"), 1775),
    ("Which college did the second President of the United States attend at the age of sixteen?", "Harvard College", "John Adams", (635, "second President"), 639),
    ("Where was the twenty-sixth President of the United States born?", "Gramercy section of New York City", "Theodore Roosevelt", (2878, "twenty-sixth President"), 2884),
    ("In which city was the Count most noted for contributions to the theory of molarity and molecular weight born?", "Turin", "Amedeo Avogadro", (725, "most noted for his contributions to the theory of molarity"), 726),
    ("What was the nationality of the chemist who discovered benzene?", "English", "Michael Faraday", (76, "discovered benzene"), 74),
    ("In which home did the twenty-eighth President of the United States die?", "S Street", "Woodrow Wilson", (2129, "twenty-eighth President"), 2224),
]


def sentence_with(text: str, needle: str) -> str:
    start = text.index(needle)
    previous_end = text.rfind(". ", 0, start)
    left = 0 if previous_end < 0 else previous_end + 2
    right = text.find(". ", start + len(needle))
    return text[left: right + 1 if right >= 0 else len(text)].strip()


def main():
    passages = pd.read_parquet("data/passages.parquet")["passage"]
    rows = []
    for question, answer, entity, (pid1, descriptor), pid2 in QUESTIONS:
        first, second = passages[pid1], passages[pid2]
        assert pid1 != pid2, question
        assert descriptor in first, (question, "descriptor missing from the first passage")
        assert answer.lower() not in first.lower(), (question, "the first passage already contains the answer")
        assert answer.lower() in second.lower(), (question, "the second passage does not contain the answer")
        assert descriptor.lower() not in second.lower(), (question, "the second passage contains the descriptor")
        rows.append({
            "question": question, "answer": answer, "expected_tools": ["retrieve"],
            "evidence": [
                {"passage_id": pid1, "quote": sentence_with(first, descriptor)},
                {"passage_id": pid2, "quote": sentence_with(second, answer)},
            ],
            "chain": {"descriptor": descriptor, "entity": entity},
        })

    with open("data/eval/multi_step.jsonl", "w") as f:
        for i, row in enumerate(rows, start=1):
            f.write(json.dumps({"id": f"multi-step-{i:03d}", "category": "multi_step", **row,
                                "source": "written by hand from the cited passages"}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} multi-step questions written")


if __name__ == "__main__":
    main()
