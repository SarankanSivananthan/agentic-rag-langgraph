"""Builds the 'lookup' category of the evaluation set: questions answered by a
single retrieval, with the answer stated in a known passage.

Selection is mechanical, then reviewed by hand:

1. Start from the 918 questions of rag-mini-wikipedia's test set.
2. Keep questions with a checkable answer: not yes/no, 3-60 characters, and
   found verbatim in between 1 and 5 passages of the corpus.
3. Shuffle with a fixed seed and review in that order, keeping the first 30
   that are well posed. Every reviewed question gets a recorded decision and
   reason (REVIEW below, written to lookup_review.csv), so what was rejected
   and why is as visible as what was kept.

The questions are NOT filtered by whether the retriever or the generator gets
them right: choosing the ones a system already solves would make the
comparison meaningless. Gold answers are verbatim spans of the evidence
passage (shortened by hand where the dataset gave a whole sentence).

Run from the project root: python scripts/build_lookup_questions.py
"""
from __future__ import annotations

import json
import re

import pandas as pd

SEED = 0
TARGET = 30

# Review decisions, by question id in the dataset's test set.
# KEEP: qid -> (gold answer, evidence passage id[, evidence quote]). The gold
# answer is a verbatim span of that passage, shortened by hand where the
# dataset gave a whole sentence. The evidence quote defaults to the sentence
# containing the answer; it is given explicitly where that sentence alone does
# not show why the answer fits the question.
KEEP = {
    1612: ("San Carlos, Maldonado", 51),
    1129: ("Summerhill, New York", 149),
    816: ("March 4, 1825", 2520),
    937: ("marsupium", 1737),
    80: ("Avogadro", 725),  # the passage reads "Lorenzo Romano Amedeo Carlo Avogadro"
    629: ("Warrior King", 1334),
    1653: ("Partido Colorado and Partido Blanco", 19),
    28: ("John Wilkes Booth", 357),
    1335: ("baladiyah", 2552),
    389: ("quacks like a duck", 3192),
    1231: ("for camouflage", 959,
           "All penguins are countershaded - that is, they have a white underside and a dark (mostly black) upperside. This is for camouflage."),
    1516: ("spherical", 2687),
    673: ("between 32 and 62 kilograms", 1021),
    1331: ("Doha Players Theatre", 2550),
    1556: ("mud or sand", 2687),
    164: ("mimicry", 2409),
    1518: ("every few years or more", 2689),
    1258: ("3 meters", 955,
           "Typically, penguins do not approach closer than about 3 meters (9 ft); they become nervous at about that distance. This is also the distance that Antarctic tourists are told to keep from penguins"),
    1448: ("Malay", 2773),
    846: ("1814", 2512, "When he was appointed to Secretary of War in 1814"),
    295: ("British North America Act", 2301),
    976: ("roo bars", 1763),
    1549: ("Paracryptodira", 2691),
    978: ("hopping", 1738),
    1202: ("holt", 1914),
    403: ("more than 78 million", 880,
           "As a result, most Egyptians are skeptical about the process of democratization and the role of the elections. Less than 25 percent of the country's 32 million registered voters (out of a population of more than 78 million) turned out for the 2005 elections."),
    1237: ("between 17 and 20", 952),
    1000: ("pumapard", 2485),
    26: ("March 4, 1861", 278),
    858: ("Maria Hester Monroe Gouverneur", 2522),
}

# REJECT: qid -> reason, for every question reviewed before the 30th was kept.
REJECT = {
    1094: "fragment of a sentence, not a question ('The State Court rules what?')",
    1281: "asks about skin colour; the gold answer describes fur",
    1412: "no subject: which city's bus network?",
    1042: "fragment ('What centred in Sierra?')",
    1002: "the passage gives two answers (dense vegetation, or up trees); gold lists one",
    1272: "fragment",
    1211: "no subject: which animal's guard hair?",
    1057: "ambiguous; the evidence is about a law partnership, not a factory",
    979: "no subject ('the creatures')",
    895: "ambiguous: John Adams or John Quincy Adams",
    1273: "refers to 'the page' the question was read from",
    909: "ambiguous: John Adams or John Quincy Adams",
    1279: "vague ('What is largest polar bear on record?')",
    842: "wrong name and wrong fact (a term of office, not an election)",
    347: "the passage names three species; gold lists one",
    1565: "fragment",
    140: "pronoun without a referent",
    983: "no subject",
    725: "pronoun without a referent",
    1498: "vague ('from'); the evidence is a birthplace",
    876: "dataset inconsistency: same answer as question 877 but a different year (1772 vs 1832)",
    877: "dataset inconsistency (see 876)",
    1038: "many valid answers",
    154: "vague ('What happened from 1730 to 1744?')",
    487: "premise wrong: Alexander I was the tsar, not a general",
    49: "ill-formed question",
    1191: "evidence passage does not clearly concern otters",
    782: "no subject",
    635: "ambiguous: 'makes up' a parliament",
    514: "no subject: weather where?",
    1157: "fragment",
    1566: "no subject ('the largest species')",
    155: "vague",
    889: "ambiguous: Adams",
    522: "no subject",
    637: "vague",
    182: "no subject",
    872: "fragment",
    511: "open-ended",
}


def main():
    qa = pd.read_parquet("data/source/qa_test.parquet")
    passages = pd.read_parquet("data/passages.parquet")["passage"]
    low = passages.str.lower()

    answers = qa["answer"].astype(str)
    plain = answers.str.lower().str.strip().str.strip(".")
    cand = qa[~plain.str.match(r"^(yes|no)\b") & answers.str.len().between(3, 60)].copy()
    cand["pids"] = cand["answer"].map(
        lambda a: low[low.str.contains(re.escape(a.lower().strip().strip(".")), regex=True)].index.tolist()
    )
    cand = cand[cand["pids"].map(len).between(1, 5)]
    shuffled = cand.sample(frac=1, random_state=SEED)

    rows, review = [], []
    for qid, row in shuffled.iterrows():
        if len(rows) >= TARGET:
            break
        if qid in KEEP:
            gold, pid, *explicit_quote = KEEP[qid]
            text = passages[pid]
            start = text.lower().find(gold.lower())
            assert start >= 0, (qid, gold, pid)
            if explicit_quote:
                quote = explicit_quote[0]
                assert quote in text and gold.lower() in quote.lower(), (qid, "quote not verbatim or missing the answer")
            else:
                # The sentence around the answer is the evidence quote.
                previous_end = text.rfind(". ", 0, start)
                left = 0 if previous_end < 0 else previous_end + 2
                right = text.find(". ", start + len(gold))
                quote = text[left: right + 1 if right >= 0 else len(text)].strip()
            rows.append({
                "id": f"lookup-{len(rows) + 1:03d}",
                "category": "lookup",
                "question": row["question"].strip(),
                "answer": gold,
                "expected_tools": ["retrieve"],
                "evidence": [{"passage_id": int(pid), "quote": quote}],
                "source": f"rag-mini-wikipedia test question {qid}",
            })
            review.append({"source_id": qid, "decision": "keep", "note": gold})
        elif qid in REJECT:
            review.append({"source_id": qid, "decision": "reject", "note": REJECT[qid]})
        else:
            raise SystemExit(f"question {qid} was reached in review order without a recorded decision")

    with open("data/eval/lookup.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame(review).to_csv("data/eval/lookup_review.csv", index=False)
    print(f"{len(rows)} lookup questions kept, {sum(r['decision'] == 'reject' for r in review)} rejected")


if __name__ == "__main__":
    main()
