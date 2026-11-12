"""Instructions shared by the baseline and the agent.

The faithfulness rules are identical on both sides on purpose: when the two are
compared, the only thing that should differ is whether the system may choose
its own tools and repeat searches, not how it was told to treat the sources.
"""

ANSWER_RULES = (
    "Answer using only facts found in the knowledge base passages. Do not use your own memory, "
    "even if you are sure you know the answer. If the passages do not contain the answer, reply "
    "exactly: \"I don't know - the knowledge base does not contain this.\" "
    "Give the final answer as one short sentence."
)

BASELINE_SYSTEM_PROMPT = ANSWER_RULES

BASELINE_USER_TEMPLATE = "Passages:\n{context}\n\nQuestion: {question}"
