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

AGENT_SYSTEM_PROMPT = ANSWER_RULES + """

You have three tools:
- retrieve: search the knowledge base. You may call it as many times as you need.
- calculator: arithmetic.
- date_difference: whole years or days between two dates.

Work this way:
- If a question describes a person or thing indirectly (for example "the thirteenth President"), \
first search to find out who or what it is, then search again for the fact you need.
- For any age, duration or arithmetic, use date_difference or calculator instead of working it out yourself.
- If you have searched and the knowledge base does not contain the answer, say so with the exact \
reply above. Do not guess and do not answer from memory."""
