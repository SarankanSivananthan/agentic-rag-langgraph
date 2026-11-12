"""The tools the model can call.

  retrieve         search the Wikipedia knowledge base
  calculator       evaluate an arithmetic expression
  date_difference  whole years or days between two dates

Each returns a plain string. Failures come back as "Error: ..." text rather
than exceptions, so the model reads what went wrong and can try again instead
of the whole run crashing.
"""
from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from agentic_rag.calculator import CalculatorError, calculate, date_difference
from agentic_rag.retriever import Retriever

DEFAULT_TOP_K = 4


def make_tools(retriever: Retriever, k: int = DEFAULT_TOP_K) -> list[BaseTool]:
    def retrieve(query: str) -> str:
        if not query.strip():
            return "Error: the query is empty."
        hits = retriever.search(query, k=k)
        if not hits:
            return "No passages found."
        return "\n\n".join(f"[{i}] {hit.text}" for i, hit in enumerate(hits, start=1))

    def calculator(expression: str) -> str:
        try:
            return calculate(expression)
        except CalculatorError as error:
            return f"Error: {error}"

    def date_difference_tool(start: str, end: str, unit: str = "years") -> str:
        try:
            return date_difference(start, end, unit)
        except CalculatorError as error:
            return f"Error: {error}"

    return [
        StructuredTool.from_function(
            retrieve, name="retrieve",
            description=(
                "Search the Wikipedia knowledge base and return the most relevant passages. "
                "Use a short, specific query naming the person or thing you need a fact about. "
                "The passages are the only source of facts; they may not contain the answer."
            ),
        ),
        StructuredTool.from_function(
            calculator, name="calculator",
            description=(
                "Evaluate an arithmetic expression using numbers and + - * / // % ** with "
                "parentheses, for example '(62 - 32) / 2'. Use it instead of calculating by hand."
            ),
        ),
        StructuredTool.from_function(
            date_difference_tool, name="date_difference",
            description=(
                "Time between two dates, in whole 'years' (counted like an age: a birthday not "
                "yet reached does not count) or in 'days'. Dates look like 'February 12, 1809', "
                "'1809-02-12' or '1809'. Use it for ages and durations instead of subtracting years."
            ),
        ),
    ]
