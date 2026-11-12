import pytest

from agentic_rag.retriever import Hit
from agentic_rag.tools import make_tools


class StubRetriever:
    def __init__(self, hits):
        self.hits, self.queries = hits, []

    def search(self, query, k=4):
        self.queries.append((query, k))
        return self.hits[:k]


@pytest.fixture
def tools():
    hits = [Hit("Montevideo is the capital of Uruguay.", 1, 0.2), Hit("Uruguay is in South America.", 2, 0.5)]
    stub = StubRetriever(hits)
    return {t.name: t for t in make_tools(stub, k=2)}, stub


def test_the_model_gets_three_named_tools_with_descriptions(tools):
    by_name, _ = tools
    assert set(by_name) == {"retrieve", "calculator", "date_difference"}
    assert all(t.description for t in by_name.values())


def test_retrieve_numbers_the_passages_and_passes_k_through(tools):
    by_name, stub = tools
    result = by_name["retrieve"].invoke({"query": "capital of Uruguay"})
    assert result == "[1] Montevideo is the capital of Uruguay.\n\n[2] Uruguay is in South America."
    assert stub.queries == [("capital of Uruguay", 2)]


def test_retrieve_reports_an_empty_query_and_no_results():
    empty_query = {t.name: t for t in make_tools(StubRetriever([]))}["retrieve"]
    assert empty_query.invoke({"query": "   "}).startswith("Error")
    assert empty_query.invoke({"query": "anything"}) == "No passages found."


def test_calculator_evaluates_and_reports_errors_as_text(tools):
    calculator = tools[0]["calculator"]
    assert calculator.invoke({"expression": "(62 - 32) / 2"}) == "15"
    assert "division by zero" in calculator.invoke({"expression": "1 / 0"})
    assert calculator.invoke({"expression": "__import__('os')"}).startswith("Error")


def test_date_difference_returns_years_or_days_and_reports_errors(tools):
    date_difference = tools[0]["date_difference"]
    assert date_difference.invoke({"start": "February 12, 1809", "end": "November 6, 1860"}) == "51 years"
    assert date_difference.invoke({"start": "1865-04-14", "end": "1865-04-15", "unit": "days"}) == "1 days"
    assert date_difference.invoke({"start": "1865", "end": "1809"}).startswith("Error")
    assert date_difference.invoke({"start": "soon", "end": "1809"}).startswith("Error")
