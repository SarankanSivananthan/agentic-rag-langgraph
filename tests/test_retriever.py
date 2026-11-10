import json

import pandas as pd
import pytest

from langchain_huggingface import HuggingFaceEmbeddings

from agentic_rag.retriever import EMBEDDING_MODEL, Hit, Retriever, build_chunks

PASSAGES = pd.DataFrame({"passage": [
    "Montevideo is the capital and largest city of Uruguay. It was founded by the Spanish in the early 18th century.",
    "Abraham Lincoln was the sixteenth President of the United States, serving from March 1861 until his assassination in April 1865.",
    "The Nile is a major river in Africa. It flows north through Sudan and Egypt into the Mediterranean Sea.",
    "Beetles are insects of the order Coleoptera. They are the largest order of animals, with over 350,000 described species.",
    "Calvin Coolidge was the thirtieth President of the United States, born on July 4, 1872 in Plymouth Notch, Vermont. " * 3,
]})


@pytest.fixture(scope="module")
def embeddings():
    # Loading the model is the slow part, so every test shares one copy.
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@pytest.fixture(scope="module")
def retriever(tmp_path_factory, embeddings):
    index_dir = tmp_path_factory.mktemp("index")
    return Retriever.from_passages(PASSAGES, index_dir=index_dir, chunk_size=120, chunk_overlap=10, embeddings=embeddings)


def test_search_returns_hits_closest_first(retriever):
    hits = retriever.search("Which river flows through Egypt?", k=3)
    assert len(hits) == 3
    assert all(isinstance(hit, Hit) for hit in hits)
    distances = [hit.distance for hit in hits]
    assert distances == sorted(distances)


def test_search_finds_the_relevant_passage(retriever):
    assert retriever.search("What is the capital of Uruguay?", k=1)[0].passage_id == 0
    assert retriever.search("Who was the sixteenth president?", k=1)[0].passage_id == 1
    assert retriever.search("Which insects form the largest order of animals?", k=1)[0].passage_id == 3


def test_hit_text_comes_from_its_source_passage(retriever):
    for hit in retriever.search("Where is the Nile?", k=4):
        assert hit.text in PASSAGES["passage"][hit.passage_id]


def test_k_larger_than_the_index_returns_everything_available(retriever):
    assert len(retriever.search("anything", k=1000)) == len(build_chunks(PASSAGES, 120, 10))


def test_chunks_respect_the_size_limit_and_keep_their_passage_id():
    chunks = build_chunks(PASSAGES, chunk_size=120, chunk_overlap=10)
    assert all(len(chunk.page_content) <= 120 for chunk in chunks)
    coolidge = [chunk for chunk in chunks if chunk.metadata["passage_id"] == 4]
    assert len(coolidge) > 1  # the long passage really was split


def test_cached_index_is_reused_when_settings_match(tmp_path, embeddings):
    first = Retriever.from_passages(PASSAGES, index_dir=tmp_path, chunk_size=120, chunk_overlap=10, embeddings=embeddings)
    # No passages given: it can only succeed by loading what the first call saved.
    second = Retriever.from_passages(None, index_dir=tmp_path, chunk_size=120, chunk_overlap=10, embeddings=embeddings)
    query = "What is the capital of Uruguay?"
    assert second.search(query, k=1) == first.search(query, k=1)


def test_cached_index_is_rebuilt_when_settings_change(tmp_path, embeddings):
    Retriever.from_passages(PASSAGES, index_dir=tmp_path, chunk_size=120, chunk_overlap=10, embeddings=embeddings)
    Retriever.from_passages(PASSAGES, index_dir=tmp_path, chunk_size=200, chunk_overlap=10, embeddings=embeddings)
    assert json.loads((tmp_path / "meta.json").read_text())["chunk_size"] == 200
