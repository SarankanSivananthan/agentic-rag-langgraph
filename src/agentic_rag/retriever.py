"""Retrieval over the Wikipedia passages — the knowledge base the agent searches.

Same setup as the rag-wikipedia project (250-character chunks with 25 of
overlap, `all-MiniLM-L6-v2` embeddings, a local FAISS index), wrapped in a
small `Retriever` that returns plain `Hit` records. Building the index takes
a minute or two on CPU, so it is cached on disk and rebuilt only when the
chunking or embedding settings change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSAGES_PATH = PROJECT_ROOT / "data" / "passages.parquet"
INDEX_DIR = PROJECT_ROOT / "data" / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 25


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk."""

    text: str
    passage_id: int  # row of the source passage in passages.parquet
    distance: float  # L2 distance to the query embedding: lower means closer


def load_passages(path: Path = PASSAGES_PATH) -> pd.DataFrame:
    """The 3,200 Wikipedia passages of rag-datasets/rag-mini-wikipedia (CC-BY 3.0)."""
    return pd.read_parquet(path)


def build_chunks(
    passages: pd.DataFrame, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[Document]:
    """Split every passage into chunks, each remembering which passage it came from."""
    documents = [
        Document(page_content=row.passage, metadata={"passage_id": int(row.Index)})
        for row in passages.itertuples()
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


class Retriever:
    def __init__(self, store: FAISS):
        self._store = store

    @classmethod
    def from_passages(
        cls,
        passages: pd.DataFrame | None = None,
        index_dir: Path = INDEX_DIR,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        embedding_model: str = EMBEDDING_MODEL,
        embeddings: Embeddings | None = None,
    ) -> "Retriever":
        """Load the cached index if it was built with these settings, else build and cache it.

        Pass an already-loaded `embeddings` object to share one model between
        retrievers; `embedding_model` is then only the name recorded in the
        cache settings and must describe that object.
        """
        settings = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
        }
        embeddings = embeddings or HuggingFaceEmbeddings(model_name=embedding_model)
        meta_path = Path(index_dir) / "meta.json"

        if meta_path.exists() and json.loads(meta_path.read_text()) == settings:
            # The pickle in this directory was written by this project, never downloaded.
            return cls(FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True))

        passages = load_passages() if passages is None else passages
        store = FAISS.from_documents(build_chunks(passages, chunk_size, chunk_overlap), embeddings)
        store.save_local(str(index_dir))
        meta_path.write_text(json.dumps(settings))
        return cls(store)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        """The k chunks closest to the query, closest first."""
        results = self._store.similarity_search_with_score(query, k=k)
        return [
            Hit(text=doc.page_content, passage_id=doc.metadata["passage_id"], distance=float(distance))
            for doc, distance in results
        ]
