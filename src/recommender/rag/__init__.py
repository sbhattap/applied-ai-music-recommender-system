"""Retrieval-augmented layer: evidence documents, hybrid retriever, citations."""

from .documents import Document, build_documents
from .retriever import Evidence, Retriever, SongHit, get_retriever

__all__ = ["Document", "Evidence", "Retriever", "SongHit", "build_documents", "get_retriever"]
