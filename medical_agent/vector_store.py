from __future__ import annotations

import re
from typing import Callable, Optional


from .models import DocumentChunk
from .llm_clients import build_embedding_client


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "these",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "without",
    "would",
    "you",
    "your",
}


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.lower())
        if token and token not in _STOPWORDS
    }


class InMemoryVectorStore:
    """基于关键词重叠的轻量向量存储实现."""

    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def search(
        self,
        query: str,
        n_results: int = 10,
        filter_doc_ids: Optional[list[str]] = None,
    ) -> list[tuple[str, float, dict]]:
        query_tokens = _tokenize(query)
        results: list[tuple[str, float, dict]] = []

        for chunk in self._chunks.values():
            if filter_doc_ids and chunk.doc_id not in filter_doc_ids:
                continue
            content_tokens = _tokenize(chunk.content)
            section_tokens = _tokenize(f"{chunk.section_title} {chunk.source_title}")
            overlap_content = len(query_tokens & content_tokens)
            overlap_section = len(query_tokens & section_tokens)
            score = (overlap_content + (2 * overlap_section)) / max(
                len(query_tokens),
                1,
            )
            metadata = {
                "doc_id": chunk.doc_id,
                "source_title": chunk.source_title,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number or -1,
                "content": chunk.content,
            }
            results.append((chunk.chunk_id, score, metadata))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:n_results]

    def get(self, ids: list[str]) -> dict:
        found = [chunk_id for chunk_id in ids if chunk_id in self._chunks]
        return {"ids": found}

    def delete_document(self, doc_id: str) -> None:
        for chunk_id, chunk in list(self._chunks.items()):
            if chunk.doc_id == doc_id:
                del self._chunks[chunk_id]


class VectorStore:
    """基于Chroma的向量存储."""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "medical_knowledge",
        embedding_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
        embedding_model: str = "text-embedding-3-large",
        env_path: str = ".env",
        base_url: str | None = None,
        app_name: str = "MedAI",
        app_url: str = "https://example.com",
    ) -> None:
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_dir,
                anonymized_telemetry=False,
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_fn = embedding_fn or self._build_embedding_fn(
            embedding_model=embedding_model,
            env_path=env_path,
            base_url=base_url,
            app_name=app_name,
            app_url=app_url,
        )

    def _build_embedding_fn(
        self,
        embedding_model: str,
        env_path: str,
        base_url: str | None,
        app_name: str,
        app_url: str,
    ) -> Callable[[list[str]], list[list[float]]]:
        """通过OpenRouter获取嵌入函数."""
        client = build_embedding_client(
            model=embedding_model,
            env_path=env_path,
            base_url=base_url,
            app_name=app_name,
            app_url=app_url,
        )
        return client.embed

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """批量添加文档块."""
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        embeddings = self.embedding_fn(documents)
        metadatas = [
            {
                "doc_id": chunk.doc_id,
                "source_title": chunk.source_title,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number or -1,
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        n_results: int = 10,
        filter_doc_ids: Optional[list[str]] = None,
    ) -> list[tuple[str, float, dict]]:
        """向量检索."""
        query_embedding = self.embedding_fn([query])[0]

        where_filter = None
        if filter_doc_ids:
            where_filter = {"doc_id": {"$in": filter_doc_ids}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        output: list[tuple[str, float, dict]] = []
        for i, chunk_id in enumerate(results["ids"][0]):
            score = 1 - results["distances"][0][i]
            metadata = results["metadatas"][0][i]
            metadata["content"] = results["documents"][0][i]
            output.append((chunk_id, score, metadata))

        return output

    def get(self, ids: list[str]) -> dict:
        """获取指定chunk."""
        return self.collection.get(ids=ids)

    def delete_document(self, doc_id: str) -> None:
        """删除文档的所有块."""
        self.collection.delete(where={"doc_id": doc_id})
