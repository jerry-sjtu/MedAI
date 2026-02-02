from __future__ import annotations

from typing import Callable, Optional

import chromadb
from chromadb.config import Settings

from .models import DocumentChunk


class VectorStore:
    """基于Chroma的向量存储."""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "medical_knowledge",
        embedding_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
    ) -> None:
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
        self.embedding_fn = embedding_fn or self._get_embedding_fn()

    def _get_embedding_fn(self) -> Callable[[list[str]], list[list[float]]]:
        """获取嵌入函数 - 可替换为其他模型."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-m3")
        return lambda texts: model.encode(texts).tolist()

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

    def delete_document(self, doc_id: str) -> None:
        """删除文档的所有块."""
        self.collection.delete(where={"doc_id": doc_id})
