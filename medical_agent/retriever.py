from __future__ import annotations

from typing import NamedTuple

from .knowledge_graph import KnowledgeGraph
from .vector_store import VectorStore


class RetrievalResult(NamedTuple):
    chunk_id: str
    content: str
    score: float
    metadata: dict
    retrieval_method: str


class HybridRetriever:
    """混合检索器：结合向量检索 + 图遍历."""

    def __init__(self, vector_store: VectorStore, knowledge_graph: KnowledgeGraph) -> None:
        self.vector_store = vector_store
        self.kg = knowledge_graph

    def _context_sort_key(self, context_id: str, current_index: int) -> tuple[int, int]:
        node_data = self.kg.get_node(context_id) or {}
        context_index = node_data.get("chunk_index")
        if context_index is None:
            return (1, 0)
        prefer_forward = 0 if context_index > current_index else 1
        return (prefer_forward, abs(context_index - current_index))

    def retrieve(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int = 5,
        expand_context: bool = True,
        context_window: int = 1,
    ) -> list[RetrievalResult]:
        """混合检索流程."""
        vector_results = self.vector_store.search(
            query=query,
            n_results=top_k * 2,
            filter_doc_ids=doc_ids,
        )

        if not vector_results:
            return []

        max_score = max(result[1] for result in vector_results)
        if max_score <= 0.05:
            return []

        candidate_chunk_ids = [result[0] for result in vector_results[:top_k]]

        results: list[RetrievalResult] = []
        seen_ids: set[str] = set()

        for chunk_id, score, metadata in vector_results[:top_k]:
            if chunk_id not in seen_ids:
                results.append(
                    RetrievalResult(
                        chunk_id=chunk_id,
                        content=metadata.get("content", ""),
                        score=score,
                        metadata=metadata,
                        retrieval_method="vector",
                    )
                )
                seen_ids.add(chunk_id)

            if expand_context:
                context_chunks = self.kg.get_context_window(
                    chunk_id,
                    window_size=context_window,
                )
                current_index = metadata.get("chunk_index")
                if current_index is not None:
                    context_chunks = sorted(
                        context_chunks,
                        key=lambda context_id: (
                            self._context_sort_key(context_id, current_index)
                        ),
                    )
                for context_id in context_chunks:
                    if context_id in seen_ids:
                        continue
                    node_data = self.kg.get_node(context_id) or {}
                    if node_data:
                        results.append(
                            RetrievalResult(
                                chunk_id=context_id,
                                content=node_data.get("content", ""),
                                score=score * 0.9,
                                metadata={
                                    "doc_id": node_data.get("doc_id"),
                                    "chunk_index": node_data.get("chunk_index"),
                                },
                                retrieval_method="graph",
                            )
                        )
                        seen_ids.add(context_id)

        if expand_context:
            related_chunks = self.kg.get_related_chunks(
                candidate_chunk_ids,
                max_hops=2,
                max_results=5,
            )
            for chunk_id in related_chunks:
                if chunk_id in seen_ids:
                    continue
                node_data = self.kg.get_node(chunk_id) or {}
                if node_data:
                    results.append(
                        RetrievalResult(
                            chunk_id=chunk_id,
                            content=node_data.get("content", ""),
                            score=0.5,
                            metadata={
                                "doc_id": node_data.get("doc_id"),
                                "chunk_index": node_data.get("chunk_index"),
                            },
                            retrieval_method="graph",
                        )
                    )
                    seen_ids.add(chunk_id)

        return results
