from __future__ import annotations

from typing import Optional

import networkx as nx

from .models import DocumentChunk


class KnowledgeGraph:
    """基于NetworkX的轻量级知识图谱."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def add_document(self, doc_id: str, title: str, metadata: dict) -> None:
        """添加文档节点."""
        self.graph.add_node(
            doc_id,
            node_type="document",
            title=title,
            **metadata,
        )

    def add_chunk(self, chunk: DocumentChunk) -> None:
        """添加文档块节点，并建立与文档的关系."""
        self.graph.add_node(
            chunk.chunk_id,
            node_type="chunk",
            content=chunk.content,
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
        )
        self.graph.add_edge(chunk.doc_id, chunk.chunk_id, relation="contains")

        if chunk.chunk_index > 0:
            prev_chunks = [
                node_id
                for node_id, data in self.graph.nodes(data=True)
                if data.get("doc_id") == chunk.doc_id
                and data.get("chunk_index") == chunk.chunk_index - 1
            ]
            for prev_chunk in prev_chunks:
                self.graph.add_edge(prev_chunk, chunk.chunk_id, relation="next")

    def add_entity(self, entity: str, entity_type: str) -> str:
        """添加实体节点."""
        entity_id = f"entity:{entity}"
        self.graph.add_node(
            entity_id,
            node_type="entity",
            name=entity,
            entity_type=entity_type,
        )
        return entity_id

    def link_chunk_entity(self, chunk_id: str, entity_id: str) -> None:
        """建立块与实体的关系."""
        self.graph.add_edge(chunk_id, entity_id, relation="mentions")

    def get_related_chunks(
        self,
        chunk_ids: list[str],
        max_hops: int = 2,
        max_results: int = 10,
    ) -> list[str]:
        """基于图遍历获取相关块（用于扩展检索上下文）."""
        related: set[str] = set()
        for chunk_id in chunk_ids:
            if chunk_id not in self.graph:
                continue
            for neighbor in nx.single_source_shortest_path_length(
                self.graph,
                chunk_id,
                cutoff=max_hops,
            ):
                node_data = self.graph.nodes.get(neighbor, {})
                if node_data.get("node_type") == "chunk":
                    related.add(neighbor)

        related -= set(chunk_ids)
        return list(related)[:max_results]

    def get_context_window(self, chunk_id: str, window_size: int = 1) -> list[str]:
        """获取上下文窗口（前后相邻的块）."""
        node_data = self.graph.nodes.get(chunk_id, {})
        if not node_data:
            return []

        doc_id = node_data.get("doc_id")
        chunk_index = node_data.get("chunk_index", 0)

        context_chunks: list[tuple[int, str]] = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == "chunk" and data.get("doc_id") == doc_id:
                idx = data.get("chunk_index", 0)
                if abs(idx - chunk_index) <= window_size and node_id != chunk_id:
                    context_chunks.append((idx, node_id))

        context_chunks.sort(key=lambda item: item[0])
        return [chunk for _, chunk in context_chunks]

    def save(self, path: str) -> None:
        """持久化图."""
        nx.write_graphml(self.graph, path)

    def load(self, path: str) -> None:
        """加载图."""
        self.graph = nx.read_graphml(path)

    def get_node(self, node_id: str) -> Optional[dict]:
        """读取节点数据."""
        if node_id not in self.graph:
            return None
        return self.graph.nodes.get(node_id, {})
