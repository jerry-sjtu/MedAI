from __future__ import annotations

import json
import os

from .generator import GroundedGenerator
from .knowledge_graph import KnowledgeGraph
from .processor import DocumentProcessor
from .retriever import HybridRetriever
from .vector_store import VectorStore


class KnowledgeWorkspace:
    """知识工作空间 - 统一入口."""

    def __init__(
        self,
        persist_dir: str = "./knowledge_base",
        llm_client=None,
        vector_store: VectorStore | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        retriever: HybridRetriever | None = None,
        generator: GroundedGenerator | None = None,
        processor: DocumentProcessor | None = None,
    ) -> None:
        self.persist_dir = persist_dir

        self.vector_store = vector_store or VectorStore(persist_dir=f"{persist_dir}/chroma")
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.retriever = retriever or HybridRetriever(self.vector_store, self.knowledge_graph)
        self.generator = generator or GroundedGenerator(llm_client)
        self.processor = processor or DocumentProcessor(
            knowledge_graph=self.knowledge_graph,
            vector_store=self.vector_store,
        )

        self.active_doc_ids: set[str] = set()

    def add_document(
        self,
        doc_id: str,
        title: str,
        text: str,
        metadata: dict | None = None,
    ) -> int:
        """添加文档到工作空间."""
        num_chunks = self.processor.process_document(
            doc_id=doc_id,
            title=title,
            text=text,
            metadata=metadata,
        )
        self.active_doc_ids.add(doc_id)
        return num_chunks

    def remove_document(self, doc_id: str) -> None:
        """从工作空间移除文档."""
        self.vector_store.delete_document(doc_id)
        self.active_doc_ids.discard(doc_id)

    def query(
        self,
        question: str,
        top_k: int = 5,
        conversation_history: list[dict] | None = None,
    ):
        """提问 - 只基于工作空间内的文档回答."""
        results = self.retriever.retrieve(
            query=question,
            doc_ids=list(self.active_doc_ids) if self.active_doc_ids else None,
            top_k=top_k,
            expand_context=True,
        )
        return self.generator.generate(
            query=question,
            retrieval_results=results,
            conversation_history=conversation_history,
        )

    def save(self) -> None:
        """持久化工作空间状态."""
        os.makedirs(self.persist_dir, exist_ok=True)
        self.knowledge_graph.save(f"{self.persist_dir}/graph.graphml")
        with open(f"{self.persist_dir}/active_docs.json", "w", encoding="utf-8") as file:
            json.dump(list(self.active_doc_ids), file, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """加载工作空间状态."""
        graph_path = f"{self.persist_dir}/graph.graphml"
        if os.path.exists(graph_path):
            self.knowledge_graph.load(graph_path)

        docs_path = f"{self.persist_dir}/active_docs.json"
        if os.path.exists(docs_path):
            with open(docs_path, encoding="utf-8") as file:
                self.active_doc_ids = set(json.load(file))
