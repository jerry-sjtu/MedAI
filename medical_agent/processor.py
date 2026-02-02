from __future__ import annotations

from collections.abc import Iterator
import re

from .knowledge_graph import KnowledgeGraph
from .models import DocumentChunk
from .vector_store import VectorStore


class DocumentProcessor:
    """文档处理器 - 将预抽取的文本转换为可索引的块."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        knowledge_graph: KnowledgeGraph | None = None,
        vector_store: VectorStore | None = None,
        entity_extractor=None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kg = knowledge_graph
        self.vector_store = vector_store
        self.entity_extractor = entity_extractor or self._get_entity_extractor()

    def _get_entity_extractor(self):
        """获取医学实体抽取器（可替换为专业NER模型）."""

        def extract_entities(text: str) -> list[tuple[str, str]]:
            entities: list[tuple[str, str]] = []
            drug_pattern = r"([A-Z][a-z]+(?:mab|nib|pril|olol|statin))"
            for match in re.finditer(drug_pattern, text):
                entities.append((match.group(1), "Drug"))
            return entities

        return extract_entities

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        source_title: str,
        section_titles: list[str] | None = None,
    ) -> Iterator[DocumentChunk]:
        """将文本切分为块."""
        paragraphs = text.split("\n\n")

        current_chunk = ""
        current_section = section_titles[0] if section_titles else ""
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if para.startswith("#") or para.isupper():
                current_section = para.lstrip("#").strip()
                continue

            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += f"{para}\n\n"
                continue

            if current_chunk:
                yield DocumentChunk(
                    doc_id=doc_id,
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    source_title=source_title,
                    section_title=current_section,
                    entities=[entity[0] for entity in self.entity_extractor(current_chunk)],
                )
                chunk_index += 1

            overlap_text = (
                current_chunk[-self.chunk_overlap :]
                if len(current_chunk) > self.chunk_overlap
                else ""
            )
            current_chunk = f"{overlap_text}{para}\n\n"

        if current_chunk.strip():
            yield DocumentChunk(
                doc_id=doc_id,
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                source_title=source_title,
                section_title=current_section,
                entities=[entity[0] for entity in self.entity_extractor(current_chunk)],
            )

    def process_document(
        self,
        doc_id: str,
        title: str,
        text: str,
        metadata: dict | None = None,
        image_urls: list[str] | None = None,
    ) -> int:
        """处理单个文档：切块 -> 抽取实体 -> 存入图和向量库."""
        if not self.kg or not self.vector_store:
            raise ValueError("knowledge_graph and vector_store must be provided")

        metadata = metadata or {}

        self.kg.add_document(doc_id, title, metadata)

        chunks = list(self.chunk_text(text, doc_id, title))
        for chunk in chunks:
            if image_urls:
                chunk.image_urls.extend(image_urls)
            self.kg.add_chunk(chunk)
            for entity in chunk.entities:
                entity_id = self.kg.add_entity(entity, "medical_entity")
                self.kg.link_chunk_entity(chunk.chunk_id, entity_id)

        self.vector_store.add_chunks(chunks)

        return len(chunks)
