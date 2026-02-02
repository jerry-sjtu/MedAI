from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


@dataclass
class DocumentChunk:
    """文档块 - 最小检索单元."""

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    content: str = ""
    chunk_index: int = 0

    source_title: str = ""
    section_title: str = ""
    page_number: Optional[int] = None
    image_urls: list[str] = field(default_factory=list)

    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    summary: str = ""

    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CitedSource:
    """引用来源 - 用于追溯回答的出处."""

    chunk_id: str
    doc_id: str
    source_title: str
    section_title: str
    content_snippet: str
    relevance_score: float
    page_number: Optional[int] = None


@dataclass
class GroundedResponse:
    """带引用的回答."""

    answer: str
    citations: list[CitedSource]
    confidence: float
    is_grounded: bool
