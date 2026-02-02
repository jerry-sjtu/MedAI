from __future__ import annotations

import json
import re

from .models import CitedSource, GroundedResponse
from .retriever import RetrievalResult


class GroundedGenerator:
    """基于检索结果的受限生成器."""

    SYSTEM_PROMPT = """You are a medical research assistant. You MUST follow these rules strictly:

1. **ONLY use information from the provided sources.** Never use external knowledge.
2. **If the sources don't contain relevant information, say "根据提供的资料，我无法回答这个问题。"**
3. **Every claim must be cited.** Use [Source X] format where X is the source number.
4. **Do not infer or extrapolate beyond what the sources explicitly state.**
5. **Respond in the same language as the user's question.**

Sources are provided in this format:
[Source 1] Title: xxx | Section: yyy
Content: ...

[Source 2] ...
"""

    RESPONSE_FORMAT_PROMPT = """
After your response, provide a JSON block with citation details:
```json
{
  "citations_used": [1, 2],  // Source numbers you cited
  "confidence": 0.95,        // How confident you are (0-1)
  "is_fully_grounded": true  // Whether answer is 100% from sources
}
```
"""

    def __init__(self, llm_client) -> None:
        self.llm = llm_client

    def _build_source_context(
        self, retrieval_results: list[RetrievalResult]
    ) -> tuple[str, dict[int, RetrievalResult]]:
        source_map: dict[int, RetrievalResult] = {}
        context_parts: list[str] = []

        for index, result in enumerate(retrieval_results, 1):
            source_map[index] = result
            context_parts.append(
                f"[Source {index}] "
                f"Title: {result.metadata.get('source_title', 'Unknown')} | "
                f"Section: {result.metadata.get('section_title', 'N/A')}\n"
                f"Content: {result.content}\n"
            )

        return "\n".join(context_parts), source_map

    def generate(
        self,
        query: str,
        retrieval_results: list[RetrievalResult],
        conversation_history: list[dict] | None = None,
    ) -> GroundedResponse:
        source_context, source_map = self._build_source_context(retrieval_results)

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        if conversation_history:
            messages.extend(conversation_history[-6:])

        user_message = f"""
## Available Sources

{source_context}

## User Question

{query}

{self.RESPONSE_FORMAT_PROMPT}
"""
        messages.append({"role": "user", "content": user_message})

        response_text = self.llm.chat(messages)

        return self._parse_response(response_text, source_map)

    def _parse_response(
        self,
        response_text: str,
        source_map: dict[int, RetrievalResult],
    ) -> GroundedResponse:
        answer = response_text
        citations: list[CitedSource] = []
        confidence = 0.8
        is_grounded = True

        if "```json" in response_text:
            parts = response_text.split("```json")
            answer = parts[0].strip()
            try:
                json_str = parts[1].split("```")[0].strip()
                meta = json.loads(json_str)
                confidence = meta.get("confidence", 0.8)
                is_grounded = meta.get("is_fully_grounded", True)

                for source_num in meta.get("citations_used", []):
                    if source_num in source_map:
                        citations.append(self._build_citation(source_map[source_num]))
            except (json.JSONDecodeError, IndexError):
                pass

        if not citations:
            cited_nums = set(re.findall(r"\[Source (\d+)\]", answer))
            for num_str in cited_nums:
                source_num = int(num_str)
                if source_num in source_map:
                    citations.append(self._build_citation(source_map[source_num]))

        return GroundedResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            is_grounded=is_grounded,
        )

    def _build_citation(self, result: RetrievalResult) -> CitedSource:
        return CitedSource(
            chunk_id=result.chunk_id,
            doc_id=result.metadata.get("doc_id", ""),
            source_title=result.metadata.get("source_title", ""),
            section_title=result.metadata.get("section_title", ""),
            content_snippet=f"{result.content[:200]}...",
            relevance_score=result.score,
            page_number=result.metadata.get("page_number"),
        )
