from __future__ import annotations

import json
import re

from .llm_clients import BaseLLMClient


class LLMEntityExtractor:
    """LLM-based entity extractor with structured JSON output."""

    SYSTEM_PROMPT = """You are a medical entity extraction engine.
Extract named entities from the given text and return ONLY valid JSON.
Do NOT include any extra text outside JSON.

Entity types:
- Drug
- Disease
- Symptom
- Procedure
- Test
- Anatomy
- Gene
- Chemical
- Other
"""

    USER_PROMPT = """Extract entities from the following text.

Output schema (JSON):
{
  "entities": [
    {
      "text": "entity surface form",
      "type": "Drug|Disease|Symptom|Procedure|Test|Anatomy|Gene|Chemical|Other",
      "start": 0,
      "end": 10,
      "normalized": "optional canonical form or empty string",
      "confidence": 0.0
    }
  ]
}

Text:
\"\"\"{text}\"\"\"
"""

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm_client = llm_client

    def extract_entities(self, text: str) -> list[tuple[str, str]]:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT.format(text=text)},
        ]
        response_text = self.llm_client.chat(messages)
        payload = _parse_json_block(response_text)
        entities = payload.get("entities", [])

        output: list[tuple[str, str]] = []
        for item in entities:
            entity_text = str(item.get("text", "")).strip()
            entity_type = str(item.get("type", "Other")).strip() or "Other"
            if entity_text:
                output.append((entity_text, entity_type))
        return output


def _parse_json_block(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {"entities": []}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"entities": []}
