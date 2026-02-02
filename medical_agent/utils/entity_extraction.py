from __future__ import annotations

import re


def extract_simple_drug_entities(text: str) -> list[str]:
    """简化的药物实体抽取规则."""
    drug_pattern = r"([A-Z][a-z]+(?:mab|nib|pril|olol|statin))"
    return [match.group(1) for match in re.finditer(drug_pattern, text)]
