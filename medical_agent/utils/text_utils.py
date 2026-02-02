from __future__ import annotations


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace for consistent chunking."""
    return " ".join(text.split())
