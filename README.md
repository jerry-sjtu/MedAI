# MedAI RAG Test Suite

This repository contains a lightweight, controlled test suite for evaluating medical RAG systems. The suite ships with curated test documents, test case definitions, and evaluation utilities for retrieval quality, hallucination suppression, citation accuracy, and boundary handling.

## Quick start

1. Integrate the test suite with your `KnowledgeWorkspace` implementation.
2. Add the provided documents and run the tests.

```python
from rag_test_suite import RAGTestSuite

workspace = KnowledgeWorkspace(
    persist_dir="./test_kb",
    llm_client=SimpleLLMClient(),
)

suite = RAGTestSuite(workspace)
suite.setup()
report = suite.run_all_tests()
print(report)
```

## What is included

- Controlled medical documents for reproducible evaluation.
- Test cases for retrieval, cross-document reasoning, hallucination detection, citation accuracy, and boundary conditions.
- Evaluation utilities such as retrieval metrics and LLM-as-judge scaffolding.

See `rag_test_suite.py` for the full implementation and configuration.
