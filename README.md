# MedAI RAG Test Suite

This repository contains a lightweight, controlled test suite for evaluating medical RAG systems. The suite ships with curated test documents, test case definitions, and evaluation utilities for retrieval quality, hallucination suppression, citation accuracy, and boundary handling.

## Quick start

1. Integrate the test suite with your `KnowledgeWorkspace` implementation.
2. Add the provided documents and run the tests.

```python
from tests.rag_test_suite import RAGTestSuite

workspace = KnowledgeWorkspace(
    persist_dir="./test_kb",
    llm_client=SimpleLLMClient(),
)

suite = RAGTestSuite(workspace)
suite.setup()
report = suite.run_all_tests()
print(report)
```

## LLM 实体抽取（OpenRouter 统一接入）

本项目支持通过大模型进行医学实体抽取，并统一使用 OpenRouter API 接入多种模型（如 OpenAI、Gemini、Qwen、DeepSeek 等）。

### 1) 在根目录准备 `.env`

```bash
OPENROUTER_API_KEY=your_openrouter_key
```

如需自定义网关地址，可在代码中传入 `base_url`。

### 2) 使用 LLM 抽取器

```python
from medical_agent.llm_clients import build_llm_client
from medical_agent.llm_entity_extractor import LLMEntityExtractor
from medical_agent.workspace import KnowledgeWorkspace

llm_client = build_llm_client(model="openai/gpt-4o-mini")
extractor = LLMEntityExtractor(llm_client)

workspace = KnowledgeWorkspace(
    persist_dir="./test_kb",
    llm_client=llm_client,
    entity_extractor=extractor.extract_entities,
)
```

### 3) 提示词与输出 Schema

抽取器使用的系统提示词与输出结构如下：

```text
You are a medical entity extraction engine.
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
```

```json
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
```

### 4) 对接流程说明

1. `build_llm_client` 读取根目录 `.env` 并创建 LLM 客户端。
2. `LLMEntityExtractor` 通过 `chat` 调用模型并解析 JSON 结果。
3. `KnowledgeWorkspace` 将抽取结果传入 `DocumentProcessor`，并写入知识图谱。

## What is included

- Controlled medical documents for reproducible evaluation.
- Test cases for retrieval, cross-document reasoning, hallucination detection, citation accuracy, and boundary conditions.
- Evaluation utilities such as retrieval metrics and LLM-as-judge scaffolding.

See `tests/rag_test_suite.py` for the full implementation and configuration.
