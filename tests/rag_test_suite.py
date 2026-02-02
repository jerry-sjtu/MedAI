"""RAG system test suite for controlled medical QA scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Iterable, List, Protocol


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    doc_id: str
    content_snippet: str


@dataclass(frozen=True)
class GroundedResponse:
    answer: str
    citations: List[Citation]
    is_grounded: bool


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> List[object]:
        ...


class VectorStore(Protocol):
    def get(self, ids: List[str]) -> dict:
        ...


class KnowledgeWorkspace(Protocol):
    retriever: Retriever
    vector_store: object

    def add_document(self, doc_id: str, title: str, text: str) -> None:
        ...

    def query(self, question: str) -> GroundedResponse:
        ...


TEST_DOCUMENTS = [
    Document(
        doc_id="doc_diabetes",
        title="2型糖尿病诊疗规范",
        text=(
            "# 第一章 诊断标准\n\n"
            "2型糖尿病的诊断需满足以下任一条件：\n"
            "1. 空腹血糖（FPG）≥ 7.0 mmol/L\n"
            "2. 口服葡萄糖耐量试验（OGTT）2小时血糖 ≥ 11.1 mmol/L\n"
            "3. 随机血糖 ≥ 11.1 mmol/L，且伴有典型症状（多饮、多尿、体重下降）\n"
            "4. 糖化血红蛋白（HbA1c）≥ 6.5%\n\n"
            "注意：无症状患者需在不同日期复测确认。\n\n"
            "# 第二章 药物治疗\n\n"
            "## 2.1 一线用药\n\n"
            "二甲双胍（Metformin）为首选药物：\n"
            "- 起始剂量：500mg，每日2次，随餐服用\n"
            "- 最大剂量：2000mg/日\n"
            "- 禁忌症：肾功能不全（eGFR < 30）、严重肝病\n\n"
            "## 2.2 二线用药\n\n"
            "当二甲双胍单药控制不佳时，可联合：\n"
            "- SGLT2抑制剂（如达格列净）：适合有心血管疾病风险者\n"
            "- GLP-1受体激动剂（如利拉鲁肽）：适合需要减重者\n"
            "- DPP-4抑制剂（如西格列汀）：适合老年患者\n"
        ),
    ),
    Document(
        doc_id="doc_hypertension",
        title="高血压管理指南",
        text=(
            "# 血压分级\n\n"
            "- 正常血压：收缩压 < 120 mmHg 且 舒张压 < 80 mmHg\n"
            "- 正常高值：收缩压 120-139 mmHg 或 舒张压 80-89 mmHg\n"
            "- 1级高血压：收缩压 140-159 mmHg 或 舒张压 90-99 mmHg\n"
            "- 2级高血压：收缩压 160-179 mmHg 或 舒张压 100-109 mmHg\n"
            "- 3级高血压：收缩压 ≥ 180 mmHg 或 舒张压 ≥ 110 mmHg\n\n"
            "# 治疗目标\n\n"
            "一般患者：< 140/90 mmHg\n"
            "合并糖尿病：< 130/80 mmHg\n"
            "老年患者（≥65岁）：< 150/90 mmHg\n\n"
            "# 一线降压药\n\n"
            "1. ACE抑制剂（如依那普利）\n"
            "2. ARB（如缬沙坦）\n"
            "3. 钙通道阻滞剂（如氨氯地平）\n"
            "4. 噻嗪类利尿剂（如氢氯噻嗪）\n"
        ),
    ),
    Document(
        doc_id="doc_interaction",
        title="药物相互作用手册",
        text=(
            "# 二甲双胍相互作用\n\n"
            "## 禁忌联用\n"
            "- 碘造影剂：需停用二甲双胍48小时，检查肾功能后恢复\n"
            "- 大量酒精：增加乳酸酸中毒风险\n\n"
            "## 谨慎联用\n"
            "- ACE抑制剂/ARB：可能影响肾功能，需监测\n"
            "- 利尿剂：可能导致脱水，增加乳酸酸中毒风险\n\n"
            "# SGLT2抑制剂相互作用\n\n"
            "- 利尿剂：增加低血压和脱水风险\n"
            "- 胰岛素/磺脲类：增加低血糖风险，需减量\n"
        ),
    ),
]


RETRIEVAL_TEST_CASES = [
    {
        "id": "R1",
        "question": "糖尿病的诊断标准是什么？",
        "expected_keywords": ["7.0", "11.1", "HbA1c", "6.5%"],
        "expected_doc_ids": ["doc_diabetes"],
        "expected_sections": ["诊断标准"],
    },
    {
        "id": "R2",
        "question": "二甲双胍的起始剂量是多少？",
        "expected_keywords": ["500mg", "每日2次"],
        "expected_doc_ids": ["doc_diabetes"],
    },
    {
        "id": "R3",
        "question": "高血压的分级标准？",
        "expected_keywords": ["正常血压", "正常高值", "1级", "2级", "3级"],
        "expected_doc_ids": ["doc_hypertension"],
    },
    {
        "id": "R4",
        "question": "老年高血压患者的治疗目标？",
        "expected_keywords": ["150/90"],
        "expected_doc_ids": ["doc_hypertension"],
    },
]


CROSS_DOC_TEST_CASES = [
    {
        "id": "C1",
        "question": "糖尿病合并高血压患者的血压控制目标是多少？",
        "expected_keywords": ["130/80"],
        "expected_doc_ids": ["doc_hypertension"],
        "reasoning": "需要理解'合并糖尿病'条件",
    },
    {
        "id": "C2",
        "question": "糖尿病患者使用二甲双胍同时服用ARB需要注意什么？",
        "expected_keywords": ["肾功能", "监测"],
        "expected_doc_ids": ["doc_interaction"],
        "min_citations": 1,
    },
    {
        "id": "C3",
        "question": "老年糖尿病患者的二线用药推荐？",
        "expected_keywords": ["DPP-4"],
        "expected_doc_ids": ["doc_diabetes"],
    },
]


HALLUCINATION_TEST_CASES = [
    {
        "id": "H1",
        "question": "二甲双胍对孕妇安全吗？",
        "expected_behavior": "refuse_or_acknowledge_no_info",
        "forbidden_patterns": [r"安全", r"不安全", r"可以使用", r"禁止使用", r"FDA"],
    },
    {
        "id": "H2",
        "question": "糖尿病的发病机制是什么？",
        "expected_behavior": "refuse_or_acknowledge_no_info",
        "forbidden_patterns": [r"胰岛素抵抗", r"β细胞", r"遗传因素"],
    },
    {
        "id": "H3",
        "question": "推荐哪个品牌的血糖仪？",
        "expected_behavior": "refuse_or_acknowledge_no_info",
        "forbidden_patterns": [r"罗氏", r"雅培", r"强生", r"三诺"],
    },
    {
        "id": "H4",
        "question": "二甲双胍会导致维生素B12缺乏吗？",
        "expected_behavior": "refuse_or_acknowledge_no_info",
        "forbidden_patterns": [r"维生素B12"],
    },
]


BOUNDARY_TEST_CASES = [
    {"id": "E1", "question": "", "expected_behavior": "prompt_for_input"},
    {
        "id": "E2",
        "question": "这是一段超长问题。" * 300,
        "expected_behavior": "truncate_or_warn",
    },
    {
        "id": "E3",
        "question": "今天天气如何？",
        "expected_behavior": "refuse_irrelevant",
    },
    {
        "id": "E4",
        "question": "工作空间无文档时提问",
        "expected_behavior": "warn_no_documents",
    },
    {
        "id": "E5",
        "question": "What is the diagnostic standard for diabetes?",
        "expected_behavior": "cross_language_retrieval",
    },
]


def evaluate_retrieval(test_cases: Iterable[dict], workspace: KnowledgeWorkspace) -> dict:
    """评估检索质量"""
    metrics = {"recall@5": [], "precision@5": [], "mrr": []}

    for case in test_cases:
        results = workspace.retriever.retrieve(query=case["question"], top_k=5)
        retrieved_doc_ids = [getattr(r, "metadata", {}).get("doc_id") for r in results]
        expected_doc_ids = set(case.get("expected_doc_ids", []))

        hits = sum(1 for doc_id in expected_doc_ids if doc_id in retrieved_doc_ids)
        recall = hits / len(expected_doc_ids) if expected_doc_ids else 0
        metrics["recall@5"].append(recall)

        precision = hits / len(retrieved_doc_ids) if retrieved_doc_ids else 0
        metrics["precision@5"].append(precision)

        for index, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in expected_doc_ids:
                metrics["mrr"].append(1 / (index + 1))
                break
        else:
            metrics["mrr"].append(0)

    return {key: sum(values) / len(values) for key, values in metrics.items()}


def verify_citation_accuracy(response: GroundedResponse, workspace: KnowledgeWorkspace) -> dict:
    """验证引用准确性"""
    results = {
        "all_citations_exist": True,
        "content_matches": True,
        "all_sources_cited": True,
        "details": [],
    }

    def get_chunks(ids: List[str]) -> dict:
        vector_store = workspace.vector_store
        if hasattr(vector_store, "get"):
            return vector_store.get(ids=ids)
        return vector_store.collection.get(ids=ids)

    for cite in response.citations:
        exists = get_chunks([cite.chunk_id])
        if not exists.get("ids"):
            results["all_citations_exist"] = False
            results["details"].append(f"Citation {cite.chunk_id} not found")

    cited_in_text = set(re.findall(r"\[Source (\d+)\]", response.answer))
    cited_in_list = {str(i) for i in range(1, len(response.citations) + 1)}

    missing = cited_in_text - cited_in_list
    if missing:
        results["all_sources_cited"] = False
        results["details"].append(f"Missing citations for: {sorted(missing)}")

    return results


def verify_chunk_exists(chunk_id: str, workspace: KnowledgeWorkspace) -> bool:
    vector_store = workspace.vector_store
    if hasattr(vector_store, "get"):
        return bool(vector_store.get(ids=[chunk_id]).get("ids"))
    return bool(vector_store.collection.get(ids=[chunk_id]).get("ids"))


def evaluate_hallucination(test_cases: Iterable[dict], workspace: KnowledgeWorkspace) -> dict:
    """评估幻觉抑制能力"""
    results = {"refusal_rate": 0, "hallucination_rate": 0, "false_citation_rate": 0}

    cases = list(test_cases)
    refused = 0
    hallucinated = 0
    false_citations = 0
    total_citations = 0

    for case in cases:
        response = workspace.query(case["question"])
        refusal_phrases = ["无法回答", "没有相关信息", "资料中未提及", "cannot answer", "no relevant information"]
        is_refused = any(phrase in response.answer for phrase in refusal_phrases)

        if is_refused:
            refused += 1
        else:
            for pattern in case.get("forbidden_patterns", []):
                if re.search(pattern, response.answer):
                    hallucinated += 1
                    break

        for cite in response.citations:
            total_citations += 1
            if not verify_chunk_exists(cite.chunk_id, workspace):
                false_citations += 1

    n = len(cases)
    results["refusal_rate"] = refused / n if n else 0
    results["hallucination_rate"] = hallucinated / n if n else 0
    results["false_citation_rate"] = false_citations / total_citations if total_citations else 0

    return results


class RAGTestSuite:
    """RAG系统测试套件"""

    def __init__(self, workspace: KnowledgeWorkspace):
        self.workspace = workspace
        self.results = {}

    def setup(self) -> None:
        """初始化测试数据"""
        for doc in TEST_DOCUMENTS:
            self.workspace.add_document(doc_id=doc.doc_id, title=doc.title, text=doc.text)

    def run_all_tests(self) -> str:
        """运行全部测试"""
        self.results["retrieval"] = self._run_retrieval_tests()
        self.results["cross_doc"] = self._run_cross_doc_tests()
        self.results["hallucination"] = self._run_hallucination_tests()
        self.results["citation"] = self._run_citation_tests()
        self.results["boundary"] = self._run_boundary_tests()

        return self.generate_report()

    def _run_retrieval_tests(self) -> List[dict]:
        results = []
        for case in RETRIEVAL_TEST_CASES:
            response = self.workspace.query(case["question"])
            keywords_found = sum(1 for kw in case["expected_keywords"] if kw in response.answer)
            keyword_recall = keywords_found / len(case["expected_keywords"])
            cited_docs = {c.doc_id for c in response.citations}
            expected_docs = set(case["expected_doc_ids"])
            doc_hit = bool(cited_docs & expected_docs)

            results.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "keyword_recall": keyword_recall,
                    "correct_doc_cited": doc_hit,
                    "passed": keyword_recall >= 0.5 and doc_hit,
                }
            )

        return results

    def _run_cross_doc_tests(self) -> List[dict]:
        results = []
        for case in CROSS_DOC_TEST_CASES:
            response = self.workspace.query(case["question"])
            keywords_found = sum(1 for kw in case["expected_keywords"] if kw in response.answer)
            keyword_recall = keywords_found / len(case["expected_keywords"])
            cited_docs = {c.doc_id for c in response.citations}
            expected_docs = set(case["expected_doc_ids"])
            doc_hit = bool(cited_docs & expected_docs)
            min_citations = case.get("min_citations", 1)

            results.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "keyword_recall": keyword_recall,
                    "correct_doc_cited": doc_hit,
                    "citation_count": len(response.citations),
                    "passed": keyword_recall >= 0.5 and doc_hit and len(response.citations) >= min_citations,
                }
            )

        return results

    def _run_hallucination_tests(self) -> List[dict]:
        results = []
        for case in HALLUCINATION_TEST_CASES:
            response = self.workspace.query(case["question"])
            refusal_phrases = ["无法回答", "没有相关信息", "资料中未提及"]
            refused = any(phrase in response.answer for phrase in refusal_phrases)

            hallucinated = False
            for pattern in case.get("forbidden_patterns", []):
                if re.search(pattern, response.answer):
                    hallucinated = True
                    break

            passed = refused or (not hallucinated and response.is_grounded)

            results.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "refused": refused,
                    "hallucinated": hallucinated,
                    "is_grounded": response.is_grounded,
                    "passed": passed,
                }
            )

        return results

    def _run_citation_tests(self) -> List[dict]:
        results = []
        for case in RETRIEVAL_TEST_CASES + CROSS_DOC_TEST_CASES:
            response = self.workspace.query(case["question"])
            accuracy = verify_citation_accuracy(response, self.workspace)
            passed = accuracy["all_citations_exist"] and accuracy["all_sources_cited"]

            results.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "passed": passed,
                    "details": accuracy,
                }
            )

        return results

    def _run_boundary_tests(self) -> List[dict]:
        results = []
        for case in BOUNDARY_TEST_CASES:
            response = self.workspace.query(case["question"])
            is_blank = not case["question"].strip()
            refused = any(
                phrase in response.answer
                for phrase in ["无法回答", "没有相关信息", "资料中未提及", "请输入"]
            )

            passed = True
            if is_blank and "请输入" not in response.answer:
                passed = False
            if case["id"] == "E3" and not refused:
                passed = False

            results.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "expected_behavior": case["expected_behavior"],
                    "passed": passed,
                }
            )

        return results

    def generate_report(self) -> str:
        """生成测试报告"""
        report = ["# RAG System Test Report", ""]

        for category, results in self.results.items():
            passed = sum(1 for result in results if result["passed"])
            total = len(results)
            rate = (passed / total * 100) if total else 0

            report.append(f"## {category.title()} Tests")
            report.append(f"**Pass Rate: {passed}/{total} ({rate:.1f}%)**")
            report.append("")
            report.append("| ID | Question | Passed | Details |")
            report.append("|----|---------:|:------:|---------|")

            for result in results:
                status = "✅" if result["passed"] else "❌"
                question = result["question"][:30] + "..." if len(result["question"]) > 30 else result["question"]
                details = str({k: v for k, v in result.items() if k not in ["id", "question", "passed"]})
                report.append(f"| {result['id']} | {question} | {status} | {details[:50]} |")

            report.append("")

        return "\n".join(report)


JUDGE_PROMPT = """
你是一个评估AI回答质量的评委。请评估以下回答：

## 原始问题
{question}

## 可用资料
{sources}

## AI回答
{answer}

请从以下维度评分（1-5分）：

1. **事实准确性**：回答内容是否与资料一致？
2. **引用完整性**：重要观点是否都有引用支持？
3. **幻觉程度**：是否包含资料中没有的信息？（5=无幻觉，1=严重幻觉）
4. **回答有用性**：对用户问题的回答是否有帮助？

请以JSON格式返回：
```json
{
  "factual_accuracy": 4,
  "citation_completeness": 5,
  "hallucination_score": 5,
  "usefulness": 4,
  "explanation": "..."
}
```
""".strip()


def llm_judge_evaluate(question: str, sources: str, answer: str, judge_llm: object) -> dict:
    """使用LLM评估回答质量"""
    prompt = JUDGE_PROMPT.format(question=question, sources=sources, answer=answer)
    result = judge_llm.chat([{"role": "user", "content": prompt}])
    return json.loads(extract_json(result))


def extract_json(raw_text: str) -> str:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response")
    return match.group(0)


if __name__ == "__main__":
    print("This module provides the RAGTestSuite for integration into your workspace.")
