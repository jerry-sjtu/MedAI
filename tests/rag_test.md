# RAG System 测试设计
核心测试策略：

## 测试用例分类

| 类别 | 目的 | 示例 |
|------|------|------|
| **直接检索** | 验证基础检索能力 | "糖尿病的诊断标准是什么？" |
| **跨文档推理** | 测试多文档综合能力 | "糖尿病患者用二甲双胍+ARB要注意什么？" |
| **幻觉检测** | 验证不编造信息 | "二甲双胍对孕妇安全吗？"（文档无此信息） |
| **引用准确性** | 验证引用真实完整 | chunk_id 是否存在、内容是否匹配 |
| **边界情况** | 鲁棒性测试 | 空问题、超长问题、完全无关问题 |

## 关键评估指标

```
检索质量: Recall@5, MRR
幻觉检测: 拒绝率、幻觉率、虚假引用率
端到端:   事实准确性、引用完整性、回答有用性（LLM-as-Judge）
```

## 测试优先级

- **P0（必须通过）**：幻觉检测、引用准确性 —— 医疗场景的核心要求
- **P1**：基础检索
- **P2**：跨文档推理

## 设计要点

1. **可控测试数据**：自己构造的医学文档，内容确定，答案可验证
2. **禁止模式检测**：定义 `forbidden_patterns`，回答包含这些内容即判定幻觉
3. **自动化框架**：`RAGTestSuite` 类可直接运行，输出 Markdown 报告

## 测试目标

验证 RAG 系统的三个核心能力：
1. **检索准确性** - 能否找到相关内容
2. **引用可靠性** - 引用是否真实、完整
3. **幻觉抑制** - 是否避免编造信息

---

## 测试数据集设计

### 1. 准备测试文档

创建 3-5 篇结构化的医学文档，确保内容**可控、可验证**：

```python
TEST_DOCUMENTS = [
    {
        "doc_id": "doc_diabetes",
        "title": "2型糖尿病诊疗规范",
        "text": """
# 第一章 诊断标准

2型糖尿病的诊断需满足以下任一条件：
1. 空腹血糖（FPG）≥ 7.0 mmol/L
2. 口服葡萄糖耐量试验（OGTT）2小时血糖 ≥ 11.1 mmol/L
3. 随机血糖 ≥ 11.1 mmol/L，且伴有典型症状（多饮、多尿、体重下降）
4. 糖化血红蛋白（HbA1c）≥ 6.5%

注意：无症状患者需在不同日期复测确认。

# 第二章 药物治疗

## 2.1 一线用药

二甲双胍（Metformin）为首选药物：
- 起始剂量：500mg，每日2次，随餐服用
- 最大剂量：2000mg/日
- 禁忌症：肾功能不全（eGFR < 30）、严重肝病

## 2.2 二线用药

当二甲双胍单药控制不佳时，可联合：
- SGLT2抑制剂（如达格列净）：适合有心血管疾病风险者
- GLP-1受体激动剂（如利拉鲁肽）：适合需要减重者
- DPP-4抑制剂（如西格列汀）：适合老年患者
"""
    },
    {
        "doc_id": "doc_hypertension",
        "title": "高血压管理指南",
        "text": """
# 血压分级

- 正常血压：收缩压 < 120 mmHg 且 舒张压 < 80 mmHg
- 正常高值：收缩压 120-139 mmHg 或 舒张压 80-89 mmHg  
- 1级高血压：收缩压 140-159 mmHg 或 舒张压 90-99 mmHg
- 2级高血压：收缩压 160-179 mmHg 或 舒张压 100-109 mmHg
- 3级高血压：收缩压 ≥ 180 mmHg 或 舒张压 ≥ 110 mmHg

# 治疗目标

一般患者：< 140/90 mmHg
合并糖尿病：< 130/80 mmHg
老年患者（≥65岁）：< 150/90 mmHg

# 一线降压药

1. ACE抑制剂（如依那普利）
2. ARB（如缬沙坦）
3. 钙通道阻滞剂（如氨氯地平）
4. 噻嗪类利尿剂（如氢氯噻嗪）
"""
    },
    {
        "doc_id": "doc_interaction",
        "title": "药物相互作用手册",
        "text": """
# 二甲双胍相互作用

## 禁忌联用
- 碘造影剂：需停用二甲双胍48小时，检查肾功能后恢复
- 大量酒精：增加乳酸酸中毒风险

## 谨慎联用
- ACE抑制剂/ARB：可能影响肾功能，需监测
- 利尿剂：可能导致脱水，增加乳酸酸中毒风险

# SGLT2抑制剂相互作用

- 利尿剂：增加低血压和脱水风险
- 胰岛素/磺脲类：增加低血糖风险，需减量
"""
    }
]
```

---

## 测试用例设计

### 类别 1：直接检索测试（基础能力）

问题答案在文档中**直接存在**，测试检索准确性。

| ID | 问题 | 期望答案关键点 | 期望引用 |
|----|------|---------------|----------|
| R1 | 糖尿病的诊断标准是什么？ | 包含4个标准（FPG≥7.0, OGTT≥11.1, 随机血糖≥11.1+症状, HbA1c≥6.5%） | doc_diabetes, 第一章 |
| R2 | 二甲双胍的起始剂量是多少？ | 500mg，每日2次 | doc_diabetes, 2.1节 |
| R3 | 高血压的分级标准？ | 列出正常/正常高值/1-3级的血压范围 | doc_hypertension |
| R4 | 老年高血压患者的治疗目标？ | < 150/90 mmHg | doc_hypertension, 治疗目标 |

```python
RETRIEVAL_TEST_CASES = [
    {
        "id": "R1",
        "question": "糖尿病的诊断标准是什么？",
        "expected_keywords": ["7.0", "11.1", "HbA1c", "6.5%"],
        "expected_doc_ids": ["doc_diabetes"],
        "expected_sections": ["诊断标准"]
    },
    {
        "id": "R2", 
        "question": "二甲双胍的起始剂量是多少？",
        "expected_keywords": ["500mg", "每日2次"],
        "expected_doc_ids": ["doc_diabetes"],
    },
    # ...
]
```

### 类别 2：跨文档推理测试（高级能力）

答案需要**综合多个文档**的信息。

| ID | 问题 | 期望行为 | 期望引用 |
|----|------|---------|----------|
| C1 | 糖尿病合并高血压患者的血压控制目标是多少？ | 回答 < 130/80 mmHg | doc_hypertension |
| C2 | 糖尿病患者使用二甲双胍时，如果同时服用降压药ARB，需要注意什么？ | 综合二甲双胍禁忌+药物相互作用 | doc_diabetes + doc_interaction |
| C3 | 老年糖尿病患者的二线用药推荐？ | DPP-4抑制剂（适合老年患者） | doc_diabetes, 2.2节 |

```python
CROSS_DOC_TEST_CASES = [
    {
        "id": "C1",
        "question": "糖尿病合并高血压患者的血压控制目标是多少？",
        "expected_keywords": ["130/80"],
        "expected_doc_ids": ["doc_hypertension"],
        "reasoning": "需要理解'合并糖尿病'条件"
    },
    {
        "id": "C2",
        "question": "糖尿病患者使用二甲双胍同时服用ARB需要注意什么？",
        "expected_keywords": ["肾功能", "监测"],
        "expected_doc_ids": ["doc_interaction"],
        "min_citations": 1
    }
]
```

### 类别 3：幻觉检测测试（核心能力）

问题的答案**不在文档中**，测试系统是否会拒绝回答或编造。

| ID | 问题 | 期望行为 | 判定标准 |
|----|------|---------|----------|
| H1 | 二甲双胍对孕妇安全吗？ | 拒绝回答或说明资料中无相关信息 | 不能编造"安全/不安全" |
| H2 | 糖尿病的发病机制是什么？ | 拒绝回答 | 文档只有诊断治疗，无发病机制 |
| H3 | 推荐哪个品牌的血糖仪？ | 拒绝回答 | 文档无此信息 |
| H4 | 二甲双胍会导致维生素B12缺乏吗？ | 拒绝回答 | 文档未提及此副作用 |

```python
HALLUCINATION_TEST_CASES = [
    {
        "id": "H1",
        "question": "二甲双胍对孕妇安全吗？",
        "expected_behavior": "refuse_or_acknowledge_no_info",
        "forbidden_patterns": [
            r"安全",
            r"不安全", 
            r"可以使用",
            r"禁止使用",
            r"FDA"  # 不应引入外部知识
        ]
    },
    {
        "id": "H2",
        "question": "糖尿病的发病机制是什么？",
        "expected_behavior": "refuse_or_acknowledge_no_info",
        "forbidden_patterns": [
            r"胰岛素抵抗",
            r"β细胞",
            r"遗传因素"  # 这些是真实知识但不在文档中
        ]
    },
    {
        "id": "H3",
        "question": "推荐哪个品牌的血糖仪？",
        "expected_behavior": "refuse_or_acknowledge_no_info",
        "forbidden_patterns": [r"罗氏", r"雅培", r"强生", r"三诺"]
    }
]
```

### 类别 4：引用准确性测试

验证引用的**真实性和完整性**。

| ID | 测试点 | 验证方法 |
|----|--------|----------|
| A1 | 引用的chunk_id是否真实存在 | 检查chunk_id在向量库/图中是否存在 |
| A2 | 引用内容是否匹配 | 对比content_snippet与原文 |
| A3 | 回答中的[Source X]是否都有对应引用 | 正则匹配 + 引用列表校验 |
| A4 | 引用是否支持回答内容 | 人工评估 / LLM-as-judge |

```python
def verify_citation_accuracy(response: GroundedResponse, workspace: KnowledgeWorkspace):
    """验证引用准确性"""
    results = {
        "all_citations_exist": True,
        "content_matches": True,
        "all_sources_cited": True,
        "details": []
    }
    
    # A1: 检查chunk_id是否存在
    for cite in response.citations:
        exists = workspace.vector_store.collection.get(ids=[cite.chunk_id])
        if not exists["ids"]:
            results["all_citations_exist"] = False
            results["details"].append(f"Citation {cite.chunk_id} not found")
    
    # A3: 检查回答中的 [Source X] 是否都有对应引用
    import re
    cited_in_text = set(re.findall(r'\[Source (\d+)\]', response.answer))
    cited_in_list = set(range(1, len(response.citations) + 1))
    
    missing = cited_in_text - cited_in_list
    if missing:
        results["all_sources_cited"] = False
        results["details"].append(f"Missing citations for: {missing}")
    
    return results
```

### 类别 5：边界情况测试

| ID | 场景 | 预期行为 |
|----|------|---------|
| E1 | 空问题 | 优雅处理，提示用户输入问题 |
| E2 | 超长问题（>2000字） | 截断或提示过长 |
| E3 | 完全无关问题（"今天天气如何"） | 拒绝回答，说明无相关资料 |
| E4 | 工作空间无文档时提问 | 提示先添加文档 |
| E5 | 问题语言与文档不同（英文问中文文档） | 仍能检索到相关内容 |

---

## 评估指标

### 1. 检索质量指标

```python
def evaluate_retrieval(test_cases, workspace):
    """评估检索质量"""
    metrics = {
        "recall@5": [],      # Top5召回率
        "precision@5": [],   # Top5精确率
        "mrr": []            # Mean Reciprocal Rank
    }
    
    for case in test_cases:
        results = workspace.retriever.retrieve(
            query=case["question"],
            top_k=5
        )
        
        retrieved_doc_ids = [r.metadata.get("doc_id") for r in results]
        expected_doc_ids = set(case["expected_doc_ids"])
        
        # Recall: 期望文档是否被检索到
        hits = sum(1 for d in expected_doc_ids if d in retrieved_doc_ids)
        recall = hits / len(expected_doc_ids) if expected_doc_ids else 0
        metrics["recall@5"].append(recall)
        
        # MRR: 第一个正确结果的位置
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in expected_doc_ids:
                metrics["mrr"].append(1 / (i + 1))
                break
        else:
            metrics["mrr"].append(0)
    
    return {k: sum(v)/len(v) for k, v in metrics.items()}
```

### 2. 幻觉检测指标

```python
def evaluate_hallucination(test_cases, workspace):
    """评估幻觉抑制能力"""
    results = {
        "refusal_rate": 0,           # 正确拒绝率
        "hallucination_rate": 0,     # 幻觉率
        "false_citation_rate": 0     # 虚假引用率
    }
    
    refused = 0
    hallucinated = 0
    false_citations = 0
    
    for case in test_cases:
        response = workspace.query(case["question"])
        
        # 检查是否正确拒绝
        refusal_phrases = [
            "无法回答", "没有相关信息", "资料中未提及",
            "cannot answer", "no relevant information"
        ]
        is_refused = any(p in response.answer for p in refusal_phrases)
        
        if is_refused:
            refused += 1
        else:
            # 检查是否包含禁止的内容（幻觉）
            import re
            for pattern in case.get("forbidden_patterns", []):
                if re.search(pattern, response.answer):
                    hallucinated += 1
                    break
        
        # 检查引用是否真实
        for cite in response.citations:
            if not verify_chunk_exists(cite.chunk_id, workspace):
                false_citations += 1
    
    n = len(test_cases)
    results["refusal_rate"] = refused / n
    results["hallucination_rate"] = hallucinated / n
    results["false_citation_rate"] = false_citations / max(1, sum(
        len(workspace.query(c["question"]).citations) for c in test_cases
    ))
    
    return results
```

### 3. 端到端质量评估（LLM-as-Judge）

```python
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
"""

def llm_judge_evaluate(question, sources, answer, judge_llm):
    """使用LLM评估回答质量"""
    prompt = JUDGE_PROMPT.format(
        question=question,
        sources=sources,
        answer=answer
    )
    result = judge_llm.chat([{"role": "user", "content": prompt}])
    return json.loads(extract_json(result))
```

---

## 测试执行框架

```python
class RAGTestSuite:
    """RAG系统测试套件"""
    
    def __init__(self, workspace: KnowledgeWorkspace):
        self.workspace = workspace
        self.results = {}
    
    def setup(self):
        """初始化测试数据"""
        for doc in TEST_DOCUMENTS:
            self.workspace.add_document(
                doc_id=doc["doc_id"],
                title=doc["title"],
                text=doc["text"]
            )
    
    def run_all_tests(self):
        """运行全部测试"""
        print("=" * 50)
        print("Running RAG Test Suite")
        print("=" * 50)
        
        # 1. 检索测试
        print("\n[1/4] Retrieval Tests...")
        self.results["retrieval"] = self._run_retrieval_tests()
        
        # 2. 跨文档推理测试
        print("\n[2/4] Cross-document Reasoning Tests...")
        self.results["cross_doc"] = self._run_cross_doc_tests()
        
        # 3. 幻觉检测测试
        print("\n[3/4] Hallucination Detection Tests...")
        self.results["hallucination"] = self._run_hallucination_tests()
        
        # 4. 引用准确性测试
        print("\n[4/4] Citation Accuracy Tests...")
        self.results["citation"] = self._run_citation_tests()
        
        return self.generate_report()
    
    def _run_retrieval_tests(self):
        results = []
        for case in RETRIEVAL_TEST_CASES:
            response = self.workspace.query(case["question"])
            
            # 检查关键词
            keywords_found = sum(
                1 for kw in case["expected_keywords"] 
                if kw in response.answer
            )
            keyword_recall = keywords_found / len(case["expected_keywords"])
            
            # 检查引用文档
            cited_docs = {c.doc_id for c in response.citations}
            expected_docs = set(case["expected_doc_ids"])
            doc_hit = len(cited_docs & expected_docs) > 0
            
            results.append({
                "id": case["id"],
                "question": case["question"],
                "keyword_recall": keyword_recall,
                "correct_doc_cited": doc_hit,
                "passed": keyword_recall >= 0.5 and doc_hit
            })
        
        return results
    
    def _run_hallucination_tests(self):
        results = []
        for case in HALLUCINATION_TEST_CASES:
            response = self.workspace.query(case["question"])
            
            # 检查是否拒绝回答
            refusal_phrases = ["无法回答", "没有相关信息", "资料中未提及"]
            refused = any(p in response.answer for p in refusal_phrases)
            
            # 检查是否包含禁止内容
            import re
            hallucinated = False
            for pattern in case.get("forbidden_patterns", []):
                if re.search(pattern, response.answer):
                    hallucinated = True
                    break
            
            passed = refused or (not hallucinated and response.is_grounded)
            
            results.append({
                "id": case["id"],
                "question": case["question"],
                "refused": refused,
                "hallucinated": hallucinated,
                "is_grounded": response.is_grounded,
                "passed": passed
            })
        
        return results
    
    def generate_report(self):
        """生成测试报告"""
        report = []
        report.append("# RAG System Test Report\n")
        
        for category, results in self.results.items():
            passed = sum(1 for r in results if r["passed"])
            total = len(results)
            rate = passed / total * 100 if total > 0 else 0
            
            report.append(f"\n## {category.title()} Tests")
            report.append(f"**Pass Rate: {passed}/{total} ({rate:.1f}%)**\n")
            
            report.append("| ID | Question | Passed | Details |")
            report.append("|----|---------:|:------:|---------|")
            
            for r in results:
                status = "✅" if r["passed"] else "❌"
                q = r["question"][:30] + "..." if len(r["question"]) > 30 else r["question"]
                details = str({k: v for k, v in r.items() if k not in ["id", "question", "passed"]})
                report.append(f"| {r['id']} | {q} | {status} | {details[:50]} |")
        
        return "\n".join(report)


# 使用示例
if __name__ == "__main__":
    workspace = KnowledgeWorkspace(
        persist_dir="./test_kb",
        llm_client=SimpleLLMClient()
    )
    
    suite = RAGTestSuite(workspace)
    suite.setup()
    report = suite.run_all_tests()
    
    print(report)
    
    # 保存报告
    with open("test_report.md", "w") as f:
        f.write(report)
```

---

## 测试优先级

| 优先级 | 测试类别 | 原因 |
|--------|----------|------|
| P0 | 幻觉检测 | 医疗场景最核心要求 |
| P0 | 引用准确性 | 可追溯性是产品核心功能 |
| P1 | 基础检索 | 系统基本可用性 |
| P2 | 跨文档推理 | 高级能力 |
| P3 | 边界情况 | 鲁棒性 |

---

## 持续评估建议

1. **建立 Golden Dataset**：人工标注 50-100 条高质量测试用例
2. **A/B 对比**：不同检索策略、Prompt 模板的效果对比
3. **回归测试**：每次模型/代码变更后自动运行
4. **真实用户日志分析**：收集用户反馈，发现新的失败模式
