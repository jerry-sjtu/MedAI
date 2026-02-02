# Medical Agent RAG Architecture Design

## 设计目标

实现类似 NotebookLM 的知识工作空间功能：
1. **严格引用约束**：模型只能基于给定文档回答，不允许"编造背景知识"
2. **可追溯引用**：每个回答都能追溯到原文具体段落
3. **轻量级实现**：适合科研项目，专注数据分析和实验

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 向量检索 | Chroma | 轻量级向量数据库 |
| 图数据 | NetworkX | 纯Python图库，无需外部服务 |
| LLM | OpenAI/Claude API | 可配置 |
| 嵌入模型 | text-embedding-3-small / BGE-M3 | 支持中文医学文本 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Knowledge Workspace                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │   Document   │     │   Indexing   │     │   Storage    │        │
│  │   Ingestion  │────▶│   Pipeline   │────▶│    Layer     │        │
│  └──────────────┘     └──────────────┘     └──────────────┘        │
│         │                    │                    │                 │
│         ▼                    ▼                    ▼                 │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                    Retrieval Engine                       │      │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │      │
│  │  │  Vector    │  │   Graph    │  │  Hybrid Fusion     │  │      │
│  │  │  Search    │  │  Traverse  │  │  + Re-ranking      │  │      │
│  │  └────────────┘  └────────────┘  └────────────────────┘  │      │
│  └──────────────────────────────────────────────────────────┘      │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │              Grounded Generation Layer                    │      │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │      │
│  │  │  Context   │  │  Citation  │  │  Hallucination     │  │      │
│  │  │  Assembly  │  │  Tracking  │  │  Prevention        │  │      │
│  │  └────────────┘  └────────────┘  └────────────────────┘  │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 核心模块设计

### 1. 数据模型

```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid

@dataclass
class DocumentChunk:
    """文档块 - 最小检索单元"""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""                    # 所属文档ID
    content: str = ""                   # 文本内容
    chunk_index: int = 0                # 在文档中的位置索引
    
    # 元数据
    source_title: str = ""              # 文档标题
    section_title: str = ""             # 章节标题
    page_number: Optional[int] = None   # 页码
    image_urls: list[str] = field(default_factory=list)  # 关联图片
    
    # 语义信息
    entities: list[str] = field(default_factory=list)    # 抽取的实体
    keywords: list[str] = field(default_factory=list)    # 关键词
    summary: str = ""                   # 块摘要（可选）
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)


@dataclass 
class CitedSource:
    """引用来源 - 用于追溯回答的出处"""
    chunk_id: str
    doc_id: str
    source_title: str
    section_title: str
    content_snippet: str               # 原文片段（用于展示）
    relevance_score: float             # 相关性分数
    page_number: Optional[int] = None


@dataclass
class GroundedResponse:
    """带引用的回答"""
    answer: str                         # 生成的回答
    citations: list[CitedSource]        # 引用来源列表
    confidence: float                   # 置信度
    is_grounded: bool                   # 是否完全基于文档
```

### 2. 知识图谱模块 (NetworkX)

```python
import networkx as nx
from typing import Literal

class KnowledgeGraph:
    """基于NetworkX的轻量级知识图谱"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_document(self, doc_id: str, title: str, metadata: dict):
        """添加文档节点"""
        self.graph.add_node(
            doc_id,
            node_type="document",
            title=title,
            **metadata
        )
    
    def add_chunk(self, chunk: DocumentChunk):
        """添加文档块节点，并建立与文档的关系"""
        self.graph.add_node(
            chunk.chunk_id,
            node_type="chunk",
            content=chunk.content,
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index
        )
        # 文档 -> 块
        self.graph.add_edge(
            chunk.doc_id, 
            chunk.chunk_id, 
            relation="contains"
        )
        # 块 -> 块（相邻关系）
        if chunk.chunk_index > 0:
            prev_chunks = [
                n for n, d in self.graph.nodes(data=True)
                if d.get("doc_id") == chunk.doc_id 
                and d.get("chunk_index") == chunk.chunk_index - 1
            ]
            for prev_chunk in prev_chunks:
                self.graph.add_edge(
                    prev_chunk, 
                    chunk.chunk_id, 
                    relation="next"
                )
    
    def add_entity(self, entity: str, entity_type: str):
        """添加实体节点"""
        entity_id = f"entity:{entity}"
        self.graph.add_node(
            entity_id,
            node_type="entity",
            name=entity,
            entity_type=entity_type
        )
        return entity_id
    
    def link_chunk_entity(self, chunk_id: str, entity_id: str):
        """建立块与实体的关系"""
        self.graph.add_edge(
            chunk_id, 
            entity_id, 
            relation="mentions"
        )
    
    def get_related_chunks(
        self, 
        chunk_ids: list[str], 
        max_hops: int = 2,
        max_results: int = 10
    ) -> list[str]:
        """基于图遍历获取相关块（用于扩展检索上下文）"""
        related = set()
        
        for chunk_id in chunk_ids:
            if chunk_id not in self.graph:
                continue
            # BFS遍历
            for neighbor in nx.single_source_shortest_path_length(
                self.graph, chunk_id, cutoff=max_hops
            ):
                node_data = self.graph.nodes.get(neighbor, {})
                if node_data.get("node_type") == "chunk":
                    related.add(neighbor)
        
        # 移除输入的块
        related -= set(chunk_ids)
        return list(related)[:max_results]
    
    def get_context_window(
        self, 
        chunk_id: str, 
        window_size: int = 1
    ) -> list[str]:
        """获取上下文窗口（前后相邻的块）"""
        node_data = self.graph.nodes.get(chunk_id, {})
        if not node_data:
            return []
        
        doc_id = node_data.get("doc_id")
        chunk_index = node_data.get("chunk_index", 0)
        
        # 找同文档的相邻块
        context_chunks = []
        for n, d in self.graph.nodes(data=True):
            if (d.get("node_type") == "chunk" 
                and d.get("doc_id") == doc_id):
                idx = d.get("chunk_index", 0)
                if abs(idx - chunk_index) <= window_size and n != chunk_id:
                    context_chunks.append((idx, n))
        
        # 按索引排序
        context_chunks.sort(key=lambda x: x[0])
        return [c[1] for c in context_chunks]

    def save(self, path: str):
        """持久化图"""
        nx.write_graphml(self.graph, path)
    
    def load(self, path: str):
        """加载图"""
        self.graph = nx.read_graphml(path)
```

### 3. 向量索引模块 (Chroma)

```python
import chromadb
from chromadb.config import Settings
import hashlib

class VectorStore:
    """基于Chroma的向量存储"""
    
    def __init__(
        self, 
        persist_dir: str = "./chroma_db",
        collection_name: str = "medical_knowledge"
    ):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_fn = self._get_embedding_fn()
    
    def _get_embedding_fn(self):
        """获取嵌入函数 - 可替换为其他模型"""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('BAAI/bge-m3')  # 支持中文
        return lambda texts: model.encode(texts).tolist()
    
    def add_chunks(self, chunks: list[DocumentChunk]):
        """批量添加文档块"""
        if not chunks:
            return
        
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        embeddings = self.embedding_fn(documents)
        metadatas = [
            {
                "doc_id": c.doc_id,
                "source_title": c.source_title,
                "section_title": c.section_title,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number or -1,
            }
            for c in chunks
        ]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def search(
        self, 
        query: str, 
        n_results: int = 10,
        filter_doc_ids: list[str] = None
    ) -> list[tuple[str, float, dict]]:
        """向量检索
        
        Returns:
            list of (chunk_id, score, metadata)
        """
        query_embedding = self.embedding_fn([query])[0]
        
        where_filter = None
        if filter_doc_ids:
            where_filter = {"doc_id": {"$in": filter_doc_ids}}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        output = []
        for i, chunk_id in enumerate(results["ids"][0]):
            # Chroma返回的是距离，转换为相似度
            score = 1 - results["distances"][0][i]
            metadata = results["metadatas"][0][i]
            metadata["content"] = results["documents"][0][i]
            output.append((chunk_id, score, metadata))
        
        return output
    
    def delete_document(self, doc_id: str):
        """删除文档的所有块"""
        self.collection.delete(where={"doc_id": doc_id})
```

### 4. 混合检索引擎

```python
from typing import NamedTuple

class RetrievalResult(NamedTuple):
    chunk_id: str
    content: str
    score: float
    metadata: dict
    retrieval_method: str  # "vector" | "graph" | "hybrid"


class HybridRetriever:
    """混合检索器：结合向量检索 + 图遍历"""
    
    def __init__(
        self, 
        vector_store: VectorStore, 
        knowledge_graph: KnowledgeGraph
    ):
        self.vector_store = vector_store
        self.kg = knowledge_graph
    
    def retrieve(
        self,
        query: str,
        doc_ids: list[str] = None,      # 限定在特定文档范围内
        top_k: int = 5,
        expand_context: bool = True,     # 是否扩展上下文
        context_window: int = 1          # 上下文窗口大小
    ) -> list[RetrievalResult]:
        """混合检索流程"""
        
        # Step 1: 向量检索获取候选
        vector_results = self.vector_store.search(
            query=query,
            n_results=top_k * 2,  # 多取一些用于后续过滤
            filter_doc_ids=doc_ids
        )
        
        # Step 2: 图扩展（获取相关上下文）
        candidate_chunk_ids = [r[0] for r in vector_results[:top_k]]
        
        expanded_chunks = set()
        if expand_context:
            # 2.1 获取相邻块（上下文窗口）
            for chunk_id in candidate_chunk_ids:
                context = self.kg.get_context_window(
                    chunk_id, 
                    window_size=context_window
                )
                expanded_chunks.update(context)
            
            # 2.2 获取通过实体关联的块
            related = self.kg.get_related_chunks(
                candidate_chunk_ids, 
                max_hops=2,
                max_results=5
            )
            expanded_chunks.update(related)
        
        # Step 3: 合并结果
        results = []
        seen_ids = set()
        
        # 先添加向量检索结果
        for chunk_id, score, metadata in vector_results[:top_k]:
            if chunk_id not in seen_ids:
                results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    content=metadata.get("content", ""),
                    score=score,
                    metadata=metadata,
                    retrieval_method="vector"
                ))
                seen_ids.add(chunk_id)
        
        # 添加图扩展结果（分数稍低）
        for chunk_id in expanded_chunks:
            if chunk_id not in seen_ids:
                node_data = self.kg.graph.nodes.get(chunk_id, {})
                if node_data:
                    results.append(RetrievalResult(
                        chunk_id=chunk_id,
                        content=node_data.get("content", ""),
                        score=0.5,  # 固定分数或可通过重排序调整
                        metadata={
                            "doc_id": node_data.get("doc_id"),
                            "chunk_index": node_data.get("chunk_index")
                        },
                        retrieval_method="graph"
                    ))
                    seen_ids.add(chunk_id)
        
        return results
```

### 5. Grounded Generation（核心：防止幻觉 + 引用追踪）

```python
import json
from string import Template

class GroundedGenerator:
    """基于检索结果的受限生成器"""
    
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

    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLM客户端，需要实现 chat(messages) 方法
        """
        self.llm = llm_client
    
    def _build_source_context(
        self, 
        retrieval_results: list[RetrievalResult]
    ) -> tuple[str, dict[int, RetrievalResult]]:
        """构建来源上下文文本"""
        source_map = {}
        context_parts = []
        
        for i, result in enumerate(retrieval_results, 1):
            source_map[i] = result
            context_parts.append(
                f"[Source {i}] "
                f"Title: {result.metadata.get('source_title', 'Unknown')} | "
                f"Section: {result.metadata.get('section_title', 'N/A')}\n"
                f"Content: {result.content}\n"
            )
        
        return "\n".join(context_parts), source_map
    
    def generate(
        self,
        query: str,
        retrieval_results: list[RetrievalResult],
        conversation_history: list[dict] = None
    ) -> GroundedResponse:
        """生成带引用的回答"""
        
        # 构建上下文
        source_context, source_map = self._build_source_context(
            retrieval_results
        )
        
        # 构建消息
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        
        # 添加历史对话（如果有）
        if conversation_history:
            messages.extend(conversation_history[-6:])  # 保留最近3轮
        
        # 用户消息：包含来源和问题
        user_message = f"""
## Available Sources

{source_context}

## User Question

{query}

{self.RESPONSE_FORMAT_PROMPT}
"""
        messages.append({"role": "user", "content": user_message})
        
        # 调用LLM
        response_text = self.llm.chat(messages)
        
        # 解析响应
        return self._parse_response(response_text, source_map)
    
    def _parse_response(
        self, 
        response_text: str, 
        source_map: dict[int, RetrievalResult]
    ) -> GroundedResponse:
        """解析LLM响应，提取引用信息"""
        
        # 分离回答和JSON元数据
        answer = response_text
        citations = []
        confidence = 0.8
        is_grounded = True
        
        # 尝试提取JSON块
        if "```json" in response_text:
            parts = response_text.split("```json")
            answer = parts[0].strip()
            try:
                json_str = parts[1].split("```")[0].strip()
                meta = json.loads(json_str)
                confidence = meta.get("confidence", 0.8)
                is_grounded = meta.get("is_fully_grounded", True)
                
                # 构建引用列表
                for source_num in meta.get("citations_used", []):
                    if source_num in source_map:
                        result = source_map[source_num]
                        citations.append(CitedSource(
                            chunk_id=result.chunk_id,
                            doc_id=result.metadata.get("doc_id", ""),
                            source_title=result.metadata.get("source_title", ""),
                            section_title=result.metadata.get("section_title", ""),
                            content_snippet=result.content[:200] + "...",
                            relevance_score=result.score,
                            page_number=result.metadata.get("page_number")
                        ))
            except (json.JSONDecodeError, IndexError):
                # JSON解析失败，尝试从文本中提取引用
                pass
        
        # 如果JSON解析失败，从文本中提取 [Source X] 引用
        if not citations:
            import re
            cited_nums = set(re.findall(r'\[Source (\d+)\]', answer))
            for num_str in cited_nums:
                source_num = int(num_str)
                if source_num in source_map:
                    result = source_map[source_num]
                    citations.append(CitedSource(
                        chunk_id=result.chunk_id,
                        doc_id=result.metadata.get("doc_id", ""),
                        source_title=result.metadata.get("source_title", ""),
                        section_title=result.metadata.get("section_title", ""),
                        content_snippet=result.content[:200] + "...",
                        relevance_score=result.score,
                        page_number=result.metadata.get("page_number")
                    ))
        
        return GroundedResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            is_grounded=is_grounded
        )
```

---

## 6. 文档处理 Pipeline

```python
from typing import Iterator
import re

class DocumentProcessor:
    """文档处理器 - 将预抽取的文本转换为可索引的块"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        knowledge_graph: KnowledgeGraph = None,
        vector_store: VectorStore = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kg = knowledge_graph
        self.vector_store = vector_store
        self.entity_extractor = self._get_entity_extractor()
    
    def _get_entity_extractor(self):
        """获取医学实体抽取器（可替换为专业NER模型）"""
        # 简化版：使用规则或调用LLM
        def extract_entities(text: str) -> list[tuple[str, str]]:
            # 返回 [(entity_name, entity_type), ...]
            # 实际可用: scispacy, medspacy, 或 LLM
            entities = []
            # 简单规则示例
            drug_pattern = r'([A-Z][a-z]+(?:mab|nib|pril|olol|statin))'
            for match in re.finditer(drug_pattern, text):
                entities.append((match.group(1), "Drug"))
            return entities
        return extract_entities
    
    def chunk_text(
        self, 
        text: str, 
        doc_id: str,
        source_title: str,
        section_titles: list[str] = None
    ) -> Iterator[DocumentChunk]:
        """将文本切分为块"""
        
        # 按段落切分
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_section = section_titles[0] if section_titles else ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 检测章节标题
            if para.startswith('#') or para.isupper():
                current_section = para.lstrip('#').strip()
                continue
            
            # 累积文本
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                # 输出当前块
                if current_chunk:
                    yield DocumentChunk(
                        doc_id=doc_id,
                        content=current_chunk.strip(),
                        chunk_index=chunk_index,
                        source_title=source_title,
                        section_title=current_section,
                        entities=[e[0] for e in self.entity_extractor(current_chunk)]
                    )
                    chunk_index += 1
                
                # 保留overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                current_chunk = overlap_text + para + "\n\n"
        
        # 最后一个块
        if current_chunk.strip():
            yield DocumentChunk(
                doc_id=doc_id,
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                source_title=source_title,
                section_title=current_section,
                entities=[e[0] for e in self.entity_extractor(current_chunk)]
            )
    
    def process_document(
        self,
        doc_id: str,
        title: str,
        text: str,
        metadata: dict = None,
        image_urls: list[str] = None
    ):
        """处理单个文档：切块 -> 抽取实体 -> 存入图和向量库"""
        
        metadata = metadata or {}
        
        # 添加文档到图
        self.kg.add_document(doc_id, title, metadata)
        
        # 切块并处理
        chunks = list(self.chunk_text(text, doc_id, title))
        
        # 添加到知识图谱
        for chunk in chunks:
            self.kg.add_chunk(chunk)
            
            # 添加实体并建立关联
            for entity in chunk.entities:
                entity_id = self.kg.add_entity(entity, "medical_entity")
                self.kg.link_chunk_entity(chunk.chunk_id, entity_id)
        
        # 添加到向量存储
        self.vector_store.add_chunks(chunks)
        
        return len(chunks)
```

---

## 7. 统一接口：Knowledge Workspace

```python
class KnowledgeWorkspace:
    """知识工作空间 - 统一入口"""
    
    def __init__(
        self,
        persist_dir: str = "./knowledge_base",
        llm_client = None
    ):
        self.persist_dir = persist_dir
        
        # 初始化组件
        self.vector_store = VectorStore(
            persist_dir=f"{persist_dir}/chroma"
        )
        self.knowledge_graph = KnowledgeGraph()
        self.retriever = HybridRetriever(
            self.vector_store, 
            self.knowledge_graph
        )
        self.generator = GroundedGenerator(llm_client)
        self.processor = DocumentProcessor(
            knowledge_graph=self.knowledge_graph,
            vector_store=self.vector_store
        )
        
        # 当前工作空间的文档ID集合
        self.active_doc_ids: set[str] = set()
    
    def add_document(
        self,
        doc_id: str,
        title: str,
        text: str,
        metadata: dict = None
    ) -> int:
        """添加文档到工作空间"""
        num_chunks = self.processor.process_document(
            doc_id=doc_id,
            title=title,
            text=text,
            metadata=metadata
        )
        self.active_doc_ids.add(doc_id)
        return num_chunks
    
    def remove_document(self, doc_id: str):
        """从工作空间移除文档"""
        self.vector_store.delete_document(doc_id)
        # 图中节点可保留（或实现删除逻辑）
        self.active_doc_ids.discard(doc_id)
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        conversation_history: list[dict] = None
    ) -> GroundedResponse:
        """提问 - 只基于工作空间内的文档回答"""
        
        # 检索（限定在当前工作空间的文档）
        results = self.retriever.retrieve(
            query=question,
            doc_ids=list(self.active_doc_ids) if self.active_doc_ids else None,
            top_k=top_k,
            expand_context=True
        )
        
        # 生成带引用的回答
        response = self.generator.generate(
            query=question,
            retrieval_results=results,
            conversation_history=conversation_history
        )
        
        return response
    
    def save(self):
        """持久化工作空间状态"""
        import os
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # 保存图
        self.knowledge_graph.save(f"{self.persist_dir}/graph.graphml")
        
        # 保存活跃文档ID
        with open(f"{self.persist_dir}/active_docs.json", "w") as f:
            json.dump(list(self.active_doc_ids), f)
    
    def load(self):
        """加载工作空间状态"""
        import os
        
        graph_path = f"{self.persist_dir}/graph.graphml"
        if os.path.exists(graph_path):
            self.knowledge_graph.load(graph_path)
        
        docs_path = f"{self.persist_dir}/active_docs.json"
        if os.path.exists(docs_path):
            with open(docs_path) as f:
                self.active_doc_ids = set(json.load(f))
```

---

## 使用示例

```python
# 初始化
from openai import OpenAI

class SimpleLLMClient:
    def __init__(self):
        self.client = OpenAI()
    
    def chat(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content

# 创建工作空间
workspace = KnowledgeWorkspace(
    persist_dir="./medical_kb",
    llm_client=SimpleLLMClient()
)

# 添加文档（假设文本已从PDF抽取）
workspace.add_document(
    doc_id="paper_001",
    title="糖尿病治疗指南2024",
    text="""
    第一章 诊断标准
    
    糖尿病的诊断基于以下标准：
    1. 空腹血糖 ≥ 7.0 mmol/L
    2. OGTT 2小时血糖 ≥ 11.1 mmol/L
    3. 随机血糖 ≥ 11.1 mmol/L 且有典型症状
    4. HbA1c ≥ 6.5%
    
    第二章 治疗方案
    
    一线药物为二甲双胍（Metformin），起始剂量500mg，每日两次...
    """,
    metadata={"type": "guideline", "year": 2024}
)

# 提问
response = workspace.query("糖尿病的诊断标准是什么？")

print(f"回答：{response.answer}")
print(f"\n引用来源：")
for cite in response.citations:
    print(f"  - {cite.source_title} / {cite.section_title}")
    print(f"    原文：{cite.content_snippet}")
print(f"\n置信度：{response.confidence}")
print(f"完全基于文档：{response.is_grounded}")
```

---

## 关键设计决策

### 1. 为什么用 NetworkX 而不是 Neo4j？

| 维度 | NetworkX | Neo4j |
|------|----------|-------|
| 部署复杂度 | 无需外部服务，纯Python | 需要运行数据库服务 |
| 适用规模 | 中小规模（<100万节点） | 大规模图 |
| 查询性能 | 内存计算，简单查询快 | 复杂图查询优化 |
| 持久化 | GraphML/Pickle | 原生持久化 |
| **适合科研** | ✅ 快速迭代，易于调试 | 过重 |

### 2. 防止幻觉的多层机制

```
┌─────────────────────────────────────────────────┐
│  Layer 1: 检索限定                              │
│  - 只在给定文档范围内检索                        │
│  - 没有相关内容就不返回                          │
├─────────────────────────────────────────────────┤
│  Layer 2: Prompt 约束                           │
│  - System prompt 明确禁止使用外部知识            │
│  - 要求每个观点都标注来源                        │
├─────────────────────────────────────────────────┤
│  Layer 3: 输出校验                              │
│  - 检查引用是否真实存在                          │
│  - 置信度自评估                                 │
├─────────────────────────────────────────────────┤
│  Layer 4: 可追溯性                              │
│  - 返回完整引用链                               │
│  - 用户可点击查看原文                            │
└─────────────────────────────────────────────────┘
```

### 3. 图结构设计

```
Document ──contains──▶ Chunk ──mentions──▶ Entity
    │                    │
    │                    ├──next──▶ Chunk (相邻)
    │                    │
    └──similar_to──▶ Document (可选：文档相似性)
```

---

## 扩展方向

1. **多模态支持**：将图片URL关联到块，检索时返回相关图片
2. **对话管理**：增加多轮对话的上下文管理
3. **实体链接**：将抽取的实体链接到医学知识库（如UMLS）
4. **重排序模型**：使用交叉编码器对检索结果精排
5. **增量更新**：支持文档的增量修改而非全量重建

---

## 依赖安装

```bash
pip install chromadb networkx sentence-transformers openai
```

## 目录结构

```
medical_agent/
├── __init__.py
├── models.py          # 数据模型
├── knowledge_graph.py # NetworkX图模块
├── vector_store.py    # Chroma向量存储
├── retriever.py       # 混合检索器
├── generator.py       # Grounded生成器
├── processor.py       # 文档处理器
├── workspace.py       # 统一入口
└── utils/
    ├── entity_extraction.py
    └── text_utils.py
```
