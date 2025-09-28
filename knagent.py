#!/usr/bin/env python3
"""
KnDAgent - Knowledge Graph Agent
Based on DeepSeek model and RAG technology, recommends the most suitable knowledge graph for medical use cases
"""

import os
import json
import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from deepseek_api_client import DeepSeekAPIClient
from config import (
    MODEL_CONFIG, GENERATION_CONFIG, RAG_CONFIG, 
    DOC_CONFIG, KNOWLEDGE_GRAPH_CONFIG, EMERGENCY_KNOWLEDGE_CONFIG,
    DECISION_SUPPORT_TOOLS_CONFIG, PROMPT_TEMPLATES, LOGGING_CONFIG
)

# Add BERT related imports
try:
    from transformers import BertTokenizer, BertModel
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: transformers library not installed, will use fallback solution")


@dataclass
class KnowledgeGraph:
    """Knowledge graph information"""
    name: str
    description: str
    domain: str
    advantages: List[str]
    limitations: List[str]
    use_cases: List[str]
    data_sources: List[str]


@dataclass
class UseCase:
    """Medical use case information"""
    name: str
    description: str
    domain: str
    requirements: List[str]
    challenges: List[str]


@dataclass
class TaskInformation:
    """Task information"""
    task_id: str
    task_description: str
    task_type: str
    priority: str
    requirements: List[str]
    context: str
    extended_info: str


@dataclass
class EmergencyKnowledge:
    """Emergency knowledge entry"""
    id: str
    title: str
    content: str
    keywords: List[str]
    category: str


@dataclass
class DecisionSupportTool:
    """Decision support tool"""
    id: str
    name: str
    description: str
    input_requirements: List[str]
    output_format: str
    applicable_scenarios: List[str]
    accuracy: str
    processing_time: str


@dataclass
class ExecutionChain:
    """Task-graph-tool execution chain"""
    task_info: TaskInformation
    selected_graph: str
    selected_tools: List[str]
    execution_strategy: str
    confidence_score: float


class DocumentProcessor:
    """Document processor"""
    
    def __init__(self, doc_path: str):
        self.doc_path = doc_path
        self.content = ""
        self.sections = {}
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=getattr(logging, LOGGING_CONFIG["level"]),
            format=LOGGING_CONFIG["format"],
            handlers=[
                logging.FileHandler(LOGGING_CONFIG["file"]),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def extract_text(self) -> str:
        """Extract text from Word document"""
        try:
            self.logger.info(f"Reading document: {self.doc_path}")
            doc = docx.Document(self.doc_path)
            full_text = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text.strip())
            
            self.content = "\n".join(full_text)
            self.logger.info(f"Document read completed, total {len(self.content)} characters")
            return self.content
        except Exception as e:
            self.logger.error(f"Document reading error: {e}")
            return ""
    
    def parse_sections(self) -> Dict[str, str]:
        """Parse document sections"""
        if not self.content:
            self.extract_text()
        
        self.logger.info("Starting document section parsing...")
        lines = self.content.split('\n')
        current_section = "Overview"
        sections = {"Overview": []}
        
        for line in lines:
            # Improved section identification logic
            if (re.match(r'^[0-9]+\.', line) or 
                re.match(r'^[一二三四五六七八九十]+、', line) or
                re.match(r'^第[一二三四五六七八九十]+章', line) or
                re.match(r'^[一二三四五六七八九十]+\.', line)):
                current_section = line.strip()
                sections[current_section] = []
                self.logger.debug(f"New section found: {current_section}")
            else:
                sections[current_section].append(line)
        
        # Clean empty sections and short sections
        self.sections = {}
        for k, v in sections.items():
            content = '\n'.join(v).strip()
            if len(content) > DOC_CONFIG["min_chunk_length"]:
                self.sections[k] = content
        
        self.logger.info(f"Section parsing completed, total {len(self.sections)} valid sections")
        return self.sections


class RAGSystem:
    """RAG retrieval augmented generation system"""
    
    def __init__(self, documents: Dict[str, str]):
        self.documents = documents
        self.embeddings = {}
        self.embedding_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.chunk_size = RAG_CONFIG["chunk_size"]
        self.top_k = RAG_CONFIG["top_k_retrieval"]
        self.similarity_threshold = RAG_CONFIG["similarity_threshold"]
        self.max_length = RAG_CONFIG["max_length"]
        self.batch_size = RAG_CONFIG["batch_size"]
        self._setup_logging()
        self._initialize_embeddings()
    
    def _setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
    
    def _initialize_embeddings(self):
        """Initialize embedding models"""
        # 优先尝试BERT模型
        if RAG_CONFIG["embedding_model"] == "bert" and BERT_AVAILABLE:
            if self._initialize_bert_model():
                return
        
        # 尝试使用离线模型
        if RAG_CONFIG.get("use_offline", False) and os.path.exists(RAG_CONFIG.get("offline_model_path", "")):
            offline_path = os.path.join(RAG_CONFIG["offline_model_path"], RAG_CONFIG["embedding_model"])
            if os.path.exists(offline_path):
                self.logger.info(f"Using offline embedding model: {offline_path}")
                self.embedding_model = SentenceTransformer(offline_path)
                return
        
        # 尝试在线下载模型
        try:
            self.logger.info(f"Loading embedding model: {RAG_CONFIG['embedding_model']}")
            self.embedding_model = SentenceTransformer(RAG_CONFIG['embedding_model'])
            self.logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Primary embedding model loading failed: {e}")
            
            # 尝试备用模型
            try:
                fallback_model = RAG_CONFIG.get("fallback_model", "all-MiniLM-L6-v2")
                self.logger.info(f"Attempting to load fallback model: {fallback_model}")
                self.embedding_model = SentenceTransformer(fallback_model)
                self.logger.info("Fallback embedding model loaded successfully")
            except Exception as e2:
                self.logger.error(f"Fallback embedding model also failed to load: {e2}")
                self.logger.warning("Using keyword-based simple retrieval as fallback")
                self.embedding_model = None
    
    def _initialize_bert_model(self):
        """初始化BERT模型"""
        try:
            bert_path = RAG_CONFIG["bert_model_path"]
            tokenizer_path = RAG_CONFIG["bert_tokenizer_path"]
            
            if not os.path.exists(bert_path):
                self.logger.warning(f"BERT model path does not exist: {bert_path}")
                return False
            
            if not os.path.exists(tokenizer_path):
                self.logger.warning(f"BERT tokenizer path does not exist: {tokenizer_path}")
                return False
            
            self.logger.info(f"Loading local BERT model: {bert_path}")
            
            # 加载BERT分词器和模型
            self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            self.bert_model = BertModel.from_pretrained(bert_path)
            
            # 设置为评估模式
            self.bert_model.eval()
            
            # 检查是否有GPU可用
            if torch.cuda.is_available():
                self.bert_model = self.bert_model.cuda()
                self.logger.info("BERT model loaded to GPU")
            else:
                self.logger.info("BERT model loaded to CPU")
            
            self.logger.info("BERT model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"BERT model loading failed: {e}")
            return False
    
    def create_embeddings(self):
        """为文档创建嵌入向量"""
        if self.bert_model is not None and self.bert_tokenizer is not None:
            self._create_bert_embeddings()
        elif self.embedding_model:
            self._create_sentence_transformer_embeddings()
        else:
            self.logger.warning("No embedding model available, skipping vectorization")
    
    def _create_bert_embeddings(self):
        """使用BERT模型创建嵌入向量"""
        self.logger.info("Starting BERT model document embedding vector creation...")
        total_chunks = 0
        
        for section_name, content in self.documents.items():
            chunks = self._split_text(content, self.chunk_size)
            section_embeddings = []
            
            # 批量处理chunks
            for i in range(0, len(chunks), self.batch_size):
                batch_chunks = chunks[i:i + self.batch_size]
                batch_embeddings = self._encode_bert_batch(batch_chunks)
                
                for j, chunk in enumerate(batch_chunks):
                    if j < len(batch_embeddings):
                        section_embeddings.append({
                            'text': chunk,
                            'embedding': batch_embeddings[j],
                            'section': section_name
                        })
            
            self.embeddings[section_name] = section_embeddings
            total_chunks += len(section_embeddings)
        
        self.logger.info(f"BERT embedding vector creation completed, total {total_chunks} text chunks")
    
    def _create_sentence_transformer_embeddings(self):
        """使用SentenceTransformer创建嵌入向量"""
        self.logger.info("Starting document embedding vector creation...")
        total_chunks = 0
        
        for section_name, content in self.documents.items():
            chunks = self._split_text(content, self.chunk_size)
            section_embeddings = []
            
            for chunk in chunks:
                try:
                    embedding = self.embedding_model.encode(chunk)
                    section_embeddings.append({
                        'text': chunk,
                        'embedding': embedding,
                        'section': section_name
                    })
                except Exception as e:
                    self.logger.warning(f"Embedding vector creation failed: {e}")
                    continue
            
            self.embeddings[section_name] = section_embeddings
            total_chunks += len(section_embeddings)
        
        self.logger.info(f"Embedding vector creation completed, total {total_chunks} text chunks")
    
    def _encode_bert_batch(self, texts: List[str]) -> List[np.ndarray]:
        """使用BERT模型批量编码文本"""
        try:
            # 准备输入
            inputs = self.bert_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 移动到GPU（如果可用）
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 前向传播
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # 使用[CLS]标记的输出作为句子表示
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"BERT encoding failed: {e}")
            # 返回零向量作为备用
            return [np.zeros(768) for _ in texts]  # 假设BERT输出维度为768
    
    def _split_text(self, text: str, max_length: int) -> List[str]:
        """智能文本分段"""
        # 按句子分割
        sentences = re.split(r'[。！？；\n]', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def retrieve_relevant_context(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """检索相关上下文"""
        if top_k is None:
            top_k = self.top_k
            
        self.logger.info(f"Retrieving query: {query}")
        
        if self.bert_model is not None and self.bert_tokenizer is not None:
            return self._bert_retrieval(query, top_k)
        elif self.embedding_model:
            return self._sentence_transformer_retrieval(query, top_k)
        else:
            self.logger.warning("Embedding model not loaded, using keyword-based simple retrieval as fallback")
            return self._keyword_based_retrieval(query, top_k)
    
    def _bert_retrieval(self, query: str, top_k: int) -> List[str]:
        """使用BERT模型进行检索"""
        try:
            # 编码查询
            query_embedding = self._encode_bert_batch([query])[0]
            relevant_chunks = []
            
            for section_name, section_embeddings in self.embeddings.items():
                for chunk_info in section_embeddings:
                    try:
                        similarity = cosine_similarity(
                            [query_embedding], 
                            [chunk_info['embedding']]
                        )[0][0]
                        
                        if similarity >= self.similarity_threshold:
                            relevant_chunks.append({
                                'text': chunk_info['text'],
                                'similarity': similarity,
                                'section': chunk_info['section']
                            })
                    except Exception as e:
                        self.logger.warning(f"Similarity calculation failed: {e}")
                        continue
            
            # 按相似度排序并返回top_k
            relevant_chunks.sort(key=lambda x: x['similarity'], reverse=True)
            selected_chunks = relevant_chunks[:top_k]
            
            self.logger.info(f"BERT retrieval completed, found {len(selected_chunks)} relevant chunks")
            for chunk in selected_chunks:
                self.logger.debug(f"Relevant chunk (similarity: {chunk['similarity']:.3f}): {chunk['text'][:100]}...")
            
            return [chunk['text'] for chunk in selected_chunks]
            
        except Exception as e:
            self.logger.error(f"BERT retrieval failed: {e}")
            return self._keyword_based_retrieval(query, top_k)
    
    def _sentence_transformer_retrieval(self, query: str, top_k: int) -> List[str]:
        """使用SentenceTransformer进行检索"""
        query_embedding = self.embedding_model.encode(query)
        relevant_chunks = []
        
        for section_name, section_embeddings in self.embeddings.items():
            for chunk_info in section_embeddings:
                try:
                    similarity = cosine_similarity(
                        [query_embedding], 
                        [chunk_info['embedding']]
                    )[0][0]
                    
                    if similarity >= self.similarity_threshold:
                        relevant_chunks.append({
                            'text': chunk_info['text'],
                            'similarity': similarity,
                            'section': chunk_info['section']
                        })
                except Exception as e:
                    self.logger.warning(f"Similarity calculation failed: {e}")
                    continue
        
        # 按相似度排序并返回top_k
        relevant_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        selected_chunks = relevant_chunks[:top_k]
        
        self.logger.info(f"Retrieval completed, found {len(selected_chunks)} relevant chunks")
        for chunk in selected_chunks:
            self.logger.debug(f"Relevant chunk (similarity: {chunk['similarity']:.3f}): {chunk['text'][:100]}...")
        
        return [chunk['text'] for chunk in selected_chunks]
    
    def _keyword_based_retrieval(self, query: str, top_k: int) -> List[str]:
        """基于关键词的简单检索（备用方案）"""
        self.logger.info("Using keyword-based retrieval")
        
        # Simple keyword matching
        query_words = set(query.lower().split())
        relevant_chunks = []
        
        for section_name, section_embeddings in self.embeddings.items():
            for chunk_info in section_embeddings:
                chunk_text = chunk_info['text'].lower()
                chunk_words = set(chunk_text.split())
                
                # 计算关键词重叠度
                overlap = len(query_words.intersection(chunk_words))
                if overlap > 0:
                    relevant_chunks.append({
                        'text': chunk_info['text'],
                        'overlap': overlap,
                        'section': section_name
                    })
        
        # Sort by overlap
        relevant_chunks.sort(key=lambda x: x['overlap'], reverse=True)
        selected_chunks = relevant_chunks[:top_k]
        
        self.logger.info(f"Keyword retrieval completed, found {len(selected_chunks)} relevant chunks")
        return [chunk['text'] for chunk in selected_chunks]


class TaskInformationRetriever:
    """任务信息检索器"""
    
    def __init__(self, emergency_knowledge_base: List[EmergencyKnowledge], rag_system: RAGSystem):
        self.emergency_knowledge_base = emergency_knowledge_base
        self.rag_system = rag_system
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
    
    def retrieve_task_information(self, task_description: str, top_k: int = 3) -> TaskInformation:
        """检索任务相关信息"""
        self.logger.info(f"Retrieving task information for: {task_description}")
        
        # 使用RAG系统检索相关上下文
        relevant_context = self.rag_system.retrieve_relevant_context(task_description, top_k)
        
        # 从急诊知识库中检索相关信息
        emergency_info = self._retrieve_emergency_knowledge(task_description, top_k)
        
        # 构建扩展信息
        extended_info = self._build_extended_info(relevant_context, emergency_info)
        
        # 创建任务信息对象
        task_info = TaskInformation(
            task_id=f"task_{hash(task_description) % 10000}",
            task_description=task_description,
            task_type="emergency_triage",
            priority=self._determine_priority(task_description),
            requirements=self._extract_requirements(task_description, emergency_info),
            context="; ".join(relevant_context),
            extended_info=extended_info
        )
        
        self.logger.info(f"Task information retrieved successfully: {task_info.task_id}")
        return task_info
    
    def _retrieve_emergency_knowledge(self, task_description: str, top_k: int) -> List[EmergencyKnowledge]:
        """从急诊知识库检索相关知识"""
        relevant_knowledge = []
        
        # Simple keyword matching
        task_words = set(task_description.lower().split())
        
        for knowledge in self.emergency_knowledge_base:
            knowledge_words = set(knowledge.keywords)
            overlap = len(task_words.intersection(knowledge_words))
            
            if overlap > 0:
                relevant_knowledge.append((knowledge, overlap))
        
        # Sort by overlap
        relevant_knowledge.sort(key=lambda x: x[1], reverse=True)
        
        return [knowledge for knowledge, _ in relevant_knowledge[:top_k]]
    
    def _build_extended_info(self, relevant_context: List[str], emergency_info: List[EmergencyKnowledge]) -> str:
        """构建扩展信息"""
        extended_parts = []
        
        if relevant_context:
            extended_parts.append("相关背景信息:\n" + "\n".join(relevant_context))
        
        if emergency_info:
            emergency_text = "急诊知识库信息:\n"
            for info in emergency_info:
                emergency_text += f"- {info.title}: {info.content}\n"
            extended_parts.append(emergency_text)
        
        return "\n\n".join(extended_parts)
    
    def _determine_priority(self, task_description: str) -> str:
        """确定任务优先级"""
        high_priority_keywords = ["紧急", "危重", "抢救", "生命危险", "急性"]
        medium_priority_keywords = ["急症", "疼痛", "不适", "症状"]
        
        task_lower = task_description.lower()
        
        for keyword in high_priority_keywords:
            if keyword in task_lower:
                return "high"
        
        for keyword in medium_priority_keywords:
            if keyword in task_lower:
                return "medium"
        
        return "normal"
    
    def _extract_requirements(self, task_description: str, emergency_info: List[EmergencyKnowledge]) -> List[str]:
        """提取任务需求"""
        requirements = []
        
        # Extract requirements based on task description
        if "分诊" in task_description or "triage" in task_description.lower():
            requirements.append("Patient triage assessment")
        if "生命体征" in task_description or "vital signs" in task_description.lower():
            requirements.append("Vital signs monitoring")
        if "症状" in task_description or "symptom" in task_description.lower():
            requirements.append("Symptom analysis")
        if "风险" in task_description or "risk" in task_description.lower():
            requirements.append("Risk assessment")
        if "再入院" in task_description or "readmission" in task_description.lower():
            requirements.append("Readmission prediction")
        if "药物推荐" in task_description or "medication recommendation" in task_description.lower():
            requirements.append("Medication recommendation")
        if "药物冲突" in task_description or "DDI" in task_description.upper() or "drug interaction" in task_description.lower():
            requirements.append("DDI detection")
        
        # 基于急诊知识库信息提取需求
        for info in emergency_info:
            if "分诊" in info.title:
                requirements.append("分诊决策支持")
            if "工具" in info.title:
                requirements.append("决策支持工具")
        
        return list(set(requirements))  # 去重


class GraphSelector:
    """图谱选择器"""
    
    def __init__(self, knowledge_graphs: List[KnowledgeGraph], llm):
        self.knowledge_graphs = knowledge_graphs
        self.llm = llm
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
    
    def select_knowledge_graph(self, task_info: TaskInformation) -> str:
        """选择最适合的知识图谱"""
        self.logger.info(f"Selecting knowledge graph for task: {task_info.task_id}")
        
        # 构建图谱信息
        graph_info = self._format_knowledge_graphs()
        
        # 生成选择提示词
        prompt = self._generate_graph_selection_prompt(task_info, graph_info)
        
        # Call LLM for selection
        try:
            response = self.llm.generate_text(
                prompt=prompt,
                max_tokens=GENERATION_CONFIG["max_tokens"],
                temperature=GENERATION_CONFIG["temperature"],
                top_p=GENERATION_CONFIG["top_p"]
            )
            
            # 解析响应，提取选中的图谱
            selected_graph = self._parse_graph_selection(response)
            
            self.logger.info(f"Selected knowledge graph: {selected_graph}")
            return selected_graph
            
        except Exception as e:
            self.logger.error(f"Graph selection failed: {e}")
            # 返回默认图谱
            return "established medical knowledge graph"
    
    def _format_knowledge_graphs(self) -> str:
        """格式化知识图谱信息"""
        formatted = ""
        for i, kg in enumerate(self.knowledge_graphs, 1):
            formatted += f"""
{i}. **{kg.name}**
   - Description: {kg.description}
   - Domain: {kg.domain}
   - Advantages: {', '.join(kg.advantages)}
   - Use Cases: {', '.join(kg.use_cases)}
   - Limitations: {', '.join(kg.limitations)}
"""
        return formatted
    
    def _generate_graph_selection_prompt(self, task_info: TaskInformation, graph_info: str) -> str:
        """生成图谱选择提示词"""
        return f"""<|im_start|>system
{PROMPT_TEMPLATES['graph_selection']}
<|im_end|>

<|im_start|>user
Task Information:
- Task Description: {task_info.task_description}
- Task Type: {task_info.task_type}
- Priority: {task_info.priority}
- Requirements: {', '.join(task_info.requirements)}
- Context: {task_info.context}
- Extended Information: {task_info.extended_info}

Available Knowledge Graphs:
{graph_info}
<|im_end|>

<|im_start|>assistant
"""
    
    def _parse_graph_selection(self, response: str) -> str:
        """解析图谱选择响应"""
        # 查找选中的图谱名称
        for kg in self.knowledge_graphs:
            if kg.name.lower() in response.lower():
                return kg.name
        
        # 如果没找到，返回第一个图谱
        return self.knowledge_graphs[0].name if self.knowledge_graphs else "established medical knowledge graph"


class EDCDSToolSelector:
    """ED-CDS工具选择器"""
    
    def __init__(self, tools: List[DecisionSupportTool], llm):
        self.tools = tools
        self.llm = llm
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
    
    def select_tools(self, task_info: TaskInformation, selected_graph: str) -> List[str]:
        """选择最适合的工具"""
        self.logger.info(f"Selecting tools for task: {task_info.task_id}, graph: {selected_graph}")
        
        # 构建工具信息
        tool_info = self._format_tools()
        
        # 生成工具选择提示词
        prompt = self._generate_tool_selection_prompt(task_info, selected_graph, tool_info)
        
        # Call LLM for selection
        try:
            response = self.llm.generate_text(
                prompt=prompt,
                max_tokens=GENERATION_CONFIG["max_tokens"],
                temperature=GENERATION_CONFIG["temperature"],
                top_p=GENERATION_CONFIG["top_p"]
            )
            
            # 解析响应，提取选中的工具
            selected_tools = self._parse_tool_selection(response)
            
            self.logger.info(f"Selected tools: {selected_tools}")
            return selected_tools
            
        except Exception as e:
            self.logger.error(f"Tool selection failed: {e}")
            # 返回默认工具
            return ["triage_assessment_tool"]
    
    def _format_tools(self) -> str:
        """格式化工具信息"""
        formatted = ""
        for i, tool in enumerate(self.tools, 1):
            formatted += f"""
{i}. **{tool.name}** (ID: {tool.id})
   - Description: {tool.description}
   - Input Requirements: {', '.join(tool.input_requirements)}
   - Output Format: {tool.output_format}
   - Applicable Scenarios: {', '.join(tool.applicable_scenarios)}
   - Accuracy: {tool.accuracy}
   - Processing Time: {tool.processing_time}
"""
        return formatted
    
    def _generate_tool_selection_prompt(self, task_info: TaskInformation, selected_graph: str, tool_info: str) -> str:
        """生成工具选择提示词"""
        return f"""<|im_start|>system
{PROMPT_TEMPLATES['tool_selection']}
<|im_end|>

<|im_start|>user
Selected Knowledge Graph: {selected_graph}

Task Information:
- Task Description: {task_info.task_description}
- Task Type: {task_info.task_type}
- Priority: {task_info.priority}
- Requirements: {', '.join(task_info.requirements)}
- Context: {task_info.context}

Available Tools:
{tool_info}
<|im_end|>

<|im_start|>assistant
"""
    
    def _parse_tool_selection(self, response: str) -> List[str]:
        """解析工具选择响应"""
        selected_tools = []
        
        # 查找选中的工具ID
        for tool in self.tools:
            if tool.id.lower() in response.lower() or tool.name.lower() in response.lower():
                selected_tools.append(tool.id)
        
        # 如果没找到，返回默认工具
        if not selected_tools:
            selected_tools = ["triage_assessment_tool"]
        
        return selected_tools


class KnDAgent:
    """Knowledge Graph Agent for medical emergency tasks"""
    
    def __init__(self, api_key: str = None, doc_path: str = None):
        self.api_key = api_key or MODEL_CONFIG["api_key"]
        self.doc_path = doc_path or DOC_CONFIG["doc_path"]
        self.llm = None
        self.doc_processor = None
        self.rag_system = None
        self.knowledge_graphs = []
        self.emergency_knowledge = []
        self.decision_tools = []
        self.task_retriever = None
        self.graph_selector = None
        self.tool_selector = None
        self._setup_logging()
        self._initialize()
    
    def _setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
    
    def _initialize(self):
        """Initialize the medical knowledge agent"""
        self.logger.info("Initializing Medical Knowledge Graph Recommendation Agent...")
        
        try:
            # Initialize DeepSeek API client
            self.logger.info("Initializing DeepSeek API client...")
            self.llm = DeepSeekAPIClient(api_key=self.api_key)
            
            # Test API connection
            if not self.llm.test_connection():
                raise Exception("Failed to connect to DeepSeek API")
            
            self.logger.info("DeepSeek API client initialized successfully")
            
            # Process medical knowledge document
            self.logger.info("Processing medical knowledge document...")
            self.doc_processor = DocumentProcessor(self.doc_path)
            sections = self.doc_processor.parse_sections()
            
            # Initialize RAG system
            self.rag_system = RAGSystem(sections)
            self.rag_system.create_embeddings()
            
            # Initialize knowledge graphs
            self._initialize_knowledge_graphs()
            
            # Initialize emergency knowledge base
            self._initialize_emergency_knowledge()
            
            # Initialize decision support tools
            self._initialize_decision_tools()
            
            # Initialize task processing components
            self._initialize_task_components()
            
            self.logger.info("Agent initialization completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Agent initialization failed: {e}")
            raise
    
    def _initialize_knowledge_graphs(self):
        """Initialize knowledge graph library"""
        self.logger.info("Initializing knowledge graph library...")
        
        for kg_config in KNOWLEDGE_GRAPH_CONFIG["default_graphs"]:
            kg = KnowledgeGraph(
                name=kg_config["name"],
                description=kg_config["description"],
                domain=kg_config["domain"],
                advantages=kg_config["advantages"],
                limitations=kg_config["limitations"],
                use_cases=kg_config["use_cases"],
                data_sources=kg_config["data_sources"]
            )
            self.knowledge_graphs.append(kg)
        
        self.logger.info(f"Knowledge graph library initialization completed, total {len(self.knowledge_graphs)} graphs")
    
    def _initialize_emergency_knowledge(self):
        """Initialize emergency knowledge base"""
        self.logger.info("Initializing emergency knowledge base...")
        
        # Load from external file if exists, otherwise use default
        knowledge_path = EMERGENCY_KNOWLEDGE_CONFIG.get("knowledge_base_path")
        if knowledge_path and os.path.exists(knowledge_path):
            try:
                with open(knowledge_path, 'r', encoding='utf-8') as f:
                    knowledge_data = json.load(f)
                    knowledge_list = knowledge_data.get("knowledge_base", [])
            except Exception as e:
                self.logger.warning(f"Failed to load external knowledge base: {e}, using default")
                knowledge_list = EMERGENCY_KNOWLEDGE_CONFIG["default_knowledge"]
        else:
            knowledge_list = EMERGENCY_KNOWLEDGE_CONFIG["default_knowledge"]
        
        for knowledge_config in knowledge_list:
            knowledge = EmergencyKnowledge(
                id=knowledge_config["id"],
                title=knowledge_config["title"],
                content=knowledge_config["content"],
                keywords=knowledge_config["keywords"],
                category=knowledge_config["category"]
            )
            self.emergency_knowledge.append(knowledge)
        
        self.logger.info(f"Emergency knowledge base initialization completed, total {len(self.emergency_knowledge)} entries")
    
    def _initialize_decision_tools(self):
        """Initialize decision support tools"""
        self.logger.info("Initializing decision support tools...")
        
        # Load from external file if exists, otherwise use default
        tools_path = DECISION_SUPPORT_TOOLS_CONFIG.get("tools_path")
        if tools_path and os.path.exists(tools_path):
            try:
                with open(tools_path, 'r', encoding='utf-8') as f:
                    tools_data = json.load(f)
                    tools_list = tools_data.get("tools", [])
            except Exception as e:
                self.logger.warning(f"Failed to load external tools configuration: {e}, using default")
                tools_list = DECISION_SUPPORT_TOOLS_CONFIG["default_tools"]
        else:
            tools_list = DECISION_SUPPORT_TOOLS_CONFIG["default_tools"]
        
        for tool_config in tools_list:
            tool = DecisionSupportTool(
                id=tool_config["id"],
                name=tool_config["name"],
                description=tool_config["description"],
                input_requirements=tool_config["input_requirements"],
                output_format=tool_config["output_format"],
                applicable_scenarios=tool_config["applicable_scenarios"],
                accuracy=tool_config["accuracy"],
                processing_time=tool_config["processing_time"]
            )
            self.decision_tools.append(tool)
        
        self.logger.info(f"Decision support tools initialization completed, total {len(self.decision_tools)} tools")
    
    def _initialize_task_components(self):
        """Initialize task processing components"""
        self.logger.info("Initializing task processing components...")
        
        # Initialize task information retriever
        self.task_retriever = TaskInformationRetriever(
            emergency_knowledge_base=self.emergency_knowledge,
            rag_system=self.rag_system
        )
        
        # Initialize graph selector
        self.graph_selector = GraphSelector(
            knowledge_graphs=self.knowledge_graphs,
            llm=self.llm
        )
        
        # Initialize tool selector
        self.tool_selector = EDCDSToolSelector(
            tools=self.decision_tools,
            llm=self.llm
        )
        
        self.logger.info("Task processing components initialization completed")
    
    def process_emergency_task(self, task_description: str) -> ExecutionChain:
        """
        Process emergency task through the complete execution chain
        
        Args:
            task_description: Description of the emergency task
            
        Returns:
            ExecutionChain object containing the complete execution strategy
        """
        self.logger.info(f"Processing emergency task: {task_description}")
        
        try:
            # Step 1: Retrieve task information
            self.logger.info("Step 1: Retrieving task information...")
            task_info = self.task_retriever.retrieve_task_information(task_description)
            
            # Step 2: Select knowledge graph
            self.logger.info("Step 2: Selecting knowledge graph...")
            selected_graph = self.graph_selector.select_knowledge_graph(task_info)
            
            # Step 3: Select tools
            self.logger.info("Step 3: Selecting decision support tools...")
            selected_tools = self.tool_selector.select_tools(task_info, selected_graph)
            
            # Step 4: Generate execution strategy
            self.logger.info("Step 4: Generating execution strategy...")
            execution_strategy = self._generate_execution_strategy(
                task_info, selected_graph, selected_tools
            )
            
            # Create execution chain
            execution_chain = ExecutionChain(
                task_info=task_info,
                selected_graph=selected_graph,
                selected_tools=selected_tools,
                execution_strategy=execution_strategy,
                confidence_score=self._calculate_confidence_score(task_info, selected_graph, selected_tools)
            )
            
            self.logger.info("Emergency task processing completed successfully")
            return execution_chain
            
        except Exception as e:
            self.logger.error(f"Emergency task processing failed: {e}")
            raise
    
    def _generate_execution_strategy(self, task_info: TaskInformation, 
                                   selected_graph: str, selected_tools: List[str]) -> str:
        """Generate execution strategy based on selected components"""
        strategy_parts = [
            f"Task: {task_info.task_description}",
            f"Task Type: {task_info.task_type}",
            f"Priority: {task_info.priority}",
            f"Selected Knowledge Graph: {selected_graph}",
            f"Selected Tools: {', '.join(selected_tools)}",
            f"Requirements: {', '.join(task_info.requirements)}"
        ]
        
        return "\n".join(strategy_parts)
    
    def _calculate_confidence_score(self, task_info: TaskInformation, 
                                  selected_graph: str, selected_tools: List[str]) -> float:
        """Calculate confidence score for the execution chain"""
        # Simple confidence calculation based on available information
        base_score = 0.7
        
        # Increase score based on task information completeness
        if task_info.extended_info and len(task_info.extended_info) > 100:
            base_score += 0.1
        
        # Increase score based on number of requirements met
        if len(task_info.requirements) > 0:
            base_score += 0.1
        
        # Increase score based on tool selection
        if len(selected_tools) > 0:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def generate_recommendation_prompt(self, use_case: str, relevant_context: List[str]) -> str:
        """生成推荐提示词"""
        context_text = "\n".join(relevant_context)
        knowledge_graphs_text = self._format_knowledge_graphs()
        
        # Build chat format prompt
        prompt = f"""<|im_start|>system
You are a professional medical knowledge graph expert. Based on the following medical use case and related information, please recommend the most suitable type of knowledge graph and explain the reasons for your selection. Please respond in English.
<|im_end|>

<|im_start|>user
Medical Use Case: {use_case}

Related Background Information:
{context_text}

Available Types of Knowledge Graphs:
{knowledge_graphs_text}

Please respond in the following format:

## Recommended Knowledge Graph
**Graph Name**: [Name of the recommended knowledge graph]

## Reasons for Selection
1. **Applicability**: [Why this graph is the most suitable for the use case]
2. **Advantages**: [The main advantages of this graph in this scenario]

## Implementation Suggestions
[Brief implementation suggestions and considerations]
Please ensure the recommendation is based on the specific needs of the use case and fully utilizes the provided background information.
<|im_end|>

<|im_start|>assistant
"""
        
        return prompt
    
    def _format_knowledge_graphs(self) -> str:
        """格式化知识图谱信息"""
        formatted = ""
        for i, kg in enumerate(self.knowledge_graphs, 1):
            formatted += f"""
{i}. **{kg.name}**
   - Description: {kg.description}
   - Domain: {kg.domain}
   - Advantages: {', '.join(kg.advantages)}
   - Use Cases: {', '.join(kg.use_cases)}
"""
        return formatted
    
    def recommend_knowledge_graph(self, use_case: str) -> str:
        """推荐知识图谱"""
        self.logger.info(f"Analyzing use case: {use_case}")
        
        try:
            # 检索相关上下文
            relevant_context = self.rag_system.retrieve_relevant_context(use_case)
            
            if not relevant_context:
                self.logger.warning("No relevant context found, using default knowledge graph information")
                relevant_context = ["Based on general medical knowledge"]
            
            # 生成推荐提示词
            prompt = self.generate_recommendation_prompt(use_case, relevant_context)
            
            # Debug information
            self.logger.info(f"Generated prompt length: {len(prompt)}")
            self.logger.debug(f"Prompt content: {prompt[:500]}...")
            
            # 调用大模型生成推荐
            self.logger.info("Generating recommendation...")
            recommendation = self.llm.generate_text(
                prompt=prompt,
                max_tokens=GENERATION_CONFIG["max_tokens"],
                temperature=GENERATION_CONFIG["temperature"],
                top_p=GENERATION_CONFIG["top_p"]
            )
            
            # Debug information
            self.logger.info(f"LLM response length: {len(recommendation) if recommendation else 0}")
            self.logger.debug(f"LLM response: {recommendation[:200] if recommendation else 'None'}...")
            
            # 检查响应是否为空
            if not recommendation or recommendation.strip() == "":
                self.logger.warning("LLM returned empty response, using fallback")
                return self._generate_fallback_recommendation(use_case, relevant_context)
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendation: {e}")
            return f"Sorry, an error occurred while generating the recommendation: {e}"
    
    def _generate_fallback_recommendation(self, use_case: str, relevant_context: List[str]) -> str:
        """生成备用推荐（当大模型失败时）"""
        self.logger.info("Using fallback recommendation scheme")
        
        # 基于关键词匹配的简单推荐
        use_case_lower = use_case.lower()
        
        # 关键词映射
        keyword_mapping = {
            "prediction": "dynamic clinical data knowledge graph",
            "readmission": "dynamic clinical data knowledge graph",
            "temporal": "dynamic clinical data knowledge graph",
            "longitudinal": "dynamic clinical data knowledge graph",
            "personalized": "dynamic clinical data knowledge graph",
            "drug": "established medical knowledge graph",
            "medication": "established medical knowledge graph",
            "interaction": "established medical knowledge graph",
            "ddi": "established medical knowledge graph",
            "imaging": "hybrid knowledge graph",
            "multimodal": "hybrid knowledge graph",
            "integrated": "hybrid knowledge graph"
        }
        
        # 找到最匹配的知识图谱
        best_match = None
        best_score = 0
        
        for keyword, graph_name in keyword_mapping.items():
            if keyword in use_case_lower:
                best_match = graph_name
                best_score = 1
                break
        
        if not best_match:
            best_match = "established medical knowledge graph"  # 默认推荐
        
        # 生成推荐文本
        recommendation = f"""## Recommended Knowledge Graph
**Graph Name**: {best_match}

## Reasons for Selection
1. **Applicability**: Based on keyword analysis of your use case, {best_match} is most suitable for this scenario.
2. **Advantages**: This graph provides the necessary capabilities for your specific requirements.

## Implementation Suggestions
Consider integrating this knowledge graph with your existing systems for optimal results.

---
*Note: This recommendation was generated using a fallback keyword-based system. For more precise analysis, please check the LLM connection status.*"""
        
        return recommendation
    
    def analyze_use_case(self, use_case: str) -> str:
        """分析use case"""
        self.logger.info(f"Analyzing use case: {use_case}")
        
        try:
            # Build chat format prompt
            prompt = f"""<|im_start|>system
You are a medical knowledge graph expert. Please analyze the characteristics and requirements of the following medical use case to help determine the most suitable type of knowledge graph.
<|im_end|>

<|im_start|>user
Use Case: {use_case}

Please analyze from the following dimensions:
1. **Primary Objective**: What is the core problem this use case aims to solve?
2. **Data Types**: What types of data need to be processed?
3. **Timing Requirements**: Is real-time processing or historical data analysis required?
4. **Accuracy Requirements**: What level of accuracy is required for the results?
5. **Application Scenario**: In what environment will it be used?
<|im_end|>

<|im_start|>assistant
"""
            
            analysis = self.llm.generate_text(
                prompt=prompt,
                max_tokens=GENERATION_CONFIG["max_tokens"],
                temperature=GENERATION_CONFIG["temperature"],
                top_p=GENERATION_CONFIG["top_p"]
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze use case: {e}")
            return f"Sorry, an error occurred while analyzing the use case: {e}"
    
    def interactive_mode(self):
        """Interactive mode for emergency task processing"""
        print("\n" + "="*60)
        print("🏥 KnDAgent - Knowledge Graph Agent")
        print("="*60)
        print("Supported Emergency Tasks:")
        print("1. Emergency Triage - Patient triage and priority assessment")
        print("2. Readmission Prediction - Hospital readmission risk prediction")
        print("3. Medication Recommendation - Drug recommendation and selection")
        print("4. DDI Detection - Drug-drug interaction identification and analysis")
        print("="*60)
        print("Commands:")
        print("- Input task description to get execution chain")
        print("- Type 'quit' or 'exit' to exit the program")
        print("="*60)
        
        while True:
            try:
                user_input = input("\nPlease input emergency task description: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Thank you for using! Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Process emergency task
                print(f"\nProcessing emergency task: {user_input}")
                execution_chain = self.process_emergency_task(user_input)
                
                print("\n" + "="*60)
                print("🎯 Execution Chain Results:")
                print("="*60)
                print(f"Task ID: {execution_chain.task_info.task_id}")
                print(f"Task Description: {execution_chain.task_info.task_description}")
                print(f"Task Type: {execution_chain.task_info.task_type}")
                print(f"Priority: {execution_chain.task_info.priority}")
                print(f"Selected Knowledge Graph: {execution_chain.selected_graph}")
                print(f"Selected Tools: {', '.join(execution_chain.selected_tools)}")
                print(f"Confidence Score: {execution_chain.confidence_score:.2f}")
                print("\nExecution Strategy:")
                print(execution_chain.execution_strategy)
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n\nProgram interrupted, exiting...")
                break
            except Exception as e:
                self.logger.error(f"Interactive mode error: {e}")
                print(f"\nError: {e}")


def main():
    """Main function for the medical knowledge agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="KnDAgent - Knowledge Graph Agent")
    parser.add_argument("--api-key", type=str, 
                       help="DeepSeek API key")
    parser.add_argument("--doc-path", type=str, 
                       help="Medical knowledge document path")
    parser.add_argument("--interactive", action="store_true", 
                       help="Start interactive mode")
    parser.add_argument("--test", type=str, 
                       help="Test specified emergency task")
    
    args = parser.parse_args()
    
    try:
        # Create agent
        agent = KnDAgent(
            api_key=args.api_key,
            doc_path=args.doc_path
        )
        
        if args.test:
            # Test mode
            print(f"Testing emergency task: {args.test}")
            execution_chain = agent.process_emergency_task(args.test)
            print("\nExecution Chain:")
            print(f"Task: {execution_chain.task_info.task_description}")
            print(f"Selected Graph: {execution_chain.selected_graph}")
            print(f"Selected Tools: {', '.join(execution_chain.selected_tools)}")
            print(f"Confidence: {execution_chain.confidence_score:.2f}")
        elif args.interactive:
            # Interactive mode
            agent.interactive_mode()
        else:
            # Default test
            test_task = "Emergency triage for patient with chest pain and shortness of breath"
            print(f"Default test task: {test_task}")
            execution_chain = agent.process_emergency_task(test_task)
            print("\nExecution Chain:")
            print(f"Task: {execution_chain.task_info.task_description}")
            print(f"Selected Graph: {execution_chain.selected_graph}")
            print(f"Selected Tools: {', '.join(execution_chain.selected_tools)}")
            print(f"Confidence: {execution_chain.confidence_score:.2f}")
            
    except Exception as e:
        print(f"Agent failed to run: {e}")
        logging.error(f"Agent failed to run: {e}")


if __name__ == "__main__":
    main()
