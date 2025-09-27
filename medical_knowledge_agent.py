#!/usr/bin/env python3
"""
医疗知识图谱推荐Agent
基于DeepSeek模型和RAG技术，为医疗use case推荐最适合的知识图谱
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from deploy_deepseek_vllm import DeepSeekVLLMServer


@dataclass
class KnowledgeGraph:
    """知识图谱信息"""
    name: str
    description: str
    domain: str
    advantages: List[str]
    limitations: List[str]
    use_cases: List[str]
    data_sources: List[str]


@dataclass
class UseCase:
    """医疗use case信息"""
    name: str
    description: str
    domain: str
    requirements: List[str]
    challenges: List[str]


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, doc_path: str):
        self.doc_path = doc_path
        self.content = ""
        self.sections = {}
        
    def extract_text(self) -> str:
        """从Word文档中提取文本"""
        try:
            doc = docx.Document(self.doc_path)
            full_text = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text.strip())
            
            self.content = "\n".join(full_text)
            return self.content
        except Exception as e:
            print(f"文档读取错误: {e}")
            return ""
    
    def parse_sections(self) -> Dict[str, str]:
        """解析文档章节"""
        if not self.content:
            self.extract_text()
        
        # 简单的章节分割逻辑
        lines = self.content.split('\n')
        current_section = "概述"
        sections = {"概述": []}
        
        for line in lines:
            if re.match(r'^[0-9]+\.', line) or re.match(r'^[一二三四五六七八九十]+、', line):
                current_section = line.strip()
                sections[current_section] = []
            else:
                sections[current_section].append(line)
        
        # 清理空章节
        self.sections = {k: '\n'.join(v) for k, v in sections.items() if v}
        return self.sections


class RAGSystem:
    """RAG检索增强生成系统"""
    
    def __init__(self, documents: Dict[str, str]):
        self.documents = documents
        self.embeddings = {}
        self.embedding_model = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """初始化嵌入模型"""
        try:
            # 使用轻量级的sentence-transformers模型
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("嵌入模型加载成功")
        except Exception as e:
            print(f"嵌入模型加载失败: {e}")
            self.embedding_model = None
    
    def create_embeddings(self):
        """为文档创建嵌入向量"""
        if not self.embedding_model:
            return
        
        for section_name, content in self.documents.items():
            # 分段处理长文本
            chunks = self._split_text(content, max_length=512)
            section_embeddings = []
            
            for chunk in chunks:
                embedding = self.embedding_model.encode(chunk)
                section_embeddings.append({
                    'text': chunk,
                    'embedding': embedding
                })
            
            self.embeddings[section_name] = section_embeddings
    
    def _split_text(self, text: str, max_length: int = 512) -> List[str]:
        """文本分段"""
        sentences = re.split(r'[。！？；]', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
        """检索相关上下文"""
        if not self.embedding_model:
            return list(self.documents.values())[:top_k]
        
        query_embedding = self.embedding_model.encode(query)
        relevant_chunks = []
        
        for section_name, section_embeddings in self.embeddings.items():
            for chunk_info in section_embeddings:
                similarity = cosine_similarity(
                    [query_embedding], 
                    [chunk_info['embedding']]
                )[0][0]
                
                relevant_chunks.append({
                    'text': chunk_info['text'],
                    'similarity': similarity,
                    'section': section_name
                })
        
        # 按相似度排序并返回top_k
        relevant_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        return [chunk['text'] for chunk in relevant_chunks[:top_k]]


class MedicalKnowledgeAgent:
    """医疗知识图谱推荐Agent"""
    
    def __init__(self, model_path: str, doc_path: str):
        self.llm = DeepSeekVLLMServer(model_path)
        self.doc_processor = DocumentProcessor(doc_path)
        self.rag_system = None
        self.knowledge_graphs = []
        self._initialize()
    
    def _initialize(self):
        """初始化agent"""
        print("正在初始化医疗知识图谱推荐Agent...")
        
        # 加载模型
        self.llm.initialize_engine()
        
        # 处理文档
        print("正在处理医疗知识文档...")
        sections = self.doc_processor.parse_sections()
        self.rag_system = RAGSystem(sections)
        self.rag_system.create_embeddings()
        
        # 初始化知识图谱库
        self._initialize_knowledge_graphs()
        
        print("Agent初始化完成！")
    
    def _initialize_knowledge_graphs(self):
        """初始化知识图谱库"""
        # 这里可以根据文档内容动态生成，或者使用预定义的模板
        self.knowledge_graphs = [
            KnowledgeGraph(
                name="医疗实体关系图谱",
                description="基于医疗实体和关系的知识图谱，包含疾病、症状、药物、治疗等实体及其关系",
                domain="通用医疗",
                advantages=["覆盖全面", "关系清晰", "易于扩展"],
                limitations=["缺乏时序信息", "静态知识"],
                use_cases=["疾病诊断", "药物相互作用", "症状分析"],
                data_sources=["医学文献", "临床指南", "药物数据库"]
            ),
            KnowledgeGraph(
                name="时序医疗事件图谱",
                description="记录患者医疗事件时间序列的知识图谱，包含诊断、治疗、用药等事件的时间关系",
                domain="临床决策",
                coverage="病程追踪、治疗效果评估、预后分析",
                advantages=["时序信息完整", "动态更新", "支持预测"],
                limitations=["数据量大", "隐私敏感"],
                use_cases=["病程预测", "治疗效果评估", "复发风险预测"],
                data_sources=["电子病历", "临床数据", "随访记录"]
            ),
            KnowledgeGraph(
                name="多模态医疗图谱",
                description="整合文本、图像、基因等多种数据类型的综合知识图谱",
                domain="精准医疗",
                coverage="影像诊断、基因分析、个性化治疗",
                advantages=["信息丰富", "多维度分析", "精准度高"],
                limitations=["复杂度高", "计算资源需求大"],
                use_cases=["影像诊断", "基因检测", "个性化用药"],
                data_sources=["医学影像", "基因测序", "病理报告"]
            )
        ]
    
    def generate_recommendation_prompt(self, use_case: str, relevant_context: List[str]) -> str:
        """生成推荐提示词"""
        context_text = "\n".join(relevant_context)
        
        prompt = f"""你是一个专业的医疗知识图谱专家。基于以下医疗use case和相关信息，请推荐最适合的知识图谱类型，并说明选择理由。

医疗Use Case: {use_case}

相关背景信息:
{context_text}

可用的知识图谱类型:
{self._format_knowledge_graphs()}

请按照以下格式回答:

## 推荐的知识图谱
**图谱名称**: [推荐的知识图谱名称]

## 选择理由
1. **适用性**: [为什么这个图谱最适合该use case]
2. **优势**: [该图谱在此场景下的主要优势]
3. **覆盖范围**: [该图谱如何覆盖use case的需求]
4. **数据支持**: [该图谱的数据来源和可靠性]

## 实施建议
[简要的实施建议和注意事项]

请确保推荐基于use case的具体需求，并充分利用提供的背景信息。"""

        return prompt
    
    def _format_knowledge_graphs(self) -> str:
        """格式化知识图谱信息"""
        formatted = ""
        for i, kg in enumerate(self.knowledge_graphs, 1):
            formatted += f"""
{i}. **{kg.name}**
   - 描述: {kg.description}
   - 适用领域: {kg.domain}
   - 主要优势: {', '.join(kg.advantages)}
   - 适用场景: {', '.join(kg.use_cases)}
"""
        return formatted
    
    def recommend_knowledge_graph(self, use_case: str) -> str:
        """推荐知识图谱"""
        print(f"正在分析use case: {use_case}")
        
        # 检索相关上下文
        relevant_context = self.rag_system.retrieve_relevant_context(use_case)
        
        # 生成推荐提示词
        prompt = self.generate_recommendation_prompt(use_case, relevant_context)
        
        # 调用大模型生成推荐
        print("正在生成推荐...")
        recommendation = self.llm.generate_text(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9
        )
        
        return recommendation
    
    def interactive_mode(self):
        """交互模式"""
        print("\n=== 医疗知识图谱推荐Agent ===")
        print("输入医疗use case，我将为您推荐最适合的知识图谱")
        print("输入 'quit' 或 'exit' 退出程序")
        print("=" * 50)
        
        while True:
            try:
                use_case = input("\n请输入医疗use case: ").strip()
                
                if use_case.lower() in ['quit', 'exit', '退出']:
                    print("感谢使用！再见！")
                    break
                
                if not use_case:
                    continue
                
                print("\n正在分析，请稍候...")
                recommendation = self.recommend_knowledge_graph(use_case)
                
                print("\n" + "="*50)
                print("推荐结果:")
                print("="*50)
                print(recommendation)
                print("="*50)
                
            except KeyboardInterrupt:
                print("\n\n程序被中断，正在退出...")
                break
            except Exception as e:
                print(f"\n错误: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="医疗知识图谱推荐Agent")
    parser.add_argument("--model-path", type=str, default="./deepseek-llm-7b-chat_new", 
                       help="DeepSeek模型路径")
    parser.add_argument("--doc-path", type=str, default="./data/既有知识与动态数据协同智能体0821.docx", 
                       help="医疗知识文档路径")
    parser.add_argument("--interactive", action="store_true", 
                       help="启动交互模式")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        return
    
    if not os.path.exists(args.doc_path):
        print(f"错误: 文档路径不存在: {args.doc_path}")
        return
    
    try:
        # 创建agent
        agent = MedicalKnowledgeAgent(args.model_path, args.doc_path)
        
        if args.interactive:
            # 交互模式
            agent.interactive_mode()
        else:
            # 示例use case测试
            test_use_case = "患者出现胸痛症状，需要快速诊断是否为急性心肌梗死"
            print(f"测试use case: {test_use_case}")
            recommendation = agent.recommend_knowledge_graph(test_use_case)
            print("\n推荐结果:")
            print(recommendation)
            
    except Exception as e:
        print(f"Agent初始化失败: {e}")


if __name__ == "__main__":
    main()
