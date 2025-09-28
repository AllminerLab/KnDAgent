# KnDAgent 使用说明

## 🚀 快速启动

### 1. 环境准备
确保已安装Python 3.8+和必要的依赖包：
```bash
pip install -r requirements.txt
```

### 2. 启动KnDAgent
```bash
python start_knagent.py
```

## 💡 使用方法

### 交互模式
启动后，系统会进入交互模式，你可以：

1. **直接输入医疗use case**，获取知识图谱推荐
2. **输入 `analyze` + use case**，获取详细分析
3. **输入 `quit` 或 `exit`**，退出程序

### 示例对话
```
🏥 KnDAgent - Knowledge Graph Agent
============================================================
功能说明:
1. 输入医疗use case，获取知识图谱推荐
2. 输入 'analyze' + use case，获取详细分析
3. 输入 'quit' 或 'exit' 退出程序
============================================================

请输入医疗use case: 患者出现胸痛症状，需要快速诊断是否为急性心肌梗死

正在分析use case: 患者出现胸痛症状，需要快速诊断是否为急性心肌梗死

==========================================================
🎯 推荐结果:
==========================================================
## 推荐的知识图谱
**图谱名称**: 医疗实体关系图谱

## 选择理由
1. **适用性**: 该图谱包含疾病、症状、体征等实体关系...
2. **优势**: 关系清晰，查询速度快，支持症状-疾病映射...
3. **覆盖范围**: 涵盖心血管疾病、症状表现、诊断标准等...
4. **数据支持**: 基于医学文献和临床指南，可靠性高...

## 实施建议
建议结合患者的具体症状和体征进行综合分析...
==========================================================
```

## 🔧 高级用法

### 1. 命令行参数
```bash
# 交互模式
python knagent.py --interactive

# 测试特定use case
python knagent.py --test "Emergency triage for patient with chest pain and shortness of breath"

# 使用自定义路径
python knagent.py --api-key your_api_key --doc-path ./your_doc.docx
```

### 2. 程序化调用
```python
from knagent import KnDAgent

# 创建agent
agent = KnDAgent()

# 获取推荐
recommendation = agent.recommend_knowledge_graph("你的医疗use case")
print(recommendation)

# 分析use case
analysis = agent.analyze_use_case("你的医疗use case")
print(analysis)
```

### 3. 批量处理
```python
use_cases = [
    "患者出现胸痛症状，需要快速诊断是否为急性心肌梗死",
    "需要检查患者正在服用的多种药物是否存在相互作用",
    "患者需要进行影像学检查，辅助诊断肺部疾病"
]

for use_case in use_cases:
    recommendation = agent.recommend_knowledge_graph(use_case)
    print(f"Use Case: {use_case}")
    print(f"推荐: {recommendation}")
    print("-" * 50)
```

## 📊 支持的知识图谱类型

| 图谱类型 | 适用领域 | 主要用途 |
|---------|---------|---------|
| 医疗实体关系图谱 | 通用医疗 | 疾病诊断、药物推荐、治疗方案 |
| 时序医疗事件图谱 | 临床决策 | 病程追踪、治疗效果评估、预后分析 |
| 多模态医疗图谱 | 精准医疗 | 影像诊断、基因分析、个性化治疗 |
| 药物知识图谱 | 药学 | 药物选择、剂量计算、相互作用检查 |
| 影像诊断图谱 | 影像学 | 影像解读、病变识别、诊断辅助 |

## ⚙️ 配置调整

### 1. 模型参数
在 `config.py` 中调整：
```python
MODEL_CONFIG = {
    "model_path": "./deepseek-llm-7b-chat_new",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 4096
}
```

### 2. 生成参数
```python
GENERATION_CONFIG = {
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9
}
```

### 3. RAG参数
```python
RAG_CONFIG = {
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "chunk_size": 512,
    "top_k_retrieval": 3,
    "similarity_threshold": 0.3
}
```

## 🧪 测试系统

运行测试脚本验证系统功能：
```bash
python test_knagent.py
```

测试包括：
- 基本功能测试
- RAG系统测试
- 知识图谱库测试
- 模型集成测试

## 🐛 常见问题

### 1. 模型加载失败
- 检查模型路径是否正确
- 确认GPU内存是否充足
- 检查CUDA版本兼容性

### 2. 文档读取失败
- 确认文档格式为.docx
- 检查文档是否损坏
- 验证文件权限

### 3. 嵌入模型加载失败
- 检查网络连接
- 确认sentence-transformers版本
- 尝试使用本地模型

## 📝 日志查看

系统运行日志保存在 `medical_agent.log`：
```bash
tail -f medical_agent.log
```

## 🔍 性能优化

### 1. GPU加速
- 确保CUDA环境正确配置
- 调整 `gpu_memory_utilization` 参数
- 使用多GPU并行处理

### 2. 内存优化
- 调整 `chunk_size` 减少内存占用
- 使用 `top_k_retrieval` 限制检索结果数量

### 3. 响应速度优化
- 预加载嵌入向量
- 使用异步处理
- 实现结果缓存

## 📞 获取帮助

- 查看日志文件了解详细错误信息
- 检查配置文件参数设置
- 运行测试脚本验证系统状态
- 参考README文档了解系统架构
