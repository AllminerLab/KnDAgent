# KnDAgent -

## 功能特点

- 智能推荐最适合的知识图谱类型
- 支持3种医疗知识图谱
- 支持5种机器学习工具
- 交互式命令行界面
- 支持急诊分诊、再入院预测、药物推荐、DDI检测

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动KnDAgent
```bash
python start_knagent.py
```

## 支持的知识图谱

1. **established medical knowledge graph** - 基于权威医学指南的静态知识图谱
2. **dynamic clinical data knowledge graph** - 基于患者电子病历的动态图谱
3. **hybrid graph** - 整合既有知识和动态数据的混合图谱

## 支持的机器学习工具

1. **MLP** - 多层感知机
2. **XGBoost Classifier** - 梯度提升分类器
3. **LightGBM Classifier** - 轻量级梯度提升分类器
4. **Random Forest Classifier** - 随机森林分类器
5. **Logistic Regression** - 逻辑回归分类器

## 支持的任务类型

- **急诊分诊** (Emergency Triage) - 患者分诊
- **再入院预测** (Readmission Prediction) - 医院再入院风险预测（7/30）
- **药物推荐** (Medication Recommendation) - 药物推荐和选择
- **DDI检测** (Drug-Drug Interaction Detection) - 药物相互作用识别

## 使用示例

输入急诊任务描述，系统会自动推荐最适合的知识图谱和工具组合。

## 配置说明

详见`config.py`文件，可自定义模型参数、RAG配置等。

# KnDAgent 使用说明

## 🚀 快速启动

### 1. 环境准备
确保已安装Python 3.8+和必要的依赖包：
```bash
pip install -r requirements.txt
```

### 2. 配置API密钥
在 `config.py` 中设置您的DeepSeek API密钥：
```python
MODEL_CONFIG = {
    "api_key": "your_deepseek_api_key_here",
    # ... 其他配置
}
```

### 3. 启动KnDAgent
```bash
python start_knagent.py
```

## 💡 使用方法

### 交互模式
启动后，系统会进入交互模式，支持以下急诊任务：

1. **急诊分诊** - 患者分诊和优先级评估
2. **再入院预测** - 医院再入院风险预测
3. **药物推荐** - 药物推荐和选择
4. **DDI检测** - 药物相互作用识别

### 示例对话
```
🏥 KnDAgent - Knowledge Graph Agent
============================================================
支持的任务类型:
1. 急诊分诊 - 患者分诊和优先级评估
2. 再入院预测 - 医院再入院风险预测
3. 药物推荐 - 药物推荐和选择
4. DDI检测 - 药物相互作用识别
============================================================

请输入急诊任务描述: Emergency triage for patient with chest pain and shortness of breath

正在处理急诊任务: Emergency triage for patient with chest pain and shortness of breath

==========================================================
🎯 执行链结果:
==========================================================
任务ID: task_1234
任务描述: Emergency triage for patient with chest pain and shortness of breath
任务类型: emergency_triage
优先级: high
选中的知识图谱: hybrid graph
选中的工具: [mlp_classifier]
置信度: 0.85

执行策略:
Task: Emergency triage for patient with chest pain and shortness of breath
Task Type: emergency_triage
Priority: high
Selected Knowledge Graph: hybrid graph
Selected Tools: mlp_classifier
Requirements: Patient triage assessment, Vital signs monitoring
==========================================================
```

## 🔧 高级用法

### 1. 命令行参数
```bash
# 交互模式
python knagent.py --interactive

# 测试特定急诊任务
python knagent.py --test "Emergency triage for patient with chest pain and shortness of breath"

# 使用自定义API密钥和文档知识库路径
python knagent.py --api-key your_api_key --doc-path ./your_doc.docx
```

### 2. 程序化调用
```python
from knagent import KnDAgent

# 创建agent
agent = KnDAgent()

# 处理急诊任务
execution_chain = agent.process_emergency_task("Emergency triage for patient with chest pain")
print(f"选中的图谱: {execution_chain.selected_graph}")
print(f"选中的工具: {execution_chain.selected_tools}")
print(f"置信度: {execution_chain.confidence_score}")

# 分析use case（兼容旧接口）
analysis = agent.analyze_use_case("Emergency triage task")
print(analysis)
```

### 3. 批量处理急诊任务
```python
emergency_tasks = [
    "Emergency triage for patient with chest pain and shortness of breath",
    "DDI detection for patient taking warfarin and aspirin",
    "Readmission prediction for elderly patient with heart failure",
    "Medication recommendation for patient with hypertension and diabetes"
]

for task in emergency_tasks:
    execution_chain = agent.process_emergency_task(task)
    print(f"Task: {task}")
    print(f"Selected Graph: {execution_chain.selected_graph}")
    print(f"Selected Tools: {execution_chain.selected_tools}")
    print("-" * 50)
```

## 📊 支持的知识图谱类型

| 图谱类型 | 适用领域 | 主要用途 |
|---------|---------|---------|
| established medical knowledge graph | 通用医疗 | 基于权威医学指南的静态知识图谱 |
| dynamic clinical data knowledge graph | 临床决策 | 基于患者纵向电子病历的动态图谱 |
| hybrid graph | 精准医疗 | 整合既有知识和动态数据的混合图谱 |

## 🤖 支持的机器学习工具

| 工具名称 | 类型 | 主要用途 |
|---------|------|---------|
| MLP | 神经网络 | 多层感知机分类器 |
| XGBoost Classifier | 梯度提升 | 高性能分类与特征重要性分析 |
| LightGBM Classifier | 梯度提升 | 快速准确的轻量级分类器 |
| Random Forest Classifier | 集成学习 | 基于多决策树的鲁棒分类器 |
| Logistic Regression | 线性模型 | 可解释的线性分类器 |

## ⚙️ 配置调整

### 1. API配置
在 `config.py` 中设置DeepSeek API：
```python
MODEL_CONFIG = {
    "api_url": "https://api.deepseek.com/v1/chat/completions",
    "api_key": "your_deepseek_api_key_here",
    "model_name": "deepseek-chat",
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9
}
```

### 2. 生成参数
```python
GENERATION_CONFIG = {
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1
}
```

### 3. RAG参数
```python
RAG_CONFIG = {
    "embedding_model": "bert",
    "chunk_size": 512,
    "top_k_retrieval": 3,
    "similarity_threshold": 0.3,
    "max_length": 512,
    "batch_size": 8
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

### 1. API连接失败
- 检查DeepSeek API密钥是否正确
- 确认网络连接正常
- 验证API配额是否充足

### 2. 文档知识库读取失败
- 确认文档知识库路径为 `./data/KnDAgent.docx`
- 检查文档格式是否为.docx
- 验证文件权限

### 3. 嵌入模型加载失败
- 检查BERT模型路径是否正确
- 确认transformers库版本
- 尝试使用备用模型

## 📝 日志查看

系统运行日志保存在 `KnDAgent.log`：
```bash
tail -f medical_agent.log
```

## 🔍 性能优化

### 1. API调用优化
- 合理设置请求频率
- 使用批量处理减少API调用
- 实现结果缓存机制

### 2. 内存优化
- 调整 `chunk_size` 减少内存占用
- 使用 `top_k_retrieval` 限制检索结果数量
- 优化批处理大小

### 3. 响应速度优化
- 预加载嵌入向量
- 使用异步处理
- 实现结果缓存

## 📞 获取帮助

- 查看日志文件了解详细错误信息
- 检查配置文件参数设置
- 运行测试脚本验证系统状态
- 参考README文档了解系统架构

