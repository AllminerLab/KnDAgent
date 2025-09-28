# KnDAgent - Knowledge Graph Agent

基于DeepSeek大模型和RAG技术的医疗知识图谱智能推荐系统。

## 功能特点

- 智能推荐最适合的知识图谱类型
- 支持3种医疗知识图谱
- 支持5种机器学习工具
- 基于RAG技术的文档分析
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
2. **dynamic clinical data knowledge graph** - 基于患者纵向电子病历的动态图谱
3. **hybrid graph** - 整合既有知识和动态数据的混合图谱

## 支持的机器学习工具

1. **MLP** - 多层感知机
2. **XGBoost Classifier** - 梯度提升分类器
3. **LightGBM Classifier** - 轻量级梯度提升分类器
4. **Random Forest Classifier** - 随机森林分类器
5. **Logistic Regression** - 逻辑回归分类器

## 支持的任务类型

- **急诊分诊** (Emergency Triage) - 患者分诊和优先级评估
- **再入院预测** (Readmission Prediction) - 医院再入院风险预测
- **药物推荐** (Medication Recommendation) - 药物推荐和选择
- **DDI检测** (Drug-Drug Interaction Detection) - 药物相互作用识别

## 使用示例

输入急诊任务描述，系统会自动推荐最适合的知识图谱和工具组合。

## 配置说明

详见`config.py`文件，可自定义模型参数、RAG配置等。
