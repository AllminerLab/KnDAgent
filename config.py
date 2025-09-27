#!/usr/bin/env python3
"""
Medical Knowledge Graph Recommendation Agent configuration file
"""

# Model configuration
MODEL_CONFIG = {
    "api_url": "https://api.deepseek.com/v1/chat/completions",
    "api_key": "your_api_key_here",
    "model_name": "deepseek-chat",
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9
}

# Generation parameters configuration
GENERATION_CONFIG = {
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1
}

# RAG system configuration
RAG_CONFIG = {
    "embedding_model": "bert",  # Use BERT model
    "bert_model_path": "./txt_tokenizer",  # Local BERT model path
    "bert_tokenizer_path": "./txt_tokenizer",  # Local BERT tokenizer path
    "offline_model_path": "./models/",  # Local model path
    "fallback_model": "all-MiniLM-L6-v2",  # Fallback model
    "use_offline": True,  # Prefer offline models
    "chunk_size": 512,
    "top_k_retrieval": 3,
    "similarity_threshold": 0.3,
    "max_length": 512,  # BERT max input length
    "batch_size": 8  # Batch processing size
}

# Document processing configuration
DOC_CONFIG = {
    "doc_path": "./data/KnDAgent.docx",
    "max_section_length": 1000,
    "min_chunk_length": 50
}

# Knowledge graph configuration
KNOWLEDGE_GRAPH_CONFIG = {
    "default_graphs": [
        {
            "name": "established medical knowledge graph",
            "description": "A static knowledge graph constructed based on authoritative medical guidelines, pharmacopoeias, and literature, with diseases and medications as core entities, representing clinical diagnosis and treatment logic through structured triples",
            "domain": "Clinical decision support in emergency departments",
            "advantages": ["Authoritativeness_Assurance", "Structured_Reasoning", "Noise_Robustness", "Standardization_Compatibility"],
            "limitations": ["Update_Lag", "lack_of_Personalization", "Coverage_Bottleneck", "Static_Nature_Constraints"],
            "use_cases": ["Clinical decision support", "Medication safety", "Treatment planning"],
            "data_sources": ["Medical guidelines", "Pharmacopoeias", "Clinical literature"]
        },
        {
            "name": "dynamic clinical data knowledge graph",
            "description": "A temporal evolution graph constructed based on patients' longitudinal electronic medical records (EMR), using 'patient-visit' as dual indexing to dynamically record individualized clinical event chains.",
            "domain": "Personalized decision support in emergency departments.",
            "advantages": ["Personalized_modeling", "Real-time_capability", "Fine-grained_temporal_analysis", "Sparse_data_completion", "Low-frequency_pattern_mining"],
            "limitations": ["Data_heterogeneity", "Noise_sensitivity", "Short-term_bias", "Privacy_constraints"],
            "use_cases": ["Personalized treatment", "Temporal analysis", "Patient monitoring"],
            "data_sources": ["Electronic medical records", "Patient visit data", "Clinical events"]
        },
        {
            "name": "hybrid graph",
            "description": "A collaborative knowledge base constructed by integrating existing Medical Knowledge Graph (EM-KG) and Dynamic Clinical Data Knowledge Graph (DCD-KG), enabling complementary enhancement between authoritative medical rules and individualized clinical experience",
            "domain": "Multimodal decision support in emergency departments.",
            "advantages": ["Knowledge_complementarity", "Task_adaptability", "Robustness_enhancement"],
            "limitations": ["Fusion_noise", "Computational_complexity", "Knowledge_conflict_arbitration", "Interpretability_challenges"],
            "use_cases": ["Comprehensive decision support", "Knowledge integration", "Multimodal analysis"],
            "data_sources": ["Combined knowledge bases", "Multiple data sources", "Integrated systems"]
        }
    ]
}

# Emergency knowledge base configuration
EMERGENCY_KNOWLEDGE_CONFIG = {
    "knowledge_base_path": "./data/emergency_knowledge.json",
    "default_knowledge": [
        {
            "id": "emergency_triage_1",
            "title": "Emergency Triage Basic Principles",
            "content": "Emergency triage is the process of assigning patients to different priority treatment areas based on the urgency and severity of their condition. Basic principles include: 1) Vital signs assessment priority; 2) Symptom severity grading; 3) Time sensitivity consideration; 4) Resource allocation optimization.",
            "keywords": ["emergency triage", "vital signs", "priority", "severity", "time sensitivity"],
            "category": "triage_principles"
        },
        {
            "id": "emergency_triage_2", 
            "title": "Emergency Triage Classification Standards",
            "content": "According to international standards, emergency triage is typically divided into 5 levels: Level 1 (Critical) - immediate rescue; Level 2 (Severe) - treatment within 10 minutes; Level 3 (Urgent) - treatment within 30 minutes; Level 4 (Less urgent) - treatment within 60 minutes; Level 5 (Non-urgent) - treatment within 120 minutes.",
            "keywords": ["triage classification", "critical", "severe", "urgent", "less urgent", "non-urgent"],
            "category": "classification_standards"
        },
        {
            "id": "readmission_prediction_1",
            "title": "Readmission Risk Factors",
            "content": "Key risk factors for hospital readmission include: 1) Previous admission history; 2) Chronic disease burden; 3) Age and comorbidities; 4) Social determinants of health; 5) Medication adherence; 6) Follow-up care quality.",
            "keywords": ["readmission", "risk factors", "chronic disease", "comorbidities", "medication adherence"],
            "category": "readmission_prediction"
        },
        {
            "id": "medication_recommendation_1",
            "title": "Medication Recommendation Principles",
            "content": "Medication recommendation principles include: 1) Evidence-based selection; 2) Patient-specific factors; 3) Drug interactions consideration; 4) Dosage optimization; 5) Safety monitoring; 6) Cost-effectiveness analysis.",
            "keywords": ["medication recommendation", "evidence-based", "patient-specific", "drug interactions", "dosage optimization"],
            "category": "medication_recommendation"
        },
        {
            "id": "ddi_detection_1",
            "title": "Drug-Drug Interaction (DDI) Detection",
            "content": "DDI detection involves: 1) Pharmacokinetic interactions (absorption, distribution, metabolism, excretion); 2) Pharmacodynamic interactions (synergistic, antagonistic effects); 3) Severity classification (minor, moderate, major, contraindicated); 4) Clinical significance assessment; 5) Management strategies and monitoring recommendations.",
            "keywords": ["DDI", "drug-drug interaction", "pharmacokinetic", "pharmacodynamic", "severity classification", "clinical significance"],
            "category": "ddi_detection"
        }
    ]
}

# Decision support tools configuration
DECISION_SUPPORT_TOOLS_CONFIG = {
    "tools_path": "./data/decision_tools.json",
    "default_tools": [
        {
            "id": "mlp_classifier",
            "name": "Multi-Layer Perceptron Classifier",
            "description": "Neural network-based classifier for complex pattern recognition in medical data",
            "input_requirements": ["Numerical features", "Categorical features", "Patient demographics", "Clinical measurements"],
            "output_format": "Classification probability + Confidence score + Feature importance",
            "applicable_scenarios": ["Emergency triage", "Readmission prediction", "Medication recommendation", "DDI detection"],
            "accuracy": "92%",
            "processing_time": "< 30 seconds"
        },
        {
            "id": "xgboost_classifier",
            "name": "XGBoost Classifier",
            "description": "Gradient boosting framework for high-performance classification with feature importance analysis",
            "input_requirements": ["Structured data", "Numerical features", "Categorical features", "Target labels"],
            "output_format": "Classification result + Probability distribution + Feature importance ranking",
            "applicable_scenarios": ["Emergency triage", "Readmission prediction", "Medication recommendation", "DDI detection"],
            "accuracy": "94%",
            "processing_time": "< 20 seconds"
        },
        {
            "id": "lightgbm_classifier",
            "name": "LightGBM Classifier",
            "description": "Light gradient boosting machine for fast and accurate classification with low memory usage",
            "input_requirements": ["Structured data", "Numerical features", "Categorical features", "Target labels"],
            "output_format": "Classification result + Probability distribution + Feature importance ranking",
            "applicable_scenarios": ["Emergency triage", "Readmission prediction", "Medication recommendation", "DDI detection"],
            "accuracy": "93%",
            "processing_time": "< 15 seconds"
        },
        {
            "id": "random_forest_classifier",
            "name": "Random Forest Classifier",
            "description": "Ensemble learning method using multiple decision trees for robust classification",
            "input_requirements": ["Structured data", "Numerical features", "Categorical features", "Target labels"],
            "output_format": "Classification result + Probability distribution + Feature importance ranking",
            "applicable_scenarios": ["Emergency triage", "Readmission prediction", "Medication recommendation", "DDI detection"],
            "accuracy": "91%",
            "processing_time": "< 25 seconds"
        },
        {
            "id": "logistic_regression",
            "name": "Logistic Regression",
            "description": "Linear classification model for interpretable binary and multiclass classification",
            "input_requirements": ["Numerical features", "Categorical features (encoded)", "Target labels"],
            "output_format": "Classification result + Probability distribution + Coefficient analysis",
            "applicable_scenarios": ["Emergency triage", "Readmission prediction", "Medication recommendation", "DDI detection"],
            "accuracy": "89%",
            "processing_time": "< 10 seconds"
        }
    ]
}

# Prompt templates configuration
PROMPT_TEMPLATES = {
    "graph_selection": """You are an expert in medical knowledge graphs and emergency department decision support. Based on the following task information and available knowledge graphs, please select the most suitable knowledge graph for the emergency task.

Task Information:
{task_info}

Available Knowledge Graphs:
{graph_info}

Please respond in the following format:

## Selected Knowledge Graph
**Graph Name**: [Name of the selected knowledge graph]

## Selection Rationale
1. **Task Alignment**: [How this graph aligns with the emergency task requirements]
2. **Capability Match**: [Why this graph's capabilities are most suitable]
3. **Data Requirements**: [How this graph meets the data needs of the task]

## Expected Benefits
[What benefits this graph will provide for the emergency task]""",

    "tool_selection": """You are an expert in emergency department clinical decision support tools. Based on the selected knowledge graph and available tools, please select the most appropriate tools for the emergency task.

Selected Knowledge Graph: {selected_graph}

Task Information: {task_info}

Available Tools:
{tool_info}

Please respond in the following format:

## Selected Tools
**Primary Tool**: [Name of the primary tool]
**Secondary Tools**: [Names of secondary tools, if any]

## Tool Selection Rationale
1. **Task Requirements**: [How the selected tools meet the task requirements]
2. **Graph Integration**: [How the tools work with the selected knowledge graph]
3. **Workflow Efficiency**: [How the tools improve the emergency workflow]

## Implementation Strategy
[How to implement and use the selected tools effectively]"""
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "medical_agent.log"
}
