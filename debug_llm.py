#!/usr/bin/env python3
"""
大模型诊断脚本
调试DeepSeek模型输出为空的问题
"""

import os
import sys
from deploy_deepseek_vllm import DeepSeekVLLMServer

def test_simple_generation():
    """测试简单生成"""
    print("🧪 Testing simple text generation...")
    
    try:
        server = DeepSeekVLLMServer("./deepseek-llm-7b-chat_new")
        server.initialize_engine()
        
        # 测试1：非常简单的提示词
        simple_prompt = "Hello"
        print(f"\nTest 1: Simple prompt: '{simple_prompt}'")
        
        response = server.generate_text(
            prompt=simple_prompt,
            max_tokens=10,
            temperature=0.7,
            top_p=0.9
        )
        
        print(f"Response: '{response}'")
        print(f"Response length: {len(response) if response else 0}")
        print(f"Response type: {type(response)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_medical_prompt():
    """测试医疗提示词"""
    print("\n🏥 Testing medical prompt...")
    
    try:
        server = DeepSeekVLLMServer("./deepseek-llm-7b-chat_new")
        server.initialize_engine()
        
        # 测试2：医疗相关提示词
        medical_prompt = """You are a medical expert. Please recommend a knowledge graph for drug interaction prediction.

Please respond in English with a simple answer."""
        
        print(f"Test 2: Medical prompt length: {len(medical_prompt)}")
        
        response = server.generate_text(
            prompt=medical_prompt,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9
        )
        
        print(f"Response: '{response}'")
        print(f"Response length: {len(response) if response else 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_full_prompt():
    """测试完整提示词"""
    print("\n📝 Testing full recommendation prompt...")
    
    try:
        server = DeepSeekVLLMServer("./deepseek-llm-7b-chat_new")
        server.initialize_engine()
        
        # 测试3：完整的推荐提示词
        full_prompt = """You are a professional medical knowledge graph expert. Based on the following medical use case and related information, please recommend the most suitable type of knowledge graph and explain the reasons for your selection. Please respond in English.

Medical Use Case: Prediction Task: Readmission Prediction

Related Background Information:
Based on general medical knowledge

Available Types of Knowledge Graphs:
1. **established medical knowledge graph**
   - Description: A static knowledge graph constructed based on authoritative medical guidelines, pharmacopoeias, and literature, with diseases and medications as core entities, representing clinical diagnosis and treatment logic through structured triples
   - Domain: Clinical decision support in emergency departments
   - Advantages: Authoritativeness_Assurance, Structured_Reasoning, Noise_Robustness, Standardization_Compatibility
   - Use Cases: Clinical decision support, Medication safety, Treatment planning

2. **dynamic clinical data knowledge graph**
   - Description: A temporal evolution graph constructed based on patients' longitudinal electronic medical records (EMR), using 'patient-visit' as dual indexing to dynamically record individualized clinical event chains.
   - Domain: Personalized decision support in emergency departments.
   - Advantages: Personalized_modeling, Real-time_capability, Fine-grained_temporal_analysis, Sparse_data_completion, Low-frequency_pattern_mining
   - Use Cases: Personalized treatment, Temporal analysis, Patient monitoring

3. **hybrid knowledge graph**
   - Description: A collaborative knowledge base constructed by integrating existing Medical Knowledge Graph (EM-KG) and Dynamic Clinical Data Knowledge Graph (DCD-KG), enabling complementary enhancement between authoritative medical rules and individualized clinical experience
   - Domain: Multimodal decision support in emergency departments.
   - Advantages: Knowledge_complementarity, Task_adaptability, Robustness_enhancement
   - Use Cases: Comprehensive decision support, Knowledge integration, Multimodal analysis

Please respond in the following format:

## Recommended Knowledge Graph
**Graph Name**: [Name of the recommended knowledge graph]

## Reasons for Selection
1. **Applicability**: [Why this graph is the most suitable for the use case]
2. **Advantages**: [The main advantages of this graph in this scenario]

## Implementation Suggestions
[Brief implementation suggestions and considerations]
Please ensure the recommendation is based on the specific needs of the use case and fully utilizes the provided background information."""
        
        print(f"Test 3: Full prompt length: {len(full_prompt)}")
        
        response = server.generate_text(
            prompt=full_prompt,
            max_tokens=500,
            temperature=0.7,
            top_p=0.9
        )
        
        print(f"Response: '{response}'")
        print(f"Response length: {len(response) if response else 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_different_parameters():
    """测试不同参数"""
    print("\n⚙️ Testing different generation parameters...")
    
    try:
        server = DeepSeekVLLMServer("./deepseek-llm-7b-chat_new")
        server.initialize_engine()
        
        test_prompt = "Please recommend a knowledge graph for medical prediction tasks."
        
        # 测试不同的参数组合
        param_combinations = [
            {"max_tokens": 50, "temperature": 0.1, "top_p": 0.9},
            {"max_tokens": 100, "temperature": 0.5, "top_p": 0.8},
            {"max_tokens": 200, "temperature": 0.9, "top_p": 0.7},
            {"max_tokens": 500, "temperature": 0.7, "top_p": 0.9}
        ]
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\nTest 4.{i}: Parameters: {params}")
            
            response = server.generate_text(
                prompt=test_prompt,
                **params
            )
            
            print(f"Response: '{response}'")
            print(f"Response length: {len(response) if response else 0}")
            
            if response and len(response) > 0:
                print("✅ This parameter combination works!")
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """主诊断函数"""
    print("🔍 DeepSeek LLM Debug Tool")
    print("=" * 60)
    
    tests = [
        ("Simple Generation", test_simple_generation),
        ("Medical Prompt", test_medical_prompt),
        ("Full Prompt", test_full_prompt),
        ("Parameter Testing", test_different_parameters)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} exception: {e}")
    
    # 诊断结果
    print("\n" + "="*60)
    print("📊 Debug Results")
    print("="*60)
    print(f"Total Tests: {total}")
    print(f"Passed Tests: {passed}")
    print(f"Failed Tests: {total - passed}")
    
    if passed == total:
        print("\n🎉 All tests passed! LLM is working normally")
    else:
        print(f"\n⚠️  {total-passed} tests failed")
        print("\nPossible issues:")
        print("1. Model configuration problems")
        print("2. Prompt format issues")
        print("3. Model loading problems")
        print("4. GPU memory issues")
    
    print("="*60)

if __name__ == "__main__":
    main()
