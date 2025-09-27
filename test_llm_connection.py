#!/usr/bin/env python3
"""
测试大模型连接和生成功能
"""

import os
import sys
from deploy_deepseek_vllm import DeepSeekVLLMServer

def test_llm_basic():
    """测试大模型基本功能"""
    print("🧪 测试大模型基本功能...")
    
    try:
        # 创建服务器实例
        model_path = "./deepseek-llm-7b-chat_new"
        if not os.path.exists(model_path):
            print(f"❌ 模型路径不存在: {model_path}")
            return False
        
        server = DeepSeekVLLMServer(model_path)
        
        # 初始化引擎
        print("正在初始化模型...")
        server.initialize_engine()
        print("✅ 模型初始化成功")
        
        # 测试简单生成
        test_prompt = "请简单介绍一下医疗知识图谱的作用"
        print(f"\n测试提示词: {test_prompt}")
        
        response = server.generate_text(
            prompt=test_prompt,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9
        )
        
        print(f"✅ 生成成功，响应长度: {len(response) if response else 0}")
        print(f"响应内容: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_medical_prompt():
    """测试医疗相关的提示词"""
    print("\n🏥 测试医疗相关提示词...")
    
    try:
        server = DeepSeekVLLMServer("./deepseek-llm-7b-chat_new")
        server.initialize_engine()
        
        # 测试医疗提示词
        medical_prompt = """你是一个专业的医疗知识图谱专家。基于以下医疗use case，请推荐最适合的知识图谱类型，并说明选择理由。

医疗Use Case: 药物相互作用（DDI）预测任务

请按照以下格式回答:

## 推荐的知识图谱
**图谱名称**: [推荐的知识图谱名称]

## 选择理由
1. **适用性**: [为什么这个图谱最适合该use case]
2. **优势**: [该图谱在此场景下的主要优势]
3. **覆盖范围**: [该图谱如何覆盖use case的需求]
4. **数据支持**: [该图谱的数据来源和可靠性]

## 实施建议
[简要的实施建议和注意事项]"""

        print(f"医疗提示词长度: {len(medical_prompt)}")
        
        response = server.generate_text(
            prompt=medical_prompt,
            max_tokens=500,
            temperature=0.7,
            top_p=0.9
        )
        
        print(f"✅ 医疗提示词生成成功，响应长度: {len(response) if response else 0}")
        if response:
            print(f"响应内容:\n{response}")
        else:
            print("❌ 响应为空")
        
        return True
        
    except Exception as e:
        print(f"❌ 医疗提示词测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chat_mode():
    """测试聊天模式"""
    print("\n💬 测试聊天模式...")
    
    try:
        server = DeepSeekVLLMServer("./deepseek-llm-7b-chat_new")
        server.initialize_engine()
        
        messages = [
            {"role": "user", "content": "请推荐一个适合药物相互作用预测的知识图谱"}
        ]
        
        response = server.chat_generate(
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        
        print(f"✅ 聊天模式测试成功，响应长度: {len(response) if response else 0}")
        if response:
            print(f"响应内容:\n{response}")
        else:
            print("❌ 响应为空")
        
        return True
        
    except Exception as e:
        print(f"❌ 聊天模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🤖 大模型连接和生成功能测试")
    print("=" * 60)
    
    tests = [
        ("基本功能", test_llm_basic),
        ("医疗提示词", test_llm_medical_prompt),
        ("聊天模式", test_chat_mode)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}测试通过")
            else:
                print(f"❌ {test_name}测试失败")
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
    
    # 测试结果汇总
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    print(f"总测试数: {total}")
    print(f"通过测试: {passed}")
    print(f"失败测试: {total - passed}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 所有测试通过！大模型工作正常")
        print("现在可以重新启动医疗知识图谱推荐Agent:")
        print("python start_medical_agent.py")
    else:
        print(f"\n⚠️  有{total-passed}个测试失败，请检查大模型配置")
    
    print("="*60)

if __name__ == "__main__":
    main()
