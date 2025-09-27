#!/usr/bin/env python3
"""
BERT模型检查脚本
验证本地BERT模型是否正确加载
"""

import os
import sys
import torch
from transformers import BertTokenizer, BertModel

def check_bert_model():
    """检查BERT模型"""
    print("🔍 开始检查BERT模型...")
    
    # 检查模型路径
    bert_path = "./txt_tokenizer"
    tokenizer_path = "./txt_tokenizer"
    
    print(f"BERT模型路径: {bert_path}")
    print(f"分词器路径: {tokenizer_path}")
    
    # 检查路径是否存在
    if not os.path.exists(bert_path):
        print(f"❌ BERT模型路径不存在: {bert_path}")
        return False
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ 分词器路径不存在: {tokenizer_path}")
        return False
    
    # 检查路径内容
    print(f"\n📁 模型目录内容:")
    try:
        files = os.listdir(bert_path)
        for file in files:
            print(f"  - {file}")
    except Exception as e:
        print(f"  无法读取目录: {e}")
        return False
    
    # 尝试加载分词器
    print(f"\n🔤 尝试加载分词器...")
    try:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        print("✅ 分词器加载成功")
        
        # 测试分词
        test_text = "医疗知识图谱"
        tokens = tokenizer.tokenize(test_text)
        print(f"  测试分词: '{test_text}' -> {tokens}")
        
    except Exception as e:
        print(f"❌ 分词器加载失败: {e}")
        return False
    
    # 尝试加载模型
    print(f"\n🤖 尝试加载BERT模型...")
    try:
        model = BertModel.from_pretrained(bert_path)
        print("✅ BERT模型加载成功")
        
        # 检查模型信息
        print(f"  模型类型: {type(model).__name__}")
        print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 检查设备
        device = next(model.parameters()).device
        print(f"  模型设备: {device}")
        
        # 测试推理
        print(f"\n🧪 测试模型推理...")
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            print(f"  输出形状: {embeddings.shape}")
            print(f"  输出类型: {embeddings.dtype}")
        
        print("✅ 模型推理测试成功")
        
    except Exception as e:
        print(f"❌ BERT模型加载失败: {e}")
        return False
    
    print(f"\n🎉 BERT模型检查完成！")
    return True

def main():
    """主函数"""
    print("🏥 BERT模型检查工具")
    print("=" * 50)
    
    try:
        success = check_bert_model()
        
        if success:
            print("\n✅ 所有检查通过！BERT模型可以正常使用")
            print("\n现在可以启动医疗知识图谱推荐Agent:")
            print("python start_medical_agent.py")
        else:
            print("\n❌ 检查失败！请检查模型文件")
            
    except Exception as e:
        print(f"\n💥 检查过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
