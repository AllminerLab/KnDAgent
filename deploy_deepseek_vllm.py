#!/usr/bin/env python3
"""
DeepSeek模型vLLM部署脚本
"""

import argparse
import asyncio
import json
from typing import List, Optional
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid


class DeepSeekVLLMServer:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
        """
        初始化DeepSeek vLLM服务器
        
        Args:
            model_path: 模型路径
            tensor_parallel_size: 张量并行大小
            gpu_memory_utilization: GPU内存利用率
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.engine = None
        
    def initialize_engine(self):
        """初始化vLLM引擎"""
        print(f"正在加载模型: {self.model_path}")
        
        # 创建LLM实例
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,  # DeepSeek模型需要这个参数
            dtype="auto",  # 自动选择数据类型
            max_model_len=4096,  # 最大模型长度
        )
        
        print("模型加载完成！")
        
    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, 
                     top_p: float = 0.9, stop: Optional[List[str]] = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            stop: 停止词列表
            
        Returns:
            生成的文本
        """
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or ["<|endoftext|>", "<|im_end|>"]
        )
        
        # 生成文本
        outputs = self.llm.generate([prompt], sampling_params)
        
        # 返回生成的文本
        result = outputs[0].outputs[0].text
        
        # 调试信息
        print(f"DEBUG: Generated text length: {len(result) if result else 0}")
        print(f"DEBUG: Generated text: {result[:100] if result else 'None'}...")
        
        return result
    
    def chat_generate(self, messages: List[dict], max_tokens: int = 512, 
                     temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        聊天模式生成
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}, ...]
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            
        Returns:
            生成的回复
        """
        # 构建DeepSeek聊天格式的提示
        prompt = self._build_chat_prompt(messages)
        
        return self.generate_text(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|endoftext|>", "<|im_end|>"]
        )
    
    def _build_chat_prompt(self, messages: List[dict]) -> str:
        """
        构建DeepSeek聊天格式的提示
        
        Args:
            messages: 消息列表
            
        Returns:
            格式化的提示字符串
        """
        prompt = ""
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        
        # 添加assistant的开始标记
        prompt += "<|im_start|>assistant\n"
        
        return prompt


def main():
    parser = argparse.ArgumentParser(description="DeepSeek模型vLLM部署")
    parser.add_argument("--model-path", type=str, default="./deepseek-llm-7b-chat_new", 
                       help="模型路径")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, 
                       help="张量并行大小")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, 
                       help="GPU内存利用率")
    parser.add_argument("--max-tokens", type=int, default=512, 
                       help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="温度参数")
    parser.add_argument("--top-p", type=float, default=0.9, 
                       help="top-p采样参数")
    
    args = parser.parse_args()
    
    # 创建服务器实例
    server = DeepSeekVLLMServer(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # 初始化引擎
    server.initialize_engine()
    
    print("\n=== DeepSeek vLLM 部署成功 ===")
    print("模型已加载，可以开始对话！")
    print("输入 'quit' 或 'exit' 退出程序")
    print("=" * 40)
    
    # 交互式对话
    while True:
        try:
            user_input = input("\n用户: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
                
            if not user_input:
                continue
            
            # 构建消息格式
            messages = [{"role": "user", "content": user_input}]
            
            # 生成回复
            print("助手: ", end="", flush=True)
            response = server.chat_generate(
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n程序被中断，正在退出...")
            break
        except Exception as e:
            print(f"\n错误: {e}")


if __name__ == "__main__":
    main()
