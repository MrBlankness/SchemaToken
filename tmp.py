import json
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import re
import ast
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time
import random

class Config:
    def __init__(self):
        # 原始模型配置
        self.original_model_path = "Qwen/Qwen3-8B"  # 原始模型路径
        
        # 微调后模型配置
        self.finetuned_model_path = "/data/llmbase/ljw/tokenization/models/tuned-qwen3-8b/stage-2"  # 微调后模型路径
        
        # 数据配置
        self.val_dataset_path = "/data/llmbase/ljw/tokenization/data/eval/val_data.jsonl"
        
        # 输出文件配置
        self.original_output_path = "/data/llmbase/ljw/tokenization/data/eval/repeat-10/original_output.jsonl"
        self.finetuned_output_path = "/data/llmbase/ljw/tokenization/data/eval/repeat-10/finetuned_output.jsonl"
        self.glm_output_path = "/data/llmbase/ljw/tokenization/data/eval/repeat-10-case-100-end/glm_output.jsonl"
        self.metrics_output_path = "/data/llmbase/ljw/tokenization/data/eval/repeat-10/metrics.json"
        
        # 评估配置
        self.max_retries = 5  # 最大重试次数
        self.repeat_num = 10

config = Config()

def calculate_token_ids(tokenizer, dataset):
    """生成模型输出并保存结果"""
    token_nums = []
    
    for item in tqdm(dataset, desc="Generating model outputs"):
        # 构建输入
        instruction = item["instruction"]
        input_value = item["input"]
        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        text = "who are you?"
        model_inputs = tokenizer([text], return_tensors="pt")

        # 打印分词结果（打印实际token，不要打印序号）
        tokens = tokenizer.convert_ids_to_tokens(model_inputs.input_ids[0])
        print(f"Tokens: {tokens}")
        print(f"Input IDs: {model_inputs.input_ids[0]}")

        token_nums.append(len(model_inputs.input_ids[0]))

    avg_tokens = sum(token_nums) / len(token_nums) if token_nums else 0

    return avg_tokens

def load_val_dataset():
    """加载验证数据集"""
    dataset = []
    with open("/data/llmbase/ljw/tokenization/data/eval/val_data.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def main():
    # 加载验证数据集
    val_dataset = load_val_dataset()
    val_dataset = val_dataset[757:758]  # 仅使用前100条数据进行测试
    print(f"Loaded {len(val_dataset)} validation samples")
    
    # 加载原始模型
    print("Loading original model...")
    original_tokenizer = AutoTokenizer.from_pretrained(
        config.original_model_path,
        cache_dir="/data/llmbase/ljw/hf_cache",  # 设置缓存目录
        local_files_only=True  # 确保只使用本地文件
    )
    # 加载微调后模型
    print("\nLoading finetuned model...")
    finetuned_tokenizer = AutoTokenizer.from_pretrained(config.finetuned_model_path)

    original_avg_tokens = calculate_token_ids(original_tokenizer, val_dataset)
    print(f"Original model average token count: {original_avg_tokens:.2f}")
    finetuned_avg_tokens = calculate_token_ids(finetuned_tokenizer, val_dataset)
    print(f"Finetuned model average token count: {finetuned_avg_tokens:.2f}")

if __name__ == "__main__":
    main()