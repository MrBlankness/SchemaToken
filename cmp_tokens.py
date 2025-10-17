import json
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

# 配置参数
class Config:
    def __init__(self):
        # 原始模型配置
        self.original_model_path = "Qwen/Qwen3-8B"  # 原始模型路径
        
        # 微调后模型配置
        self.finetuned_model_path = "/data/llmbase/ljw/tokenization/models/tuned-qwen3-8b/stage-2"  # 微调后模型路径
        
        # 数据配置
        self.val_dataset_path = "/data/llmbase/ljw/tokenization/data/eval/val_data.jsonl"

config = Config()

def count_tokens(tokenizer, text):
    """计算文本的token数量"""
    return len(tokenizer.encode(text))

def compare_token_counts():
    """比较微调前后模型输入token的数量"""
    # 加载验证数据集
    dataset = []
    with open(config.val_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    
    print(f"Loaded {len(dataset)} validation samples")
    
    # 加载原始模型的tokenizer
    print("Loading original model tokenizer...")
    original_tokenizer = AutoTokenizer.from_pretrained(
        config.original_model_path,
        device_map="cuda",
        cache_dir="/data/llmbase/ljw/hf_cache",  # 设置缓存目录
        local_files_only=True
    )
    
    # 加载微调后模型的tokenizer
    print("Loading finetuned model tokenizer...")
    finetuned_tokenizer = AutoTokenizer.from_pretrained(
        config.finetuned_model_path,
        device_map="cuda",
        local_files_only=True
    )
    
    # 统计token数量
    original_counts = []
    finetuned_counts = []
    
    for item in tqdm(dataset, desc="Counting tokens"):
        # 构建输入文本
        messages = [
            {"role": "system", "content": item["instruction"]},
            {"role": "user", "content": item["input"]}
        ]
        
        # 使用tokenizer的chat模板格式化文本
        text = original_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 计算原始tokenizer的token数量
        orig_count = count_tokens(original_tokenizer, text)
        original_counts.append(orig_count)
        
        # 计算微调后tokenizer的token数量
        finetuned_count = count_tokens(finetuned_tokenizer, text)
        finetuned_counts.append(finetuned_count)
    
    # 计算统计指标
    def calculate_stats(counts):
        return {
            "mean": np.mean(counts),
            "median": np.median(counts),
            "min": np.min(counts),
            "max": np.max(counts),
            "std": np.std(counts),
            "total": np.sum(counts)
        }
    
    original_stats = calculate_stats(original_counts)
    finetuned_stats = calculate_stats(finetuned_counts)
    
    # 计算token减少比例
    reduction_rates = []
    for orig, finetuned in zip(original_counts, finetuned_counts):
        if orig > 0:
            reduction_rates.append((orig - finetuned) / orig)
    
    reduction_stats = {
        "mean": np.mean(reduction_rates) * 100,
        "median": np.median(reduction_rates) * 100,
        "min": np.min(reduction_rates) * 100,
        "max": np.max(reduction_rates) * 100,
        "std": np.std(reduction_rates) * 100
    }
    
    # 打印结果
    print("\nToken Count Comparison:")
    print(f"{'Statistic':<15} | {'Original':>10} | {'Finetuned':>10} | {'Reduction (%)':>12}")
    print(f"{'-'*15} | {'-'*10} | {'-'*10} | {'-'*12}")
    print(f"{'Mean':<15} | {original_stats['mean']:>10.1f} | {finetuned_stats['mean']:>10.1f} | {reduction_stats['mean']:>12.2f}")
    print(f"{'Median':<15} | {original_stats['median']:>10.1f} | {finetuned_stats['median']:>10.1f} | {reduction_stats['median']:>12.2f}")
    print(f"{'Min':<15} | {original_stats['min']:>10.1f} | {finetuned_stats['min']:>10.1f} | {reduction_stats['min']:>12.2f}")
    print(f"{'Max':<15} | {original_stats['max']:>10.1f} | {finetuned_stats['max']:>10.1f} | {reduction_stats['max']:>12.2f}")
    print(f"{'Std Dev':<15} | {original_stats['std']:>10.1f} | {finetuned_stats['std']:>10.1f} | {reduction_stats['std']:>12.2f}")
    print(f"{'Total':<15} | {original_stats['total']:>10.1f} | {finetuned_stats['total']:>10.1f} | {reduction_stats['mean']:>12.2f}")
    
    # 保存结果
    results = {
        "original": {
            "token_counts": original_counts,
            "statistics": original_stats
        },
        "finetuned": {
            "token_counts": finetuned_counts,
            "statistics": finetuned_stats
        },
        "reduction": {
            "rates": reduction_rates,
            "statistics": reduction_stats
        }
    }
    
    output_path = "/data/llmbase/ljw/tokenization/data/eval/token_comparison.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    compare_token_counts()