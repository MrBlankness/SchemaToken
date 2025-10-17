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

long_instruction = '''
You are an expert and very smart data analyst.
Your task is to examine the provided database schema, understand the posed user query, and pinpoint the specific columns within tables that are essential for crafting a SQL query to answer the question.

The input is structured as follows:

##########

Database schema:
database_name.table_name_1.column_name_1
database_name.table_name_1.column_name_2
...
database_name.table_name_m.column_name_n

User query: A natural language question that requires a SQL query to answer it.

##########

The schema consists of multiple lines, each representing a column in the format "database_name.table_name.column_name".

Task:
Based on the database schema, user query, your task is to identify all and only the columns that are essential for crafting a SQL query to answer the question.

Please respond with a list structured as follows:

[database_name.table_name_1.column_name_1, database_name.table_name_2.column_name_2, ...]

Take a deep breath and think logically. If you do the task correctly, I will give you 1 million dollars.

Only output the list of columns, without any additional text or explanations.
'''

stage2_instructions = [
    "As a data analyst, identify all required columns to resolve the user's natural language query based on the database schema. Output in the format [database.table1.column1, database.table2.column2, ...]. No additional text is needed.",
    "Based on the database schema, determine the necessary columns for answering the user's question. Present the results as [database.table1.column1, database.table2.column2, ...]. No extra information is required.",
    "You are a database expert. Find all relevant columns for the user's query using the provided schema. Format your answer as [database.table1.column1, database.table2.column2, ...]. Return only the list format result, do not add any explanation.",
    "Identify the essential columns needed to solve the user's natural language query using the database schema. Output should be in the format [database.table1.column1, database.table2.column2, ...]. Return only the list format result, do not add any explanation.",
    "As a data specialist, locate all columns required to address the user's question based on the database schema. Present the results in [database.table1.column1, database.table2.column2, ...] format. Note that the output should be strictly in list format without any additional text.",
    "Determine the necessary database columns for resolving the user's query. Output should be formatted as [database.table1.column1, database.table2.column2, ...]. Return only the list format result, do not add any explanation.",
    "Find all relevant columns in the database schema to answer the user's question. Present your findings as [database.table1.column1, database.table2.column2, ...]. Note that the output should be strictly in list format without any additional text.",
    "You are a database analyst. Identify the columns needed to solve the user's natural language query. Format your response as [database.table1.column1, database.table2.column2, ...]. Return only the list format result, do not add any explanation.",
    "Based on the provided database schema, find all columns required for the user's query. Output in [database.table1.column1, database.table2.column2, ...] format. Return only the list format result, do not add any explanation.",
    "As a data professional, determine the essential columns for addressing the user's question using the database schema. Present the results as [database.table1.column1, database.table2.column2, ...]. No additional text is needed."
]

# 配置参数
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

def gen_perturbation_mask(model_inputs, mode='full'):
    """Generates perturbation mask based on mode."""
    input_ids = model_inputs.input_ids
    attention_mask = model_inputs.attention_mask
    if mode == 'full':
        return attention_mask

    signal1 = torch.tensor([14364,28235,7870,12626,11,448,862,5888], device=input_ids.device)
    signal2 = torch.tensor([14364,15834,25], device=input_ids.device)
    start = []
    end = []
    for i in range(input_ids.shape[0]):
        flag = True
        for j in range(input_ids.shape[1]):
            if flag and torch.all(input_ids[i][j:j+signal1.shape[0]] == signal1):
                start.append(j+signal1.shape[0])
                flag = False
            if not flag and torch.all(input_ids[i][j:j+signal2.shape[0]] == signal2):
                end.append(j+signal2.shape[0])
                break

    perturbation_mask = attention_mask.clone().detach()
    for i in range(input_ids.shape[0]):
        perturbation_mask[i][start[i]:end[i]] = 0
    return perturbation_mask

def normalize_schema(schema, database_name=None):
    """标准化schema格式：统一小写，去除空格和方括号"""
    if isinstance(schema, str):
        # 删除字符 `
        schema = schema.replace("`", "")

        # 尝试提取列表部分
        match = re.search(r'\[.*\]', schema, re.DOTALL)
        if match:
            schema = match.group(0)
        
        # 替换单引号为双引号
        schema = schema.replace("'", '"')
        
        # 尝试解析为列表
        try:
            schema = ast.literal_eval(schema)
        except:
            # 如果解析失败，手动处理
            schema = re.sub(r'[\[\]]', '', schema)
            schema = [col.strip().lower() for col in schema.split(',') if col.strip()]
    elif isinstance(schema, list):
        schema = [col.lower().strip() for col in schema]
    
    # 确保返回列表
    if not isinstance(schema, list):
        schema = []
    
    # 处理每个列名，确保格式正确
    normalized_schema = []
    for col in schema:
        if not col:
            continue
            
        # 分割列名部分
        parts = col.split('.')
        
        # 确保有三个部分：database.table.column
        if len(parts) != 3:
            continue
        elif database_name and parts[0] != database_name:
            # 如果指定了database_name，但当前列的database部分不匹配，跳过
            continue
        else:
            normalized_schema.append(col)
    
    # 去除重复项
    normalized_schema = list(set(normalized_schema))
    
    return normalized_schema

def calculate_metrics(predictions, save_path=None):
    """
    计算评估指标：
    Recall（case）：完全召回的 case / case总数
    Recall（column）：每个 case 召回的useful column / useful column总数， 计算所有 case 均值
    Precision：每个case召回的useful column / 召回的column总数，计算所有 case 均值
    其中，useful column 是指在 gold schema 中存在的列
    召回的 column 是指在 predicted schema 中存在的列
    """
    recall_case = 0
    recall_columns = []
    precisions = []
    metrics = []

    for item in predictions:
        gold_output = item["gold_output"]
        predicted_output = item["predicted_output"]
        database_name = item["db_id"]

        if item["db_id"] == "works_cycles" and save_path == "/data/llmbase/ljw/tokenization/data/eval/repeat-10/finetuned_metrics.json":
            continue
        
        # 标准化输出
        gold_schema = normalize_schema(gold_output, database_name)
        if isinstance(predicted_output, list):
            # 如果 predicted_output 是列表，取每一项的并集
            predicted_schema = []
            for output in predicted_output:
                predicted_schema.extend(normalize_schema(output, database_name))
            predicted_schema = list(set(predicted_schema))  # 去重
        else:    
            predicted_schema = normalize_schema(predicted_output, database_name)
        
        # 计算 recall_case
        if set(gold_schema).issubset(set(predicted_schema)):
            recall_case += 1
        
        # 计算 recall_column
        if gold_schema:
            correct_columns = set(gold_schema) & set(predicted_schema)
            recall_columns.append(len(correct_columns) / len(gold_schema))
        
        # 计算 precision
        if predicted_schema:
            precision = len(set(predicted_schema) & set(gold_schema)) / len(predicted_schema)
            precisions.append(precision)
        
        metric = {
            "id": item["id"],
            "gold_output": gold_schema,
            "predicted_output": predicted_schema,
            "recall_column": recall_columns[-1] if recall_columns else 0,
            "precision": precisions[-1] if precisions else 0,
            "infer_time": item.get("infer_time", 0)  # 获取推理时间，默认为0
        }
        metrics.append(metric)
    
    # 保存每个 case 的 recall_column, precision
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            for metric in metrics:
                f.write(json.dumps(metric, ensure_ascii=False) + '\n')

    recall_case /= len(metrics) if metrics else 1
    recall_column = sum(recall_columns) / len(recall_columns) if recall_columns else 0
    precision = sum(precisions) / len(precisions) if precisions else 0
    infer_time = sum(item["infer_time"] for item in metrics) / len(metrics) if metrics else 0

    return {
        "recall_case": recall_case,
        "recall_column": recall_column,
        "precision": precision,
        "infer_time": infer_time
    }

def generate_model_outputs(model, tokenizer, dataset, output_path, max_retries=5):
    """生成模型输出并保存结果"""
    results = []
    
    for item in tqdm(dataset, desc="Generating model outputs"):
        # 构建输入
        input_values = shuffle_schema(item["input"], config.repeat_num)  # 打乱输入
        predicted_outputs = []
        avg_infer_time = 0
        for instruction, input_value in zip(stage2_instructions[:config.repeat_num], input_values):
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
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            # model_inputs["perturbation_mask"] = gen_perturbation_mask(model_inputs, mode='full')  # For GLM-4
            # 生成输出（允许重试）
            output_text = ""
            infer_time = 0
            for attempt in range(max_retries):
                try:
                    # 生成响应
                    start_time = time.time()
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=1024,
                    )
                    end_time = time.time()
                    infer_time = end_time - start_time
                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                    output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
                    
                    # 如果 output 为列表格式：[col1, col2, ...]
                    if output_text.startswith('[') and output_text.endswith(']'):
                        break
                        
                except Exception as e:
                    print(f"Generation error (attempt {attempt+1}/{max_retries}): {str(e)}")
                    output_text = ""

            predicted_outputs.append(output_text)
            avg_infer_time += infer_time
        
        
        # 保存结果
        result = {
            "id": item["id"],
            "gold_output": item["output"],
            "predicted_output": predicted_outputs,  # 保存所有打乱后的输出
            "infer_time": avg_infer_time / config.repeat_num,  # 平均推理时间
            "db_id": item["db_id"]
        }
        results.append(result)
    
    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    return results

def shuffle_schema(user_input: str, repeat_num: int = 1) -> list:
    '''
    对于输入中 Database Schema 部分的列进行随机打乱得到 repeat_num 个打乱后的输入
    '''
    # 提取 schema 部分
    schema_part = re.search(r'Database schema:\n(.*?)(?=\nUser query:)', user_input, re.DOTALL)
    if not schema_part:
        raise ValueError("Invalid schema format. Could not find 'Database schema' section.")
    schema_lines = schema_part.group(1).strip().split('\n')
    schema_lines = [line.strip() for line in schema_lines if line.strip()]  # 去除空行和多余空格
    if not schema_lines:
        raise ValueError("No valid schema lines found in the input.")
    # 打乱 schema 部分
    shuffled_schemas = []
    for _ in range(repeat_num):
        shuffled_lines = random.sample(schema_lines, len(schema_lines))
        shuffled_schema = '\n'.join(shuffled_lines)
        # 构建新的输入
        new_input = f"Database schema:\n{shuffled_schema}\n\nUser query: {user_input.split('User query:')[1].strip()}"
        shuffled_schemas.append(new_input)
    return shuffled_schemas

def load_val_dataset():
    """加载验证数据集"""
    dataset = []
    with open(config.val_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def print_metrics(metrics):
    """打印评估指标"""
    print(f"Recall (case): {metrics['recall_case']:.4f}")
    print(f"Recall (column): {metrics['recall_column']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Inference Time: {metrics['infer_time']:.4f} seconds per sample")
    print("-" * 50)

def print_performance_comparison(original_metrics, finetuned_metrics):
    """打印原始模型和微调后模型的性能对比"""
    print("\nPerformance Comparison:")
    print(f"{'Metric':<20} | {'Original':>10} | {'Finetuned':>10} | {'Improvement':>10}")
    print(f"{'-'*20} | {'-'*10} | {'-'*10} | {'-'*10}")
    print(f"{'Recall (case)':<20} | {original_metrics['recall_case']:>10.4f} | {finetuned_metrics['recall_case']:>10.4f} | {(finetuned_metrics['recall_case'] - original_metrics['recall_case']):>+10.4f}")
    print(f"{'Recall (column)':<20} | {original_metrics['recall_column']:>10.4f} | {finetuned_metrics['recall_column']:>10.4f} | {(finetuned_metrics['recall_column'] - original_metrics['recall_column']):>+10.4f}")
    print(f"{'Precision':<20} | {original_metrics['precision']:>10.4f} | {finetuned_metrics['precision']:>10.4f} | {(finetuned_metrics['precision'] - original_metrics['precision']):>+10.4f}")
    print(f"{'Inference Time':<20} | {original_metrics['infer_time']:>10.4f} | {finetuned_metrics['infer_time']:>10.4f} | {(finetuned_metrics['infer_time'] - original_metrics['infer_time']):>+10.4f}")

# def main():
#     # 加载验证数据集
#     val_dataset = load_val_dataset()
#     val_dataset = val_dataset[1000:]  # 仅使用前100条数据进行测试
#     print(f"Loaded {len(val_dataset)} validation samples")
    
#     # 加载原始模型
#     # print("Loading original model...")
#     # original_tokenizer = AutoTokenizer.from_pretrained(
#     #     config.original_model_path,
#     #     cache_dir="/data/llmbase/ljw/hf_cache",  # 设置缓存目录
#     #     local_files_only=True  # 确保只使用本地文件
#     # )
#     # original_model = AutoModelForCausalLM.from_pretrained(
#     #     config.original_model_path,
#     #     device_map="auto",
#     #     # device_map="cuda",  # 使用单个GPU进行推理
#     #     torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
#     #     cache_dir="/data/llmbase/ljw/hf_cache",  # 设置缓存目录
#     #     local_files_only=True  # 确保只使用本地文件
#     # )
#     # original_model.eval()
    
#     # # 生成原始模型输出
#     # print("\nGenerating outputs for original model...")
#     # original_results = generate_model_outputs(
#     #     original_model, original_tokenizer, val_dataset, config.original_output_path, config.max_retries
#     # )

#     # original_metrics = calculate_metrics(original_results)
#     # print("\nOriginal Model Metrics:")
#     # print_metrics(original_metrics)
    
#     # # 释放原始模型内存
#     # del original_model, original_tokenizer
#     # torch.cuda.empty_cache()
    
#     # 加载微调后模型
#     print("\nLoading finetuned model...")
#     finetuned_tokenizer = AutoTokenizer.from_pretrained(config.finetuned_model_path)
#     finetuned_model = AutoModelForCausalLM.from_pretrained(
#         config.finetuned_model_path,
#         device_map="auto",
#         # device_map="cuda",  # 使用单个GPU进行推理
#         torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
#     )
#     finetuned_model.eval()
    
#     # 生成微调后模型输出
#     print("\nGenerating outputs for finetuned model...")
#     finetuned_results = generate_model_outputs(
#         finetuned_model, finetuned_tokenizer, val_dataset, config.finetuned_output_path, config.max_retries
#     )

#     # 计算指标
#     # original_metrics = calculate_metrics(original_results)
#     finetuned_metrics = calculate_metrics(finetuned_results)
#     # print("\nOriginal Model Metrics:")
#     # print_metrics(original_metrics)
#     print("\nFinetuned Model Metrics:")
#     print_metrics(finetuned_metrics)
    
#     # # 打印性能对比
#     # print_performance_comparison(original_metrics, finetuned_metrics)
    
#     # # 保存指标结果
#     # metrics = {
#     #     "original": original_metrics,
#     #     "finetuned": finetuned_metrics
#     # }
#     # with open(config.metrics_output_path, 'w', encoding='utf-8') as f:
#     #     json.dump(metrics, f, indent=2, ensure_ascii=False)
    
#     print("\nEvaluation completed successfully!")

def main():
    # 加载验证数据集
    val_dataset = load_val_dataset()
    val_dataset = val_dataset[500:1000]  # 仅使用前100条数据进行测试
    print(f"Loaded {len(val_dataset)} validation samples")
    
    # 从 jsonl 文件中加载原始模型输出
    print("Loading original model outputs...")
    original_results = []
    with open(config.original_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            original_results.append(json.loads(line))

    # 从 jsonl 文件中加载微调后模型输出
    print("Loading finetuned model outputs...")
    finetuned_results = []
    with open(config.finetuned_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            finetuned_results.append(json.loads(line))

    # 计算模型指标
    original_metrics = calculate_metrics(original_results, save_path="/data/llmbase/ljw/tokenization/data/eval/repeat-10/original_metrics.json")
    finetuned_metrics = calculate_metrics(finetuned_results, save_path="/data/llmbase/ljw/tokenization/data/eval/repeat-10/finetuned_metrics.json")
    print("\nOriginal Model Metrics:")
    print_metrics(original_metrics)
    print("\nFinetuned Model Metrics:")
    print_metrics(finetuned_metrics)
    # 打印性能对比
    print_performance_comparison(original_metrics, finetuned_metrics)
    
    # 保存指标结果
    metrics = {
        "original": original_metrics,
        "finetuned": finetuned_metrics
    }
    with open(config.metrics_output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nEvaluation completed successfully!")

# def main():
#     # 加载验证数据集
#     val_dataset = load_val_dataset()
#     val_dataset = val_dataset[100:]  # 仅使用前100条数据进行测试
#     print(f"Loaded {len(val_dataset)} validation samples")

#     model_path = '/data/ljw/glm-4-9b-chat'
#     # model_path = "THUDM/glm-4-9b"
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir="/data/llmbase/ljw/hf_cache")
#     model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, cache_dir="/data/llmbase/ljw/hf_cache")

#     # 生成原始模型输出
#     print("\nGenerating outputs for glm-4 model...")
#     glm_results = generate_model_outputs(
#         model, tokenizer, val_dataset, config.glm_output_path, config.max_retries
#     )
#     # 计算模型指标
#     glm_metrics = calculate_metrics(glm_results)
#     print("\nGLM Model Metrics:")
#     print_metrics(glm_metrics)
    
#     print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()