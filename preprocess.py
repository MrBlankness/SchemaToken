import os
import json
import csv
import random
import pandas as pd
from collections import defaultdict
import chardet

# 配置参数
class DataConfig:
    def __init__(self):
        # 第一阶段数据配置
        self.stage1_database_dir = "/data/llmbase/ljw/bird/train/train_databases"  # 数据库目录
        self.stage1_output_file = "/data/llmbase/ljw/tokenization/data/stage-1/all_data.jsonl"  # 第一阶段输出文件
        self.stage1_train_file = "/data/llmbase/ljw/tokenization/data/stage-1/train.jsonl"  # 第一阶段训练集
        self.stage1_val_file = "/data/llmbase/ljw/tokenization/data/stage-1/val.jsonl"     # 第一阶段验证集
        
        # 第二阶段数据配置
        self.stage2_schema_dir = "/data/llmbase/ljw/tokenization/data/stage-2/schema_tokens"  # 模式token目录
        self.stage2_questions_file = "/data/llmbase/ljw/tokenization/data/stage-2/raw_data.json"  # 原始数据文件
        self.stage2_output_file = "/data/llmbase/ljw/tokenization/data/stage-2/all_data.jsonl"  # 第二阶段输出文件
        self.stage2_train_file = "/data/llmbase/ljw/tokenization/data/stage-2/train.jsonl"  # 第二阶段训练集
        self.stage2_val_file = "/data/llmbase/ljw/tokenization/data/stage-2/val.jsonl"     # 第二阶段验证集
        
        # 通用配置
        self.train_val_ratio = 0.8  # 训练验证比例

config = DataConfig()

# 第一阶段指令列表（英文）
stage1_instructions = [
    "Generate the description for the given database column. Only output the description without any additional text.",
    "What is the meaning of this database column? Output only the meaning without any extra information.",
    "Provide an explanation for this database column. Only return the explanation without any additional context.",
    "Explain the function of this database column. Output just the description of the column without any extra text.",
    "What does this column represent in the database? Output only the description.",
    "Interpret the meaning of this database column. Only provide the meaning without extra text.",
    "What is the description of this database column? Output only the description without any extra text.",
    "Can you provide a description for this database column? Please output only the description without any additional information.",
    "What does this column in the database signify? Only output the description.",
    "Explain the meaning of this database column. Only answer with the meaning without any additional text.",
]

# 第二阶段指令列表（英文）
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

def generate_stage1_data():
    """生成第一阶段数据：列名到描述"""
    print("Generating Stage 1 data...")
    
    # 收集所有列信息
    all_columns = []
    
    # 遍历数据库目录
    for database_name in os.listdir(config.stage1_database_dir):
        database_path = os.path.join(config.stage1_database_dir, database_name)
        
        # 检查是否是目录
        if not os.path.isdir(database_path):
            continue
            
        # 查找database_description目录
        desc_dir = os.path.join(database_path, "database_description")
        if not os.path.exists(desc_dir):
            print(f"Warning: database_description directory not found in {database_path}")
            continue
            
        # 遍历所有CSV文件
        for table_file in os.listdir(desc_dir):
            if not table_file.endswith(".csv"):
                continue
                
            table_name = os.path.splitext(table_file)[0]
            table_path = os.path.join(desc_dir, table_file)

            try:
                # 检测文件编码
                with open(table_path, 'rb') as f:
                    raw_data = f.read(4096)  # 读取部分内容用于检测编码
                    result = chardet.detect(raw_data)
                    encoding = result['encoding'] or 'utf-8'
                
                # 使用检测到的编码打开文件
                with open(table_path, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # 获取列名
                        column_name = row.get('original_column_name') or row.get('column_name')
                        if not column_name:
                            continue
                            
                        # 获取描述
                        description = row.get('column_description') or column_name
                        
                        # 存储列信息
                        column_id = f"{database_name}.{table_name}.{column_name}"
                        all_columns.append({
                            "db_id": database_name,
                            "table": table_name,
                            "column": column_name,
                            "description": description,
                            "id": column_id
                        })
            except Exception as e:
                print(f"Error processing file {table_path}: {str(e)}")
                continue
    
    print(f"Found {len(all_columns)} columns across all databases")
    
    # 生成训练数据
    data = []
    sample_id = 0
    
    for column in all_columns:
        for instruction in stage1_instructions:
            data.append({
                "id": f"stage1_{sample_id}",
                "instruction": instruction,
                "input": f"{column['db_id']}.{column['table']}.{column['column']}",
                "output": column["description"],
                "db_id": column["db_id"]
            })
            sample_id += 1
    
    print(f"Generated {len(data)} samples for Stage 1")
    
    # 分割训练验证集
    # 按token分组数据
    token_groups = defaultdict(list)
    for item in data:
        token = item['input']  # 提取token
        token_groups[token].append(item)
    
    # 分割数据确保每个token都有训练样本
    train_data = []
    val_data = []
    
    for token, items in token_groups.items():
        # 确保每个token至少有1个训练样本
        min_train = max(1, int(len(items) * 0.8))
        train_items = items[:min_train]
        val_items = items[min_train:]
        
        train_data.extend(train_items)
        val_data.extend(val_items)

    # 随机打乱数据
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    # 保存数据
    with open(config.stage1_output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(config.stage1_train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(config.stage1_val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Stage 1 data saved: {len(train_data)} train, {len(val_data)} val")

def load_schema_tokens(schema_dir):
    """加载模式token"""
    schema_tokens = {}
    
    for file_name in os.listdir(schema_dir):
        if not file_name.endswith(".txt"):
            continue
            
        db_id = os.path.splitext(file_name)[0]
        file_path = os.path.join(schema_dir, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f if line.strip()]
            schema_tokens[db_id] = tokens
    
    return schema_tokens

def generate_stage2_data():
    """生成第二阶段数据：模式链接"""
    print("Generating Stage 2 data...")
    
    # 加载模式token
    schema_tokens = load_schema_tokens(config.stage2_schema_dir)
    print(f"Loaded schemas for {len(schema_tokens)} databases")
    
    # 加载问题数据
    with open(config.stage2_questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions")
    
    # 生成训练数据
    data = []
    sample_id = 0
    
    for question in questions:
        db_id = question["db_id"]
        if db_id not in schema_tokens:
            print(f"Warning: schema tokens not found for database {db_id}")
            continue
            
        # 获取完整schema
        full_schema = "\n".join(schema_tokens[db_id])
        
        # for instruction in stage2_instructions:
        #     # 格式化输出：[database.table1.column1, database.table2.column2, ...]
        #     output_tokens = question.get("gold_schema", [])
        #     # 添加数据库前缀
        #     output_tokens = [f"{db_id}.{token}" for token in output_tokens if token]
        #     output_str = ", ".join(output_tokens)
        #     if not output_str:
        #         output_str = "[]"
        #     else:
        #         output_str = f"[{output_str}]"
            
        #     data.append({
        #         "id": f"stage2_{sample_id}",
        #         "instruction": instruction,
        #         "input": f"Database schema:\n{full_schema}\n\nUser query: {question['question']}",
        #         "output": output_str,
        #         "db_id": db_id
        #     })
        #     sample_id += 1

        instruction = random.choice(stage2_instructions)
        # 格式化输出：[database.table1.column1, database.table2.column2, ...]
        output_tokens = question.get("gold_schema", [])
        # 添加数据库前缀
        output_tokens = [f"{db_id}.{token}" for token in output_tokens if token]
        output_str = ", ".join(output_tokens)
        if not output_str:
            output_str = "[]"
        else:
            output_str = f"[{output_str}]"
        
        data.append({
            "id": f"stage2_{sample_id}",
            "instruction": instruction,
            "input": f"Database schema:\n{full_schema}\n\nUser query: {question['question']}",
            "output": output_str,
            "db_id": db_id
        })
        sample_id += 1
    
    print(f"Generated {len(data)} samples for Stage 2")
    
    # 分割训练验证集
    # 按数据库分组数据
    db_groups = defaultdict(list)
    for item in data:
        db_groups[item['db_id']].append(item)
    
    # 分割数据确保每个数据库都有训练样本
    train_data = []
    val_data = []
    
    for db_id, items in db_groups.items():
        # 确保每个数据库至少有1个训练样本
        min_train = max(1, int(len(items) * 0.8))
        train_items = items[:min_train]
        val_items = items[min_train:]
        
        train_data.extend(train_items)
        val_data.extend(val_items)

    # 随机打乱数据
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    # 保存数据
    with open(config.stage2_output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(config.stage2_train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(config.stage2_val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Stage 2 data saved: {len(train_data)} train, {len(val_data)} val")

def generate_schema_token_files():
    """为第二阶段生成模式token文件"""
    print("Generating schema token files for Stage 2...")
    
    # 创建输出目录
    os.makedirs(config.stage2_schema_dir, exist_ok=True)
    
    # 收集所有数据库的表和列
    db_structures = defaultdict(lambda: defaultdict(list))
    
    # 遍历数据库目录
    for database_name in os.listdir(config.stage1_database_dir):
        database_path = os.path.join(config.stage1_database_dir, database_name)
        
        # 检查是否是目录
        if not os.path.isdir(database_path):
            continue
            
        # 查找database_description目录
        desc_dir = os.path.join(database_path, "database_description")
        if not os.path.exists(desc_dir):
            print(f"Warning: database_description directory not found in {database_path}")
            continue
            
        # 遍历所有CSV文件
        for table_file in os.listdir(desc_dir):
            if not table_file.endswith(".csv"):
                continue
                
            table_name = os.path.splitext(table_file)[0]
            table_path = os.path.join(desc_dir, table_file)

            try:
                # 检测文件编码
                with open(table_path, 'rb') as f:
                    raw_data = f.read(4096)  # 读取部分内容用于检测编码
                    result = chardet.detect(raw_data)
                    encoding = result['encoding'] or 'utf-8'
                
                # 使用检测到的编码打开文件
                with open(table_path, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # 获取列名
                        column_name = row.get('original_column_name') or row.get('column_name')
                        if not column_name:
                            continue
                        
                        # 添加到数据库结构
                        db_structures[database_name][table_name].append(column_name)
            
            except Exception as e:
                print(f"Error processing file {table_path}: {str(e)}")
                continue
    
    # 生成模式token文件
    for db_id, tables in db_structures.items():
        output_path = os.path.join(config.stage2_schema_dir, f"{db_id}.txt")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for table_name, columns in tables.items():
                for column_name in columns:
                    token = f"{db_id}.{table_name}.{column_name}"
                    f.write(f"{token}\n")
    
    print(f"Generated schema token files for {len(db_structures)} databases")

if __name__ == "__main__":
    # 首先为第二阶段生成模式token文件
    # generate_schema_token_files()
    
    # 生成第一阶段数据
    # generate_stage1_data()
    
    # 生成第二阶段数据
    generate_stage2_data()
    
    print("Data generation completed successfully!")