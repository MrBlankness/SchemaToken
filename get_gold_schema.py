import json
from openai import OpenAI
import re
import ast
import time
import os
import math
import concurrent.futures
from tqdm import tqdm

api_key = '8cefb70606f3472d8731bd65661ce409'
base_url = 'http://8285.model.mingxingtech.com:10032/v1'

def extract_columns_from_sql(sql):
    """
    使用OpenAI API从SQL查询中提取涉及的所有列
    返回格式: [table1.column1, table2.column2, ...] 作为Python列表
    """
    prompt = f"""
Please analyze the following SQL query and list all columns involved (including table names).
Require strict Python list format output: [original_table_name.column_name, original_table_name.column_name, ...]

Important requirements:
1. Return only the list format result, do not add any explanation
2. Ensure each column includes the original table name prefix (do not use temporary aliases)
3. Include all columns used in all clauses (SELECT, WHERE, ORDER BY, etc.)
4. Wrap each element in double quotes
5. Ignore any temporary table aliases in SQL (such as T1, T2, etc.), use the original table names from the database

SQL query:
{sql}
    """
    
    try:
        for _ in range(10):
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model="deepseek.r1:70b",
                messages=[
                    {"role": "system", "content": "You are an SQL parsing assistant responsible for extracting column information from SQL queries. Please ensure the output format is a Python list, wrap each element in double quotes, and always use the original table names instead of temporary aliases."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            # 提取模型回复
            result = response.choices[0].message.content.strip()
            
            # 使用正则表达式提取列表部分
            match = re.search(r'\[.*\]', result, re.DOTALL)
            if match:
                list_str = match.group(0)
                try:
                    # 安全地将字符串转换为Python列表
                    return ast.literal_eval(list_str)
                except (SyntaxError, ValueError):
                    # 如果转换失败，尝试修复常见的格式问题
                    return fix_list_format(list_str)
        
        # 如果多次尝试都失败，返回空列表
        return []
        
    except Exception as e:
        print(f"API调用出错: {e}")
        return []

def fix_list_format(list_str):
    """
    修复常见的列表格式问题，确保可以转换为Python列表
    """
    try:
        # 尝试直接转换
        return ast.literal_eval(list_str)
    except:
        # 替换单引号为双引号
        fixed_str = list_str.replace("'", '"')
        # 移除多余的空白字符
        fixed_str = re.sub(r'\s+', ' ', fixed_str)
        try:
            return ast.literal_eval(fixed_str)
        except:
            # 如果仍然失败，返回空列表
            return []

def process_data_chunk(data_chunk, chunk_index):
    """
    处理数据块，返回处理后的数据和进度信息
    """
    processed_chunk = []

    # 使用tqdm显示进度
    pbar = tqdm(total=len(data_chunk), desc=f"处理块 {chunk_index + 1}/{len(data_chunk)}")

    for i, item in enumerate(data_chunk):
        sql = item["SQL"]
        # print(f"处理块 {chunk_index} 的 {i+1}/{len(data_chunk)}: {sql[:50]}...")
        
        # 获取列信息
        gold_schema = extract_columns_from_sql(sql)
        item["gold_schema"] = gold_schema
        processed_chunk.append(item)
        pbar.update(1)
    pbar.close()
        
    return processed_chunk

def split_into_chunks(data, num_chunks):
    """将数据分成指定数量的块"""
    chunk_size = math.ceil(len(data) / num_chunks)
    return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

def merge_chunks(output_dir, num_chunks, final_output):
    """合并所有临时块文件"""
    merged_data = []
    
    # 按顺序读取所有块文件
    for i in range(num_chunks):
        chunk_file = os.path.join(output_dir, f"chunk_{i}.json")
        if os.path.exists(chunk_file):
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
                merged_data.extend(chunk_data)
    
    # 保存合并后的数据
    with open(final_output, 'w') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"合并完成! 结果已保存到 {final_output}")

def process_json_file_parallel(input_file, final_output, num_chunks=4):
    """并行处理JSON文件"""
    # 创建临时目录
    temp_dir = os.path.join(os.path.dirname(final_output), "temp_chunks_sub")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 读取原始数据
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 将数据分成块
    chunks = split_into_chunks(data, num_chunks)
    print(f"已将数据分成 {len(chunks)} 个块进行并行处理")
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
        # 提交处理任务
        future_to_chunk = {
            executor.submit(process_data_chunk, chunk, idx): idx
            for idx, chunk in enumerate(chunks)
        }
        
        # 使用tqdm显示进度
        completed = 0
        total_chunks = len(chunks)
        pbar = tqdm(total=total_chunks, desc="处理进度")
        
        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                processed_chunk = future.result()
                # 保存处理后的块
                chunk_file = os.path.join(temp_dir, f"chunk_{chunk_index}.json")
                with open(chunk_file, 'w') as f:
                    json.dump(processed_chunk, f, indent=2, ensure_ascii=False)
                
                completed += 1
                pbar.update(1)
                print(f"块 {chunk_index} 处理完成，保存到 {chunk_file}")
                
            except Exception as e:
                print(f"处理块 {chunk_index} 时出错: {e}")
    
    pbar.close()
    
    # 合并所有块
    merge_chunks(temp_dir, num_chunks, final_output)
    
    # 清理临时文件（可选）
    # for i in range(num_chunks):
    #     chunk_file = os.path.join(temp_dir, f"chunk_{i}.json")
    #     if os.path.exists(chunk_file):
    #         os.remove(chunk_file)
    # os.rmdir(temp_dir)

def reprocess_empty_gold_schema(input_file, output_file, max_workers=8):
    """
    处理JSON文件中gold_schema为空的数据
    """
    # 读取原始数据
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 找出所有gold_schema为空的数据
    empty_indices = [i for i, item in enumerate(data) 
                    if not item.get("gold_schema") 
                    or (isinstance(item.get("gold_schema"), list) and len(item["gold_schema"]) == 0)]
    
    print(empty_indices)
    print(f"找到 {len(empty_indices)} 条gold_schema为空的数据需要重新处理")
    
    if not empty_indices:
        print("没有需要重新处理的数据")
        return
    
    # 使用进度条
    pbar = tqdm(total=len(empty_indices), desc="重新处理空gold_schema")
    
    # 并行处理
    def process_item(idx):
        item = data[idx]
        sql = item["SQL"]
        # 获取列信息
        gold_schema = extract_columns_from_sql(sql)
        # 更新数据
        data[idx]["gold_schema"] = gold_schema
        pbar.update(1)
        return idx
    
    # 创建线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有需要处理的任务
        futures = [executor.submit(process_item, idx) for idx in empty_indices]
        
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"处理出错: {e}")
    
    pbar.close()
    
    # 保存更新后的数据
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"处理完成! 结果已保存到 {output_file}")

if __name__ == "__main__":
    # 配置输入输出文件路径
    # input_json = "/data/llmbase/ljw/bird/train/train.json"
    # output_json = "/data/llmbase/ljw/tokenization/data/stage-2/raw_data.json"
    input_json = "/data/llmbase/ljw/tokenization/data/stage-2/raw_data.json"
    output_json = "/data/llmbase/ljw/tokenization/data/stage-2/raw_data.json"

    # 设置并行处理的块数（根据你的机器和API限制调整）
    num_chunks = 16  # 例如8个并行任务
    
    # process_json_file_parallel(input_json, output_json, num_chunks)

    reprocess_empty_gold_schema(input_json, output_json, max_workers=32)