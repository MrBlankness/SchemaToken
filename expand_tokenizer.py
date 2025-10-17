import os
import argparse
from transformers import AutoTokenizer

def load_schema_tokens(schema_dir):
    """加载模式token"""
    schema_tokens = []
    
    for file_name in os.listdir(schema_dir):
        if not file_name.endswith(".txt"):
            continue
            
        file_path = os.path.join(schema_dir, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f if line.strip()]
            schema_tokens.extend(tokens)
    
    return schema_tokens

def main(args):
    # load the source tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.hub_cache_dir,
        local_files_only = True
    )
    vocab = tokenizer.get_vocab()

    # 获取 schema tokens 并过滤出当前词表不存在的 token 添加到词表中
    schema_dir = "/data/llmbase/ljw/tokenization/data/stage-2/schema_tokens"
    schema_tokens = load_schema_tokens(schema_dir)
    new_tokens = [token for token in schema_tokens if token not in vocab]
    print(len(new_tokens), "new schema tokens found.")
    print("Total schema tokens:", len(schema_tokens))
    if new_tokens:
        print(f"Adding {len(new_tokens)} new schema tokens to the tokenizer.")
        tokenizer.add_tokens(new_tokens)
    else:
        print("No new schema tokens to add.")
    
    # save
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", 
        type=str,
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="/data/llmbase/ljw/tokenization/models/qwen3-8b",
    )
    parser.add_argument(
        "--hub_cache_dir", 
        type=str,
        default="/data/llmbase/ljw/hf_cache"
    )
    args = parser.parse_args()
    main(args)
    