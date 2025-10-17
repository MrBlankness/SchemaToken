import logging
import os
import json
import torch
import swanlab
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置路径和参数 (直接硬编码，不使用Config类)
MODEL_PATH = "/data/llmbase/ljw/tokenization/models/tuned-qwen3-8b/stage-2/checkpoint-14142"
VAL_DATASET_PATH = "/data/llmbase/ljw/tokenization/data/stage-2/val.jsonl"
MAX_LENGTH = 512
EVAL_BATCH_SIZE = 4 # 评估批次大小，可以适当增大

# SwanLab配置
os.environ["SWANLAB_PROJECT"] = "qwen3-sft-schema-linking-loss-check"
os.environ["SWANLAB_LOG_LEVEL"] = "info"

def process_function(example, tokenizer, max_length=MAX_LENGTH):
    """
    将数据集进行预处理
    """ 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
        # enable_thinking=config.enable_thinking  # 根据配置决定是否启用思考模式
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > max_length:  # 做一个截断
        # print(f"Warning: Input length {len(input_ids)} exceeds max_length {max_length}, truncating.")
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    swanlab.init(project="qwen3-sft-schema-linking-eval_loss-check", run_name="loss-nan-debug")
    swanlab.config.update({
        "model_path": MODEL_PATH,
        "val_dataset_path": VAL_DATASET_PATH,
        "max_length": MAX_LENGTH,
        "eval_batch_size": EVAL_BATCH_SIZE
    })

    logger.info(f"Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16, # 或 torch.float16，取决于你的训练设置和GPU支持
        device_map="auto", # 使用auto，让transformers自动分发模型
    )
    model.eval() # 设置模型为评估模式

    logger.info(f"Loading validation dataset from {VAL_DATASET_PATH}")
    val_ds_raw = Dataset.from_json(VAL_DATASET_PATH, split="validation")
    val_ds = val_ds_raw.map(
        lambda example: process_function(example, tokenizer, MAX_LENGTH),
        remove_columns=val_ds_raw.column_names,
        load_from_cache_file=False
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=MAX_LENGTH,
        pad_to_multiple_of=8
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=data_collator,
        shuffle=False
    )

    total_loss = 0.0
    nan_batches_found = 0
    
    logger.info("Starting validation loop to find NaN loss...")
    with torch.no_grad(): # 禁用梯度计算，节省显存并加速
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Checking for NaN loss")):
            inputs = {k: v.to(model.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}

            outputs = model(**inputs)
            loss = outputs.loss

            if torch.isnan(loss):
                nan_batches_found += 1
                logger.error(f"!!! NaN loss detected at batch index: {batch_idx} !!!")
                logger.error(f"  Batch {batch_idx} Input IDs (first sample decoded):")
                # 打印第一条样本的input_ids
                logger.error(tokenizer.decode(inputs['input_ids'][0].tolist(), skip_special_tokens=False))
                logger.error(f"  Batch {batch_idx} Labels (first sample decoded, -100 will show as pad token):")
                # 将-100替换为pad_token_id以便解码
                labels_for_decode = [
                    x if x != -100 else tokenizer.pad_token_id 
                    for x in inputs['labels'][0].tolist()
                ]
                logger.error(tokenizer.decode(labels_for_decode, skip_special_tokens=False))
                
                # 可选：打印logits的统计信息，如果loss是NaN，可能是logits有问题
                # logger.error(f"  Batch {batch_idx} Logits stats (min, max, mean, std):")
                # logger.error(f"  Min: {outputs.logits.min().item():.4f}, Max: {outputs.logits.max().item():.4f}, Mean: {outputs.logits.mean().item():.4f}, Std: {outputs.logits.std().item():.4f}")
                # 如果发现很大的值（inf）或NaN，可能是模型输出层的问题
                
                # 如果你想在发现第一个NaN时就停止，可以取消下面的注释
                # break 
            else:
                total_loss += loss.item()
        
    avg_loss = total_loss / (len(val_dataloader) - nan_batches_found) if (len(val_dataloader) - nan_batches_found) > 0 else 0
    logger.info(f"Validation loop finished. Total batches: {len(val_dataloader)}, NaN batches: {nan_batches_found}.")
    logger.info(f"Average loss (excluding NaN batches): {avg_loss:.4f}")

    if nan_batches_found > 0:
        swanlab.log({"eval/loss_nan_count": nan_batches_found, "eval/avg_loss": float('nan')})
    else:
        swanlab.log({"eval/avg_loss": avg_loss})
    
    swanlab.finish()

if __name__ == "__main__":
    main()