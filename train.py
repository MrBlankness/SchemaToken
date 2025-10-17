import logging
import os
import json
import torch
import swanlab
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SwanLab配置
os.environ["SWANLAB_PROJECT"] = "qwen3-sft-schema-linking"
os.environ["SWANLAB_LOG_LEVEL"] = "info"

# 全局配置
class Config:
    def __init__(self):
        self.stage = 2  # 阶段标识，1表示微调输入嵌入，2表示微调输出嵌入

        # 模型配置
        self.model_name = "Qwen/Qwen3-8B" # 原始模型名称
        # self.model_path = "/data/llmbase/ljw/tokenization/models/qwen3-8b" # 拓展词表后模型路径
        self.model_path = "/data/llmbase/ljw/tokenization/models/tuned-qwen3-8b/stage-1"  # For stage-2，微调后的模型路径
        self.output_dir = f"/data/llmbase/ljw/tokenization/models/tuned-qwen3-8b/stage-{self.stage}"
        self.cache_dir = "/data/llmbase/ljw/hf_cache"
        
        # 训练配置
        self.epochs = 3
        self.train_batch_size = 2
        self.val_batch_size = 4
        self.gradient_accumulation_steps = 2
        self.learning_rate = 1e-4
        self.freeze_strategy = "2-stage"  # 冻结策略

        # 数据配置
        self.train_dataset_path = f"/data/llmbase/ljw/tokenization/data/stage-{self.stage}/train.jsonl"
        self.val_dataset_path = f"/data/llmbase/ljw/tokenization/data/stage-{self.stage}/val.jsonl"
        # self.max_length = 1024  # stage-1 最大上下文长度
        self.max_length = 512  # stage-2 最大上下文长度
        
        # 禁用思考模式（因为我们的数据没有思考内容）
        self.enable_thinking = False

class CustomEmbeddingFreezingCallback(TrainerCallback):
    def __init__(self, original_vocab_size, is_tied_weights, stage=1):
        self.original_vocab_size = original_vocab_size
        self.is_tied_weights = is_tied_weights
        self.stage = stage  # 阶段标识
        print(f"Initializing CustomEmbeddingFreezingCallback for stage {self.stage}...")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        
        # Freeze original part of input embeddings
        if self.stage == 1 and model.get_input_embeddings().weight.grad is not None:
            model.get_input_embeddings().weight.grad.data[:self.original_vocab_size] = 0
        
        # If weights are not tied, also freeze original part of output embeddings
        elif self.stage == 2 and not self.is_tied_weights and model.get_output_embeddings().weight.grad is not None:
            model.get_output_embeddings().weight.grad.data[:self.original_vocab_size] = 0

config = Config()

def process_function(example, tokenizer, max_length=config.max_length):
    """
    将数据集进行预处理
    """ 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n",
        add_special_tokens=False,
        # enable_thinking=config.enable_thinking  # 根据配置决定是否启用思考模式
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
    if len(input_ids) > max_length:  # 做一个截断
        # print(f"Warning: Input length {len(input_ids)} exceeds max_length {max_length}, truncating.")
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# def apply_freeze_strategy(model):
#     """
#     应用参数冻结策略（2x2 LS）
#     """
#     logger.info("Applying 2x2 LS freezing strategy...")
    
#     # 冻结所有参数
#     for param in model.parameters():
#         param.requires_grad = False
    
#     # 解冻输入输出嵌入层
#     for param in model.get_input_embeddings().parameters():
#         param.requires_grad = True
    
#     for param in model.get_output_embeddings().parameters():
#         param.requires_grad = True
    
#     # 解冻顶部和底部的两层
#     num_layers = len(model.model.layers)
#     layers_to_train = [0, 1, num_layers-2, num_layers-1]
    
#     for idx in layers_to_train:
#         for param in model.model.layers[idx].parameters():
#             param.requires_grad = True
    
#     # 打印可训练参数统计
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     logger.info(f"Trainable params: {trainable_params} || All params: {total_params} || Trainable %: {100 * trainable_params / total_params:.2f}")
    
#     return model

def apply_freeze_strategy(model, stage=1):
    for param in model.parameters():
        param.requires_grad = False
    
    if stage == 1:
        print("Unfreezing input embeddings for stage 1...")
        for param in model.get_input_embeddings().parameters():
            param.requires_grad = True
    elif stage == 2 and not model.config.tie_word_embeddings:
        print("Unfreezing output embeddings for stage 2...")
        for param in model.get_output_embeddings().parameters():
            print(f"Unfreezing output embedding parameter: {param}")
            param.requires_grad = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  TRAINABLE: {name}, shape: {param.shape}, numel: {param.numel()}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable_params} || All params: {total_params} || Trainable %: {100 * trainable_params / total_params:.2f}")
    
    return model

def main():
    # 初始化SwanLab记录
    # swanlab.init()

    # 更新SwanLab配置
    swanlab.config.update({
        "model": config.model_name,
        "max_length": config.max_length,
        "epochs": config.epochs,
        "train_batch_size": config.train_batch_size,
        "val_batch_size": config.val_batch_size,
        "learning_rate": config.learning_rate,
        "freeze_strategy": config.freeze_strategy,
        "enable_thinking": config.enable_thinking
    })

    # --- 加载tokenizer ---
    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        local_files_only = True,
    )
    
    # 设置特殊token
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    
    # --- 加载数据集 ---
    # raw_train_data = load_jsonl_dataset(config.dataset_path)
    # raw_val_data = load_jsonl_dataset(config.val_dataset_path)
    
    # 转换为HuggingFace Dataset
    train_ds = Dataset.from_json(config.train_dataset_path, split="train")
    val_ds = Dataset.from_json(config.val_dataset_path, split="validation")
    
    # 预处理数据集
    train_dataset = train_ds.map(
        lambda example: process_function(example, tokenizer),
        remove_columns=train_ds.column_names
    )
    
    val_dataset = val_ds.map(
        lambda example: process_function(example, tokenizer),
        remove_columns=val_ds.column_names
    )
    
    # 打印样本示例
    logger.info("Sample input_ids: " + str(train_dataset[0]["input_ids"][:20]))
    logger.info("Sample labels: " + str(train_dataset[0]["labels"][:20]))
    
    # --- 加载模型 ---
    logger.info(f"Loading model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        # device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    
    # --- 应用冻结策略 ---
    model = apply_freeze_strategy(model, stage=config.stage)
    
    # --- 配置训练参数 ---
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs + 3,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.val_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=400,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        seed=42,
        report_to="swanlab",  # 报告到SwanLab
        metric_for_best_model="eval_loss",
        run_name=f"qwen3-8b-schema-linking-stage-{config.stage}-new-6-epoch"  # SwanLab运行名称
    )
    
    # --- 数据整理器 ---
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8
    )
    
    # --- 初始化Trainer ---
    original_tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        local_files_only=True,
        cache_dir=config.cache_dir
    )
    original_vocab_size = len(original_tokenizer)
    is_tied_weights = model.config.tie_word_embeddings
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[CustomEmbeddingFreezingCallback(original_vocab_size, is_tied_weights, stage=config.stage)],
    )
    
    # --- 开始训练 ---
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=False)
    
    # 保存训练指标
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # --- 保存模型 ---
    logger.info(f"Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    
    logger.info("Training completed successfully!")
    # swanlab.finish()  # 结束SwanLab记录

if __name__ == "__main__":
    main()