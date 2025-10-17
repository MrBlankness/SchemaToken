import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen3-8B"
# model_name = "/data/llmbase/ljw/tokenization/models/qwen3-8b"
# model_name = "/data/llmbase/ljw/tokenization/models/tuned-qwen3-8b/stage-1/checkpoint-5364"
# model_name = "/data/llmbase/ljw/tokenization/models/tuned-qwen3-8b/stage-2/checkpoint-14142"
model_name = "/data/llmbase/ljw/tokenization/models/tuned-qwen3-8b/stage-1"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir="/data/llmbase/ljw/hf_cache",  # 设置缓存目录
    local_files_only=True
    )
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir="/data/llmbase/ljw/hf_cache",  # 设置缓存目录
    local_files_only=True
)

# 从 JSONL文件中加载数据

data = []
with open("/data/llmbase/ljw/tokenization/data/stage-1/train.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

instruction = data[17659]["instruction"]
input_value = data[17659]["input"]
gold_output = data[17659]["output"]
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

print("text:", text)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("model_inputs:", model_inputs.input_ids[0])

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

print("output_ids:", output_ids)

content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

print("instruction:", instruction)
print("input:", input_value)
print("gold output:", gold_output)
print("model_output:", content)