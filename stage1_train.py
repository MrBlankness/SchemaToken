"""Stage 1: Semantic Alignment SFT.

Following §3.3 of the paper:

  • All LLM parameters (including original token embeddings) are frozen.
  • Only the codebook embeddings (BT, BC) and the new rows of the LM_head
    are updated.
  • The training objective is L_align = L_Sem + α · L_Div, where L_Sem is
    standard SFT cross-entropy on the description target and L_Div is the
    InfoNCE-style diversity regulariser implemented in
    `schema_tokens.SchemaCodebook.diversity_loss`.

The training data are constructed on the fly from the schema metadata
file produced by `data_pipeline.py`.  For each schema token we pair a
random prompt template ("Generate the description for the given database
column. ...") with the column's description text – the model is asked to
predict the *description* given the schema token in its input.
"""
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from schema_tokens import (
    extend_tokenizer_and_model,
    trainable_params_for_stage1,
)
from schema_utils import load_config


# Prompt templates from Appendix B.3 of the paper
PROMPTS_DESC = [
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


def render_target(rec: Dict) -> str:
    """Per the paper: 'This column belongs to the {table_name} table and represents {description}.'"""
    return f"This column belongs to the {rec['table']} table and represents {rec['description']}."


def parse_args():
    p = argparse.ArgumentParser(description="SchemaToken Stage 1 - Semantic Alignment")
    p.add_argument("--config", default=None, type=str)
    return p.parse_args()


class SemanticAlignmentDataset(Dataset):
    def __init__(self, schema_records: List[Dict], num_samples: int, seed: int = 0):
        self.records = schema_records
        rng = random.Random(seed)
        self.items: List[Dict[str, Any]] = []
        for _ in range(num_samples):
            rec = rng.choice(schema_records)
            prompt = rng.choice(PROMPTS_DESC)
            self.items.append({
                "schema_token": rec["schema_token"],
                "instruction": prompt,
                "response": render_target(rec),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


@dataclass
class SemanticAlignmentCollator:
    tokenizer: Any
    max_length: int = 384
    ignore_index: int = -100

    def __call__(self, batch):
        prompts = []
        full_texts = []
        for item in batch:
            messages = [
                {"role": "user", "content": f"Schema token: {item['schema_token']}\n{item['instruction']}"},
            ]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            except Exception:
                prompt = (
                    f"<|im_start|>user\nSchema token: {item['schema_token']}\n{item['instruction']}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
            full = prompt + item["response"] + (self.tokenizer.eos_token or "")
            prompts.append(prompt)
            full_texts.append(full)

        max_len = self.max_length
        full_enc = self.tokenizer(
            full_texts, padding=True, truncation=True, max_length=max_len,
            return_tensors="pt", add_special_tokens=False,
        )
        prompt_lens = []
        for p in prompts:
            ids = self.tokenizer(p, add_special_tokens=False).input_ids
            prompt_lens.append(min(len(ids), max_len))

        labels = full_enc["input_ids"].clone()
        # Mask the prompt portion and any padding from the loss.
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = self.ignore_index
        labels[full_enc["attention_mask"] == 0] = self.ignore_index

        return {
            "input_ids": full_enc["input_ids"],
            "attention_mask": full_enc["attention_mask"],
            "labels": labels,
        }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    s1 = cfg["stage1"]
    sk = cfg["schema_token"]

    os.makedirs(paths["stage1_dir"], exist_ok=True)
    if not os.path.exists(paths["schema_meta"]):
        raise RuntimeError("Run data_pipeline.py first.")

    meta = json.load(open(paths["schema_meta"], "r", encoding="utf-8"))
    schema_records = meta["schema"]

    # ---- Tokenizer + model + schema-token surgery ----
    print(f"[stage1] loading {paths['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(paths["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if s1.get("bf16", True) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        paths["base_model"], dtype=dtype, device_map="cuda"
    )

    state = extend_tokenizer_and_model(
        tokenizer, model, schema_records,
        L_T=sk["L_T"], L_C=sk["L_C"], N=sk["N"],
        freeze_lm_head_orig=True,
    )
    tokenizer = state["tokenizer"]
    model = state["model"]
    codebook = state["codebook"]

    trainable_params_for_stage1(model, codebook)
    n_train = sum(p.numel() for p in list(model.parameters()) + list(codebook.parameters()) if p.requires_grad)
    print(f"[stage1] trainable params: {n_train:,}")

    # ---- Dataset ----
    dataset = SemanticAlignmentDataset(
        schema_records, num_samples=s1["num_train_samples"], seed=0
    )
    collator = SemanticAlignmentCollator(
        tokenizer=tokenizer, max_length=s1["max_length"]
    )
    loader = DataLoader(
        dataset,
        batch_size=s1["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=collator,
    )

    # ---- Optimiser ----
    seen = set()
    params = []
    for p in list(model.parameters()) + list(codebook.parameters()):
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    optim = torch.optim.AdamW(params, lr=s1["learning_rate"])

    # ---- Train loop ----
    model.train()
    codebook.train()
    grad_accum = s1["gradient_accumulation_steps"]
    max_steps = s1.get("max_steps", -1)
    log_steps = s1.get("logging_steps", 5)
    alpha = sk["alpha"]
    K = sk["K"]
    step = 0
    micro_step = 0
    optim.zero_grad()
    for epoch in range(s1["num_train_epochs"]):
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            out = model(**batch)
            l_sem = out.loss
            l_div = codebook.diversity_loss(K=K)
            loss = (l_sem + alpha * l_div) / grad_accum
            loss.backward()
            micro_step += 1
            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                optim.zero_grad()
                step += 1
                if step % log_steps == 0 or step == 1:
                    print(
                        f"[stage1] step {step}: l_sem={l_sem.item():.4f} "
                        f"l_div={l_div.item():.4f} loss={l_sem.item() + alpha * l_div.item():.4f}"
                    )
                if max_steps > 0 and step >= max_steps:
                    break
        if max_steps > 0 and step >= max_steps:
            break

    # ---- Save artefacts ----
    out_dir = paths["stage1_dir"]
    os.makedirs(out_dir, exist_ok=True)
    torch.save(codebook.state_dict(), os.path.join(out_dir, "codebook.pt"))
    torch.save(model.lm_head.state_dict(), os.path.join(out_dir, "lm_head.pt"))
    tokenizer.save_pretrained(out_dir)
    model.config.save_pretrained(out_dir)
    print(f"[stage1] saved codebook + lm_head + tokenizer to {out_dir}")


if __name__ == "__main__":
    main()
