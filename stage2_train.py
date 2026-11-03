"""Stage 2: Pseudo-task Alignment via GRPO with a Jaccard reward.

Design (faithful to §3.4 of the paper but compute-tiny for debugging):

  • Construct random pseudo-tasks by sampling m∈[ϵ̌, ϵ̂] columns from the
    schema and asking the model to output the matching schema tokens given
    only their natural-language descriptions.
  • Sample G rollouts from the current policy (`do_sample=True`).
  • Reward each rollout with Jaccard similarity between the predicted set
    of schema tokens and the gold set.
  • Compute group-relative advantage A = (r − mean(r)) / (std(r) + eps).
  • Optimise -E[A · log π_θ(a|s)] (i.e. group-relative REINFORCE, which is
    GRPO without the importance-ratio clip and KL penalty; we add a small
    KL term against the *initial* policy if `kl_beta > 0`).

Only LoRA adapters on the LLM backbone and the trainable rows of
`SchemaLMHead` receive gradients; the codebook stays frozen with the
weights it learned in stage 1.
"""
import argparse
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from schema_tokens import (
    extend_tokenizer_and_model,
    trainable_params_for_stage2,
)
from schema_utils import jaccard, load_config, parse_schema_tokens


PROMPTS_PSEUDO = [
    "Identify the database columns related to the following descriptions: {desc_list}.",
    "Given these semantic definitions, return the corresponding schema tokens: {desc_list}.",
    "Which database schema tokens match the following meanings? {desc_list}.",
    "Map the provided descriptions to their exact schema tokens: {desc_list}.",
    "Extract the relevant schema tokens that represent these functions: {desc_list}.",
    "Translate the following column descriptions into their corresponding schema tokens: {desc_list}.",
    "Find the database schema tokens that align with the following semantics: {desc_list}.",
    "Please output the schema tokens that correspond to these column descriptions: {desc_list}.",
    "Retrieve the specific schema tokens representing the following database attributes: {desc_list}.",
    "Determine the correct schema tokens for the database fields described here: {desc_list}.",
]


def parse_args():
    p = argparse.ArgumentParser(description="SchemaToken Stage 2 - GRPO")
    p.add_argument("--config", default=None, type=str)
    return p.parse_args()


def build_pseudo_task(rng: random.Random, db_records: Dict[str, list], m_min: int, m_max: int):
    db_id = rng.choice([k for k, v in db_records.items() if len(v) >= m_min])
    candidates = db_records[db_id]
    m = rng.randint(m_min, min(m_max, len(candidates)))
    selected = rng.sample(candidates, m)
    desc_list = "; ".join(f'"{rec["description"]}"' for rec in selected)
    prompt = rng.choice(PROMPTS_PSEUDO).format(desc_list=desc_list) + " Only output the schema tokens."
    gold = {rec["schema_token"] for rec in selected}
    return prompt, gold, db_id


def render_prompt(tokenizer, prompt_text: str) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    # `enable_thinking=False` is recognised by the Qwen3 chat template and
    # is ignored by older templates – we wrap it in try/except.
    for kwargs in (
        {"enable_thinking": False, "add_generation_prompt": True},
        {"add_generation_prompt": True},
    ):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        except Exception:
            continue
    return f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def generate_rollouts(model, tokenizer, prompt_text: str, num_rollouts: int, max_new_tokens: int):
    """Returns input_ids (1, P), generated_ids (G, L), full_ids (G, P+L)."""
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_ids = enc.input_ids
    attn = enc.attention_mask
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        num_return_sequences=num_rollouts,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    p_len = input_ids.shape[1]
    generated = out[:, p_len:]
    return input_ids, generated, out, p_len


def compute_log_probs(model, full_ids: torch.Tensor, prompt_len: int, pad_id: int):
    """Per-sequence sum of log probs of the *generated* tokens.

    full_ids: (G, P+L). We forward through the model and gather log probs
    for tokens at positions [prompt_len .. P+L-1] (predicted from positions
    [prompt_len-1 .. P+L-2]).
    """
    out = model(input_ids=full_ids, attention_mask=(full_ids != pad_id).long())
    logits = out.logits  # (G, P+L, V)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    # Predict full_ids[:, t] from logits[:, t-1, :]
    target = full_ids[:, 1:]                    # (G, P+L-1)
    pred_logp = log_probs[:, :-1, :].gather(2, target.unsqueeze(-1)).squeeze(-1)  # (G, P+L-1)
    # Mask: only completion positions; we predict completions starting at
    # index `prompt_len-1` of pred_logp (which corresponds to full_ids[:, prompt_len]).
    mask = torch.zeros_like(pred_logp)
    mask[:, prompt_len - 1:] = 1.0
    # Also drop pad-predictions
    mask = mask * (target != pad_id).float()
    seq_logp = (pred_logp * mask).sum(dim=-1)
    seq_len = mask.sum(dim=-1).clamp_min(1.0)
    return seq_logp, seq_len


def main():
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    s2 = cfg["stage2"]
    sk = cfg["schema_token"]

    os.makedirs(paths["stage2_dir"], exist_ok=True)
    if not os.path.exists(paths["stage1_dir"]):
        raise RuntimeError("Run stage1_train.py first.")
    meta = json.load(open(paths["schema_meta"], "r", encoding="utf-8"))
    schema_records = meta["schema"]

    # Group records by db so pseudo-tasks stay within one database
    db_records: Dict[str, List[Dict]] = defaultdict(list)
    for r in schema_records:
        db_records[r["db_id"]].append(r)
    print(f"[stage2] dbs in scope: {list(db_records.keys())}")

    # ---- Tokenizer and model ----
    print(f"[stage2] loading {paths['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(paths["stage1_dir"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if s2.get("bf16", True) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        paths["base_model"], dtype=dtype, device_map="cuda"
    )

    state = extend_tokenizer_and_model(
        tokenizer, model, schema_records,
        L_T=sk["L_T"], L_C=sk["L_C"], N=sk["N"],
        # Stage 2: backbone trainable, so we *also* unfreeze original lm_head rows
        freeze_lm_head_orig=False,
    )
    tokenizer = state["tokenizer"]
    model = state["model"]
    codebook = state["codebook"]

    # Load Stage 1 artefacts
    cb_state = torch.load(os.path.join(paths["stage1_dir"], "codebook.pt"), map_location="cpu")
    codebook.load_state_dict(cb_state)
    codebook.to(device=model.device)
    lm_state = torch.load(os.path.join(paths["stage1_dir"], "lm_head.pt"), map_location="cpu")
    # The stage-1 SchemaLMHead used freeze_orig=True so it didn't have an
    # `orig_weight` Parameter, only a buffer. Stage 2 wants `orig_weight`
    # trainable so the names differ — load only the schema_weight.
    if "schema_weight" in lm_state:
        with torch.no_grad():
            model.lm_head.schema_weight.copy_(lm_state["schema_weight"].to(model.device).to(model.lm_head.schema_weight.dtype))
    if "orig_weight" in lm_state:
        with torch.no_grad():
            model.lm_head.orig_weight.copy_(lm_state["orig_weight"].to(model.device).to(model.lm_head.orig_weight.dtype))

    # Apply LoRA to the backbone (Qwen3 attention/MLP projections)
    lora_cfg = LoraConfig(
        r=s2.get("lora_r", 8),
        lora_alpha=s2.get("lora_alpha", 16),
        lora_dropout=s2.get("lora_dropout", 0.0),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable_params_for_stage2(model.base_model, codebook)
    # `get_peft_model` returns a wrapped object; we still want lm_head trainable.
    for p in model.base_model.lm_head.parameters():
        p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[stage2] trainable params: {n_train:,}")

    # ---- Optimiser ----
    seen = set()
    params = []
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    optim = torch.optim.AdamW(params, lr=s2["learning_rate"])

    # ---- Training loop ----
    rng = random.Random(0)
    pad_id = tokenizer.pad_token_id
    log_steps = s2.get("logging_steps", 2)
    G = s2["num_rollouts"]
    max_new_tokens = s2["max_completion_length"]
    kl_beta = s2.get("kl_beta", 0.0)

    model.train()
    for step in range(1, s2["num_train_steps"] + 1):
        # Build a pseudo-task
        prompt_text, gold_set, db_id = build_pseudo_task(
            rng, db_records, s2["m_min"], s2["m_max"]
        )
        rendered = render_prompt(tokenizer, prompt_text)

        # Rollouts (no grad) and rewards
        with torch.no_grad():
            input_ids, gen_ids, full_ids, p_len = generate_rollouts(
                model, tokenizer, rendered, G, max_new_tokens
            )
        rewards = []
        for g in gen_ids:
            text = tokenizer.decode(g, skip_special_tokens=False)
            pred = set(parse_schema_tokens(text))
            rewards.append(jaccard(pred, gold_set))
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=full_ids.device)
        # Group-relative advantage
        adv = rewards_t - rewards_t.mean()
        if rewards_t.std() > 1e-8:
            adv = adv / (rewards_t.std() + 1e-6)

        # Re-run forward with grads to get log-probs
        seq_logp, seq_len = compute_log_probs(model, full_ids, p_len, pad_id)
        # Per-token mean log-prob over completion (clamp by length)
        per_tok_logp = seq_logp / seq_len

        loss = -(adv.detach() * per_tok_logp).mean()

        # Optional simple KL against detached log-probs of the same model
        # (acts as a stay-close regulariser; we don't keep a separate ref
        # model to save memory).
        if kl_beta > 0:
            with torch.no_grad():
                ref_logp, _ = compute_log_probs(model, full_ids, p_len, pad_id)
                ref_logp = ref_logp / seq_len
            kl = (per_tok_logp - ref_logp).mean()
            loss = loss + kl_beta * (kl ** 2)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()

        if step % log_steps == 0 or step == 1:
            sample_text = tokenizer.decode(gen_ids[0], skip_special_tokens=False)[:120]
            print(
                f"[stage2] step {step}/{s2['num_train_steps']}: "
                f"reward_mean={rewards_t.mean().item():.3f} "
                f"reward_max={rewards_t.max().item():.3f} "
                f"adv_std={adv.std().item():.3f} "
                f"loss={loss.item():.4f} | sample: {sample_text!r}"
            )

    # ---- Save artefacts ----
    out_dir = paths["stage2_dir"]
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    torch.save(codebook.state_dict(), os.path.join(out_dir, "codebook.pt"))
    torch.save(model.base_model.lm_head.state_dict(), os.path.join(out_dir, "lm_head.pt"))
    tokenizer.save_pretrained(out_dir)
    print(f"[stage2] saved LoRA + codebook + lm_head + tokenizer to {out_dir}")


if __name__ == "__main__":
    main()
