"""Schema-linking evaluation for SchemaToken on BIRD-dev.

Loads the stage-2 fine-tuned model, runs the schema-linking task on a
configurable subset of BIRD dev questions, and reports the four metrics
defined in §C.3.1 of the paper:

  • SRR – Strict Recall Rate: fraction of questions for which the
          predicted set fully covers the gold set.
  • NSR – Non-Strict Recall: total |pred ∩ gold| / total |gold|.
  • NSP – Non-Strict Precision: total |pred ∩ gold| / total |pred|.
  • NSF – Non-Strict F1.

The evaluation prompt deliberately avoids passing the column descriptions
to the model — that is the central efficiency win promised by
SchemaToken. Instead the model is shown the active schema-token
*identifiers* for the question's database and asked to emit the relevant
ones.
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from schema_tokens import extend_tokenizer_and_model
from schema_utils import jaccard, load_config, parse_schema_tokens


def parse_args():
    p = argparse.ArgumentParser(description="SchemaToken evaluation")
    p.add_argument("--config", default=None, type=str)
    p.add_argument(
        "--stage", choices=["base", "stage1", "stage2"], default="stage2",
        help="Which checkpoint to evaluate. 'base' = unfine-tuned schema model.",
    )
    return p.parse_args()


def render_eval_prompt(tokenizer, question: str, schema_tokens: List[str], evidence: str = "") -> str:
    cand = ", ".join(schema_tokens)
    user = (
        "Given a natural language question and a candidate set of schema tokens "
        "(each token represents one database column), output ONLY the schema tokens "
        "that are required to answer the question. Comma-separated list, no extra text.\n"
        f"Question: {question}\n"
        + (f"Hint: {evidence}\n" if evidence else "")
        + f"Candidate schema tokens: {cand}"
    )
    msgs = [{"role": "user", "content": user}]
    for kwargs in (
        {"enable_thinking": False, "add_generation_prompt": True},
        {"add_generation_prompt": True},
    ):
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, **kwargs)
        except Exception:
            continue
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def predict_schema(model, tokenizer, prompt: str, max_new_tokens: int) -> List[str]:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    completion = out[0, enc.input_ids.shape[1]:]
    text = tokenizer.decode(completion, skip_special_tokens=False)
    return text


def main():
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    sk = cfg["schema_token"]
    eval_cfg = cfg["eval"]

    os.makedirs(paths["eval_dir"], exist_ok=True)
    meta = json.load(open(paths["schema_meta"], "r", encoding="utf-8"))
    schema_records = meta["schema"]
    examples = meta["examples"]

    by_db: Dict[str, List[str]] = defaultdict(list)
    for r in schema_records:
        by_db[r["db_id"]].append(r["schema_token"])

    # ---- Load tokenizer/model ----
    if args.stage == "base":
        tok_path = paths["base_model"]
    elif args.stage == "stage1":
        tok_path = paths["stage1_dir"]
    else:
        tok_path = paths["stage2_dir"] if os.path.exists(paths["stage2_dir"]) else paths["stage1_dir"]

    print(f"[eval] loading tokenizer from {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[eval] loading base model from {paths['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        paths["base_model"], dtype=torch.bfloat16, device_map="cuda"
    )
    state = extend_tokenizer_and_model(
        tokenizer, model, schema_records,
        L_T=sk["L_T"], L_C=sk["L_C"], N=sk["N"],
        # Use a parameter-style orig_weight so loaded state dicts (which may
        # contain it) can copy in cleanly regardless of stage.
        freeze_lm_head_orig=False,
    )
    tokenizer = state["tokenizer"]
    model = state["model"]
    codebook = state["codebook"]

    # Load weights from the chosen stage
    if args.stage in ("stage1", "stage2"):
        cb_path = os.path.join(paths["stage1_dir"], "codebook.pt") if args.stage == "stage1" else os.path.join(paths["stage2_dir"], "codebook.pt")
        if os.path.exists(cb_path):
            codebook.load_state_dict(torch.load(cb_path, map_location="cpu"))
            codebook.to(device=model.device)
        lm_path = os.path.join(paths["stage1_dir"], "lm_head.pt") if args.stage == "stage1" else os.path.join(paths["stage2_dir"], "lm_head.pt")
        if os.path.exists(lm_path):
            sd = torch.load(lm_path, map_location="cpu")
            with torch.no_grad():
                if "schema_weight" in sd:
                    model.lm_head.schema_weight.copy_(
                        sd["schema_weight"].to(model.device).to(model.lm_head.schema_weight.dtype)
                    )
                if "orig_weight" in sd:
                    model.lm_head.orig_weight.copy_(
                        sd["orig_weight"].to(model.device).to(model.lm_head.orig_weight.dtype)
                    )

    if args.stage == "stage2":
        # Load LoRA on top of the backbone (the base PEFT directory contains
        # the adapter only; passes through unchanged for any modules that
        # weren't LoRA-targeted).
        adapter_path = paths["stage2_dir"]
        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            print(f"[eval] loading LoRA adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        else:
            print(f"[eval][warn] no LoRA adapter at {adapter_path}; using base")

    model.eval()

    # ---- Evaluate ----
    cap = int(eval_cfg.get("max_questions_per_db", 4))
    counts: Dict[str, int] = defaultdict(int)
    results = []
    for ex in examples:
        if cap > 0 and counts[ex["db_id"]] >= cap:
            continue
        counts[ex["db_id"]] += 1
        candidates = by_db[ex["db_id"]]
        prompt = render_eval_prompt(
            tokenizer, ex["question"], candidates, ex.get("evidence", "")
        )
        out_text = predict_schema(model, tokenizer, prompt, eval_cfg["max_new_tokens"])
        pred_set = set(parse_schema_tokens(out_text)) & set(candidates)
        gold_set = set(ex["gold_schema_tokens"])
        results.append({
            "question_id": ex["question_id"],
            "db_id": ex["db_id"],
            "question": ex["question"],
            "gold": sorted(gold_set),
            "pred_text": out_text,
            "pred": sorted(pred_set),
            "jaccard": jaccard(pred_set, gold_set),
        })

    # ---- Metrics ----
    total_inter = 0
    total_pred = 0
    total_gold = 0
    n_strict_hits = 0
    for r in results:
        gs = set(r["gold"])
        ps = set(r["pred"])
        total_inter += len(gs & ps)
        total_pred += len(ps)
        total_gold += len(gs)
        if gs and gs.issubset(ps):
            n_strict_hits += 1
    n = len(results)
    NSR = total_inter / total_gold if total_gold else 0.0
    NSP = total_inter / total_pred if total_pred else 0.0
    NSF = (2 * NSR * NSP / (NSR + NSP)) if (NSR + NSP) else 0.0
    SRR = n_strict_hits / n if n else 0.0

    print(f"\n[eval] stage={args.stage}, n={n}")
    print(f"  SRR={SRR*100:.2f}  NSR={NSR*100:.2f}  NSP={NSP*100:.2f}  NSF={NSF*100:.2f}")
    out_path = os.path.join(paths["eval_dir"], f"results_{args.stage}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "stage": args.stage,
            "n": n,
            "SRR": SRR, "NSR": NSR, "NSP": NSP, "NSF": NSF,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
