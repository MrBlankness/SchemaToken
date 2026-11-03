"""Core SchemaToken modeling code:
  * `SchemaCodebook`         - residual codebook composition + diversity loss.
  * `SchemaInputEmbedding`   - drop-in replacement for the model's
                                 input-embedding module that uses the codebook
                                 for new (schema) token IDs.
  * `SchemaLMHead`           - drop-in replacement for `model.lm_head` that
                                 keeps the original output projection frozen
                                 and adds trainable rows for schema tokens.
  * `extend_tokenizer_and_model` - one-stop function that mutates a tokenizer
                                    and a HuggingFace causal-LM to make them
                                    aware of all schema tokens.

Design follows §3.2 of the paper (RQ-VAE-inspired residual codebooks +
LM_head extension). For simplicity each (table, column) is given a fixed
random sequence of codebook indices at init time; the *codebook embeddings*
themselves are what the optimiser learns during semantic alignment.
"""
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SchemaCodebook(nn.Module):
    """Holds the table/column codebooks and composes per-token embeddings."""

    def __init__(
        self,
        num_schema_tokens: int,
        table_id_per_token: torch.Tensor,   # (num_schema_tokens,) long
        col_id_per_token: torch.Tensor,     # (num_schema_tokens,) long
        num_tables: int,
        num_columns: int,
        L_T: int,
        L_C: int,
        N: int,
        hidden_dim: int,
    ):
        super().__init__()
        assert table_id_per_token.shape == (num_schema_tokens,)
        assert col_id_per_token.shape == (num_schema_tokens,)
        self.num_schema_tokens = num_schema_tokens
        self.num_tables = num_tables
        self.num_columns = num_columns
        self.L_T = L_T
        self.L_C = L_C
        self.N = N
        self.hidden_dim = hidden_dim

        # Indices are fixed at init; we learn the codebook entries themselves.
        self.register_buffer("table_id", table_id_per_token.long())
        self.register_buffer("col_id", col_id_per_token.long())
        gen = torch.Generator().manual_seed(0)
        self.register_buffer(
            "tab_indices", torch.randint(0, N, (num_tables, L_T), generator=gen)
        )
        self.register_buffer(
            "col_indices", torch.randint(0, N, (num_columns, L_C), generator=gen)
        )

        scale = 1.0 / math.sqrt(hidden_dim)
        self.BT = nn.Parameter(torch.randn(L_T, N, hidden_dim, generator=gen) * scale)
        self.BC = nn.Parameter(torch.randn(L_C, N, hidden_dim, generator=gen) * scale)

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------
    def compose_all(self) -> torch.Tensor:
        """Return (num_schema_tokens, hidden_dim) of composed token embeddings.

        e_τ = Σ_l BT[l, tab_idx_l] + Σ_l BC[l, col_idx_l]
        """
        tab_idx = self.tab_indices[self.table_id]   # (V_s, L_T)
        col_idx = self.col_indices[self.col_id]     # (V_s, L_C)
        e = self.BT.new_zeros(self.num_schema_tokens, self.hidden_dim)
        for l in range(self.L_T):
            e = e + self.BT[l][tab_idx[:, l]]
        for l in range(self.L_C):
            e = e + self.BC[l][col_idx[:, l]]
        return e

    def compose_local(self, local_ids: torch.Tensor) -> torch.Tensor:
        """`local_ids` shape (...) → (..., d)."""
        all_emb = self.compose_all()
        return all_emb[local_ids]

    # ------------------------------------------------------------------
    # Diversity regularisation (Eq. 9 in the paper)
    # ------------------------------------------------------------------
    def diversity_loss(self, K: int) -> torch.Tensor:
        """Per-codebook InfoNCE-style diversity loss.

        For each codebook B (BT and BC) we cluster its codes with k-means
        and pull each code towards a random same-cluster sibling while
        pushing it away from all other codes.
        """
        from sklearn.cluster import KMeans
        device = self.BT.device
        losses = []
        for codebook in (self.BT, self.BC):
            codes = codebook.reshape(-1, self.hidden_dim)  # (L*N, d)
            M = codes.size(0)
            if M < 2:
                continue
            codes_norm = F.normalize(codes, dim=-1)
            with torch.no_grad():
                np_codes = codes.detach().to(torch.float32).cpu().numpy()
                k = max(2, min(K, M - 1))
                labels = KMeans(
                    n_clusters=k, n_init=3, random_state=0
                ).fit_predict(np_codes)
                labels = torch.as_tensor(labels, device=device)

            same = labels.unsqueeze(0) == labels.unsqueeze(1)  # (M, M)
            same.fill_diagonal_(False)
            has_pos = same.any(dim=-1)
            if not has_pos.any():
                continue
            # Random same-cluster positive index per row
            scores = torch.rand(M, M, device=device).masked_fill(~same, -1.0)
            pos_idx = scores.argmax(dim=-1)
            sims = codes_norm @ codes_norm.T              # (M, M)
            diag = torch.eye(M, dtype=torch.bool, device=device)
            sims = sims.masked_fill(diag, -1e9)
            pos_sim = sims.gather(1, pos_idx.unsqueeze(1)).squeeze(1)
            denom = torch.logsumexp(sims, dim=-1)
            loss = (-(pos_sim - denom))[has_pos].mean()
            losses.append(loss)
        if not losses:
            return torch.zeros((), device=device, dtype=self.BT.dtype)
        return sum(losses) / len(losses)


class SchemaInputEmbedding(nn.Module):
    """Hybrid token embedding: original frozen weights + codebook composition."""

    def __init__(self, orig_embedding_weight: torch.Tensor, codebook: SchemaCodebook, num_orig: int):
        super().__init__()
        # Keep the original weight as a (frozen) buffer so the optimiser
        # ignores it but its values are still used by F.embedding.
        self.register_buffer("orig_weight", orig_embedding_weight.detach().clone())
        self.codebook = codebook
        self.num_orig = num_orig

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        is_schema = input_ids >= self.num_orig
        safe_ids = input_ids.clamp_max(self.num_orig - 1)
        out = F.embedding(safe_ids, self.orig_weight)
        if is_schema.any():
            local_ids = (input_ids - self.num_orig).clamp_min(0)
            schema_emb = self.codebook.compose_local(local_ids)
            schema_emb = schema_emb.to(out.dtype)
            out = torch.where(is_schema.unsqueeze(-1), schema_emb, out)
        return out


class SchemaLMHead(nn.Module):
    """Frozen original LM-head rows + trainable rows for schema tokens.

    Mirrors `Linear(d, V_orig + V_schema, bias=False)` but with the original
    rows kept fixed.  Set `freeze_orig=False` to also fine-tune the original
    rows (needed in stage 2 per the paper).
    """

    def __init__(
        self,
        orig_lm_head_weight: torch.Tensor,
        num_schema_tokens: int,
        hidden_dim: int,
        freeze_orig: bool = True,
    ):
        super().__init__()
        if freeze_orig:
            self.register_buffer("orig_weight", orig_lm_head_weight.detach().clone())
        else:
            self.orig_weight = nn.Parameter(orig_lm_head_weight.detach().clone())
        self.schema_weight = nn.Parameter(
            torch.zeros(num_schema_tokens, hidden_dim, dtype=orig_lm_head_weight.dtype)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        orig_logits = F.linear(hidden, self.orig_weight)
        schema_logits = F.linear(hidden, self.schema_weight.to(hidden.dtype))
        return torch.cat([orig_logits, schema_logits], dim=-1)


# ----------------------------------------------------------------------
# High-level setup helper
# ----------------------------------------------------------------------

def build_schema_id_maps(schema_records: List[Dict]) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str, str], int], torch.Tensor, torch.Tensor]:
    """Index every (db, table) pair and every (db, table, column) triple.

    Returns:
        table_index: {(db, table): table_idx}
        column_index: {(db, table, column): column_idx}
        table_id_per_token: tensor of shape (num_schema_tokens,)
        col_id_per_token: tensor of shape (num_schema_tokens,)
    """
    table_index: Dict[Tuple[str, str], int] = {}
    column_index: Dict[Tuple[str, str, str], int] = {}
    for rec in schema_records:
        key_t = (rec["db_id"], rec["table"])
        if key_t not in table_index:
            table_index[key_t] = len(table_index)
        key_c = (rec["db_id"], rec["table"], rec["column"])
        if key_c not in column_index:
            column_index[key_c] = len(column_index)
    table_id_per_token = []
    col_id_per_token = []
    for rec in schema_records:
        table_id_per_token.append(table_index[(rec["db_id"], rec["table"])])
        col_id_per_token.append(column_index[(rec["db_id"], rec["table"], rec["column"])])
    return (
        table_index,
        column_index,
        torch.tensor(table_id_per_token, dtype=torch.long),
        torch.tensor(col_id_per_token, dtype=torch.long),
    )


def extend_tokenizer_and_model(
    tokenizer,
    model,
    schema_records: List[Dict],
    L_T: int,
    L_C: int,
    N: int,
    freeze_lm_head_orig: bool = True,
):
    """Add every `schema_record["schema_token"]` to the tokenizer, untie the
    LM head from the embedding, and replace both the embedding and the LM
    head with `SchemaInputEmbedding` / `SchemaLMHead`.

    Returns
    -------
    Dict with: tokenizer, model, codebook, num_orig, num_total,
    schema_token_to_local_id, local_id_to_schema_token.
    """
    schema_strings = [rec["schema_token"] for rec in schema_records]

    # Detect whether the schema tokens are already in the tokenizer (e.g. when
    # loading from a stage-1 checkpoint).
    pre_size = len(tokenizer)
    existing_ids = tokenizer.convert_tokens_to_ids(schema_strings)
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else -1
    already_present = all(i is not None and i != unk_id for i in existing_ids)

    if already_present:
        # Schema tokens already in vocab. Verify they sit contiguously at the
        # tail so our `num_orig` boundary still applies.
        ids = list(existing_ids)
        assert ids == list(range(min(ids), max(ids) + 1)), (
            "expected schema-token ids to be contiguous in the tokenizer"
        )
        num_total = pre_size
        num_orig = num_total - len(schema_strings)
    else:
        num_orig = pre_size
        n_added = tokenizer.add_tokens(schema_strings, special_tokens=False)
        num_total = len(tokenizer)
        assert n_added == len(schema_strings), (
            f"expected to add {len(schema_strings)} tokens, only added {n_added}"
        )

    num_schema = len(schema_strings)
    assert num_total == num_orig + num_schema, (
        f"unexpected vocab size: orig={num_orig} schema={num_schema} total={num_total}"
    )

    # Build maps from schema_token string to local id (0..num_schema-1)
    schema_token_to_local_id: Dict[str, int] = {}
    local_id_to_schema_token: Dict[int, str] = {}
    for i, rec in enumerate(schema_records):
        schema_token_to_local_id[rec["schema_token"]] = i
        local_id_to_schema_token[i] = rec["schema_token"]

    # Resize both embedding and lm_head to accommodate the new vocab IDs.
    model.resize_token_embeddings(num_total)
    # Untie if necessary - we want lm_head independent
    if getattr(model.config, "tie_word_embeddings", False):
        model.config.tie_word_embeddings = False

    # Build codebook
    table_index, column_index, table_id_per_token, col_id_per_token = build_schema_id_maps(schema_records)
    hidden_dim = model.config.hidden_size
    codebook = SchemaCodebook(
        num_schema_tokens=num_schema,
        table_id_per_token=table_id_per_token,
        col_id_per_token=col_id_per_token,
        num_tables=len(table_index),
        num_columns=len(column_index),
        L_T=L_T,
        L_C=L_C,
        N=N,
        hidden_dim=hidden_dim,
    )
    # Match dtype/device of the model
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    # Keep the codebook itself in float32 for precise updates; we cast on
    # the way out of `compose_local`.
    codebook.to(device=model_device, dtype=torch.float32)

    # Snapshot the original embedding/lm_head weights for the wrappers
    orig_input_embed = model.get_input_embeddings()
    orig_input_weight = orig_input_embed.weight[:num_orig].detach().clone()
    orig_lm_head = model.lm_head
    orig_lm_weight = orig_lm_head.weight[:num_orig].detach().clone()

    # Wrap embedding
    new_embed = SchemaInputEmbedding(orig_input_weight, codebook, num_orig)
    model.set_input_embeddings(new_embed)

    # Wrap lm_head
    new_lm_head = SchemaLMHead(
        orig_lm_weight,
        num_schema_tokens=num_schema,
        hidden_dim=hidden_dim,
        freeze_orig=freeze_lm_head_orig,
    )
    new_lm_head.to(device=model_device, dtype=model_dtype)
    model.lm_head = new_lm_head

    # Sanity: model.config.vocab_size now reflects num_total
    model.config.vocab_size = num_total

    return {
        "tokenizer": tokenizer,
        "model": model,
        "codebook": codebook,
        "num_orig": num_orig,
        "num_total": num_total,
        "schema_token_to_local_id": schema_token_to_local_id,
        "local_id_to_schema_token": local_id_to_schema_token,
        "table_index": table_index,
        "column_index": column_index,
    }


# ----------------------------------------------------------------------
# Param-group helpers (used by the trainers)
# ----------------------------------------------------------------------

def trainable_params_for_stage1(model, codebook):
    """Stage 1: only codebooks + new lm_head rows are trainable."""
    # Freeze every parameter on the model.
    for p in model.parameters():
        p.requires_grad = False
    # Codebook params are trainable.
    for p in codebook.parameters():
        p.requires_grad = True
    # LM head schema rows are trainable (orig is a buffer, so no-op for it).
    for n, p in model.lm_head.named_parameters():
        p.requires_grad = True


def trainable_params_for_stage2(model, codebook, lora_targets=None):
    """Stage 2: codebook frozen, embed_tokens frozen, backbone + lm_head trainable.

    For LoRA-based stage 2, `peft.get_peft_model` should be applied to the
    backbone before this is called (it injects LoRA modules and freezes
    everything else by default; we then re-enable lm_head separately).
    """
    for p in codebook.parameters():
        p.requires_grad = False
    for p in model.get_input_embeddings().parameters():
        p.requires_grad = False
    # Re-enable lm_head schema rows (peft would have frozen them otherwise)
    for n, p in model.lm_head.named_parameters():
        p.requires_grad = True
