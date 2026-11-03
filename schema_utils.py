"""Common helpers shared across SchemaToken stages."""
import os
import yaml
import json
import re
from typing import Any, Dict, List


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg.get("paths", {}).items():
        if k.endswith("_dir") and isinstance(v, str):
            os.makedirs(v, exist_ok=True)
    return cfg


def schema_token_string(db_id: str, table: str, column: str) -> str:
    """Canonical textual form of a schema token.  We prefix every schema
    token with `<sch:>` so they don't collide with anything else and so we
    can recognise them in model output.  The form is stable: db.table.column.
    """
    safe = lambda s: re.sub(r"[^A-Za-z0-9_]+", "_", s)
    return f"<sch:{safe(db_id)}.{safe(table)}.{safe(column)}>"


def parse_schema_tokens(text: str) -> list[str]:
    """Extract `<sch:...>` tokens that appear in `text`."""
    return re.findall(r"<sch:[A-Za-z0-9_.]+>", text)


def jaccard(pred: set, gt: set) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    return len(pred & gt) / len(pred | gt)
