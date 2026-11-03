"""Data pipeline for SchemaToken on the BIRD-dev benchmark.

Responsibilities
----------------
1. Load BIRD's `dev_tables.json` and `dev.json`.
2. Read the per-table CSV description files in
   `dev_databases/<db_id>/database_description/<table>.csv` to obtain a
   natural-language description for every column.
3. Build a registry of schema tokens (one per (db, table, column)) along
   with the canonical text + description.
4. Extract the gold columns referenced by each dev question's SQL using a
   simple lexical match (suitable for the BIRD-dev SQL syntax).

Outputs
-------
A JSON file with two top-level lists:
  * `schema`: every schema token in the active databases.
  * `examples`: one entry per dev question with its gold schema-token list.
"""
import argparse
import csv
import json
import os
import re
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List

from schema_utils import load_config, schema_token_string


def parse_args():
    p = argparse.ArgumentParser(description="SchemaToken - data pipeline")
    p.add_argument("--config", default=None, type=str)
    return p.parse_args()


def load_table_descriptions(db_dir: str, table_name: str) -> Dict[str, str]:
    """Parse <db>/database_description/<table>.csv → {column: description}.

    BIRD ships these CSVs with a UTF-8 BOM and inconsistent quoting; we use
    csv.DictReader with utf-8-sig and normalise column names.
    """
    path = os.path.join(db_dir, "database_description", f"{table_name}.csv")
    out: Dict[str, str] = {}
    if not os.path.exists(path):
        return out
    # Some files are latin-1 instead of utf-8 because of accented chars.
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get("original_column_name") or "").strip()
                    if not name:
                        continue
                    desc = (row.get("column_description") or row.get("column_name") or "").strip()
                    val_desc = (row.get("value_description") or "").strip()
                    if val_desc:
                        desc = f"{desc}. Value: {val_desc}" if desc else val_desc
                    if not desc:
                        desc = name
                    out[name] = desc
            return out
        except UnicodeDecodeError:
            continue
        except Exception:
            return out
    return out


def list_real_columns(db_path: str) -> Dict[str, List[str]]:
    """Get authoritative column names per table from the SQLite file.

    The CSVs are sometimes stale; the SQLite schema is the source of truth.
    """
    if not os.path.exists(db_path):
        return {}
    out: Dict[str, List[str]] = {}
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        for t in tables:
            try:
                cur.execute(f"PRAGMA table_info(`{t}`)")
                cols = [r[1] for r in cur.fetchall()]
                out[t] = cols
            except Exception:
                continue
        con.close()
    except Exception:
        pass
    return out


def build_schema(cfg: dict) -> List[Dict[str, Any]]:
    paths = cfg["paths"]
    with open(paths["bird_tables_json"], "r", encoding="utf-8") as f:
        all_tables = json.load(f)

    selected = set(cfg["data"].get("databases") or [])
    schema_records: List[Dict[str, Any]] = []
    for entry in all_tables:
        db_id = entry["db_id"]
        if selected and db_id not in selected:
            continue
        db_dir = os.path.join(paths["bird_databases_dir"], db_id)
        sqlite_path = os.path.join(db_dir, f"{db_id}.sqlite")
        real_cols = list_real_columns(sqlite_path)

        # Per-table descriptions
        descriptions: Dict[str, Dict[str, str]] = {}
        for t in entry["table_names_original"]:
            descriptions[t] = load_table_descriptions(db_dir, t)

        for tab_idx, col_name in entry["column_names_original"]:
            if tab_idx == -1:
                continue  # skip global '*'
            table_name = entry["table_names_original"][tab_idx]
            # Skip columns that aren't actually present in the SQLite file
            if real_cols and col_name not in real_cols.get(table_name, []):
                continue
            desc = descriptions.get(table_name, {}).get(col_name) or col_name
            schema_records.append({
                "db_id": db_id,
                "table": table_name,
                "column": col_name,
                "description": desc,
                "schema_token": schema_token_string(db_id, table_name, col_name),
            })
    return schema_records


# ----------------------------------------------------------------------
# Gold-column extraction from the SQL string
# ----------------------------------------------------------------------

_TOKEN_RE = re.compile(r"`([^`]+)`|\"([^\"]+)\"|\b([A-Za-z_][A-Za-z0-9_ ]*)\b")
_ALIAS_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE)
_SIMPLE_COL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)|`([^`]+)`\.`([^`]+)`")


def extract_columns_from_sql(sql: str, db_columns: Dict[str, List[str]]) -> List[tuple]:
    """Best-effort lexical extractor: returns list of (table, column).

    We resolve table aliases (e.g. `T1`, `T2`) and then look for `<alias>.col`
    or `` `col` `` references. If a bare column appears we attribute it to
    whichever table actually contains that column (single-table queries).
    """
    if not sql:
        return []

    # Resolve aliases: `<table> AS <alias>`
    aliases: Dict[str, str] = {}
    for m in _ALIAS_RE.finditer(sql):
        table_name, alias = m.group(1), m.group(2)
        # Some SQLs use the table name itself in lowercase - normalise via
        # case-insensitive match against db_columns keys.
        for t in db_columns.keys():
            if t.lower() == table_name.lower():
                aliases[alias] = t
                break
        else:
            aliases[table_name] = table_name
    # Also map every real table to itself (for unaliased uses)
    for t in db_columns.keys():
        aliases.setdefault(t, t)

    found = set()

    # Pattern: alias.column or `table`.`column`
    for m in _SIMPLE_COL_RE.finditer(sql):
        if m.group(1):
            alias, col = m.group(1), m.group(2)
        else:
            alias, col = m.group(3), m.group(4)
        table = aliases.get(alias, alias)
        # Resolve case
        for tname in db_columns.keys():
            if tname.lower() == table.lower():
                # Find col case-insensitively
                for c in db_columns[tname]:
                    if c.lower() == col.lower():
                        found.add((tname, c))
                        break
                break

    # Pattern: bare backticked or quoted column names
    for raw in re.findall(r"`([^`.]+)`", sql) + re.findall(r"\"([^\".]+)\"", sql):
        for tname, cols in db_columns.items():
            for c in cols:
                if c == raw:
                    found.add((tname, c))

    # Pattern: bare unquoted identifiers that exactly match a column name
    # (only safe when the query uses a single table, otherwise this picks up
    # accidental word matches from longer backtick-quoted column names).
    if len(db_columns) == 1:
        only_table, only_cols = next(iter(db_columns.items()))
        word_to_col = {c.lower(): c for c in only_cols if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", c)}
        # Strip out backticked / quoted regions before scanning to avoid
        # picking up sub-words from inside `Free Meal Count (K-12)` etc.
        clean = re.sub(r"`[^`]*`", "", sql)
        clean = re.sub(r"\"[^\"]*\"", "", clean)
        for w in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", clean):
            c = word_to_col.get(w.lower())
            if c is not None:
                found.add((only_table, c))

    return sorted(found)


def build_examples(cfg: dict, schema_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    paths = cfg["paths"]
    with open(paths["bird_dev_json"], "r", encoding="utf-8") as f:
        dev = json.load(f)

    # Group columns by db
    db_to_cols: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    db_to_tokens: Dict[tuple, str] = {}
    for rec in schema_records:
        db_to_cols[rec["db_id"]][rec["table"]].append(rec["column"])
        db_to_tokens[(rec["db_id"], rec["table"], rec["column"])] = rec["schema_token"]

    selected = set(cfg["data"].get("databases") or [])
    cap = int(cfg["data"].get("max_dev_questions_per_db", -1))
    counts: Dict[str, int] = defaultdict(int)

    examples = []
    for q in dev:
        if selected and q["db_id"] not in selected:
            continue
        if cap > 0 and counts[q["db_id"]] >= cap:
            continue
        counts[q["db_id"]] += 1

        db_columns = {t: list(cs) for t, cs in db_to_cols[q["db_id"]].items()}
        gold_pairs = extract_columns_from_sql(q["SQL"], db_columns)
        gold_tokens = [
            db_to_tokens[(q["db_id"], t, c)]
            for (t, c) in gold_pairs
            if (q["db_id"], t, c) in db_to_tokens
        ]
        examples.append({
            "question_id": q["question_id"],
            "db_id": q["db_id"],
            "question": q["question"],
            "evidence": q.get("evidence", ""),
            "sql": q["SQL"],
            "gold_pairs": gold_pairs,
            "gold_schema_tokens": gold_tokens,
        })
    return examples


def main():
    args = parse_args()
    cfg = load_config(args.config)

    schema_records = build_schema(cfg)
    examples = build_examples(cfg, schema_records)

    out_path = cfg["paths"]["schema_meta"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "schema": schema_records,
        "examples": examples,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    by_db = defaultdict(int)
    for r in schema_records:
        by_db[r["db_id"]] += 1
    print("[data] schema tokens:", len(schema_records))
    for k, v in by_db.items():
        print(f"  - {k}: {v} columns")
    print("[data] dev examples:", len(examples))
    have_gold = sum(1 for e in examples if e["gold_schema_tokens"])
    print(f"[data] examples with extracted gold tokens: {have_gold}/{len(examples)}")
    print(f"[data] wrote {out_path}")


if __name__ == "__main__":
    main()
