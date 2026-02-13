#!/usr/bin/env python3
"""
integrity_control_layer.py

NovaWireless Trust Signals: Integrity Control Layer (Synthetic)

Purpose:
- Validate dataset integrity BEFORE trust-signal scoring/analysis
- Flag rows that violate integrity rules
- Split into CLEAN vs QUARANTINE datasets
- Write audit artifacts to /output

Reads:
- /data/<file>  (default: novawireless_synthetic_calls.csv)

Writes:
- /output/calls_clean.csv
- /output/calls_quarantine.csv
- /output/integrity_flags.csv
- /output/integrity_summary.json

Duplicate handling (--dupe_policy):
- quarantine_all (default):
    Flag and quarantine ALL rows that are part of a duplicated interaction_id group.
- quarantine_extras_keep_latest:
    Keep ONE row per duplicated interaction_id (the latest by timestamp),
    quarantine ONLY the extra rows.
- quarantine_extras_keep_first:
    Keep ONE row per duplicated interaction_id (the first by timestamp),
    quarantine ONLY the extra rows.

Usage:
  python src/control_layer/integrity_control_layer.py
  python src/control_layer/integrity_control_layer.py --file novawireless_synthetic_calls.csv
  python src/control_layer/integrity_control_layer.py --dupe_policy quarantine_extras_keep_latest
  python src/control_layer/integrity_control_layer.py --crt_min 0 --crt_max 21600
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Repo path helpers (repo-root style)
# ---------------------------

def find_repo_root(start: Path) -> Path:
    """
    Finds the folder that contains:
      - data/
      - src/
    Ensures output/ exists (creates if missing).
    """
    cur = start.resolve()
    for _ in range(50):
        if (cur / "data").is_dir() and (cur / "src").is_dir():
            (cur / "output").mkdir(parents=True, exist_ok=True)
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError(
        "Could not find repo root containing data/ and src/.\n"
        "Run this script from somewhere inside your repo."
    )


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ---------------------------
# Utility coercions
# ---------------------------

def coerce_timestamp(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _coerce_flag(series: pd.Series) -> pd.Series:
    """
    Coerce a column into 0/1 ints. Handles numeric, strings, booleans.
    """
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(int)

    s = s.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y"}
    false_set = {"0", "false", "f", "no", "n", "nan", "none", ""}

    out = []
    for v in s:
        if v in true_set:
            out.append(1)
        elif v in false_set:
            out.append(0)
        else:
            try:
                out.append(1 if float(v) > 0 else 0)
            except Exception:
                out.append(0)
    return pd.Series(out, index=series.index, dtype="int64")


def detect_crt_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect a CRT column. Variants like CRT_seconds, CRT, crt, crt_sec, etc.
    """
    cols = list(df.columns)
    preferred = [
        "CRT_seconds", "CRT", "crt", "crt_seconds", "crt_sec", "CRT_sec", "CRTSeconds", "crtSeconds"
    ]
    for c in preferred:
        if c in cols:
            return c

    for c in cols:
        cl = c.lower()
        if "crt" in cl and ("sec" in cl or "seconds" in cl or cl == "crt"):
            return c

    for c in cols:
        if "crt" in c.lower():
            return c

    return None


# ---------------------------
# Integrity Rules
# ---------------------------

@dataclass
class IntegrityConfig:
    required_columns: tuple[str, ...] = ("timestamp", "customer_phone")
    binary_like: tuple[str, ...] = ("callback_flag", "cleanup_flag", "port_out_flag", "flag_30DACC")
    unique_key: str = "interaction_id"
    timestamp_col: str = "timestamp"
    crt_min_seconds: float = 0.0
    crt_max_seconds: float = 6 * 60 * 60  # 6 hours


def _valid_key_mask(series: pd.Series) -> pd.Series:
    """
    Defines which keys are considered "valid" (counted for duplicate detection).
    Missing-like keys are ignored for dupe logic.
    """
    s = series.astype(str).str.strip()
    missing_like = {"", "nan", "none", "null", "UNKNOWN", "N/A"}
    return ~s.str.lower().isin({m.lower() for m in missing_like})


def build_flags(df: pd.DataFrame, cfg: IntegrityConfig) -> pd.DataFrame:
    """
    Base integrity flags (excluding dupe_policy resolution).
    """
    flags = pd.DataFrame(index=df.index)

    # Required column existence (dataset-level)
    missing_cols = [c for c in cfg.required_columns if c not in df.columns]
    flags["flag_missing_required_column"] = False
    if missing_cols:
        flags["flag_missing_required_column"] = True

    # Required fields non-null
    for c in cfg.required_columns:
        if c in df.columns:
            flags[f"flag_missing_{c}"] = df[c].isna()

    # Timestamp parse success
    if cfg.timestamp_col in df.columns:
        flags["flag_bad_timestamp_parse"] = df[cfg.timestamp_col].isna()

    # Binary-like columns must be {0,1} if present (after coercion)
    for c in cfg.binary_like:
        if c in df.columns:
            flags[f"flag_nonbinary_{c}"] = ~df[c].isin([0, 1])

    # CRT sanity checks if CRT exists
    crt_col = detect_crt_column(df)
    if crt_col is not None:
        crt_vals = pd.to_numeric(df[crt_col], errors="coerce")
        flags["flag_missing_crt"] = crt_vals.isna()
        flags["flag_crt_out_of_range"] = (crt_vals < cfg.crt_min_seconds) | (crt_vals > cfg.crt_max_seconds)
    else:
        flags["flag_missing_crt"] = False
        flags["flag_crt_out_of_range"] = False

    # Placeholder; will be set by dupe policy handler
    flags["flag_duplicate_interaction_id"] = False
    return flags


def apply_dupe_policy(
    df: pd.DataFrame,
    cfg: IntegrityConfig,
    dupe_policy: str,
) -> Tuple[pd.Series, dict]:
    """
    Returns:
      dup_flag_series: bool series aligned to df.index indicating rows to quarantine due to duplicate policy
      stats: dict with duplicate-related counts for reporting

    Notes:
    - Only applies if cfg.unique_key exists in df.columns.
    - Only considers "valid" keys (non-empty, non-unknown).
    """
    stats = {
        "unique_key_present": cfg.unique_key in df.columns,
        "dupe_policy": dupe_policy,
        "duplicate_ids_count": 0,
        "duplicate_rows_involved": 0,
        "duplicate_rows_removed_extras": 0,
    }

    if cfg.unique_key not in df.columns:
        return pd.Series(False, index=df.index), stats

    key_valid = _valid_key_mask(df[cfg.unique_key])
    if not key_valid.any():
        return pd.Series(False, index=df.index), stats

    dfx = df.loc[key_valid, [cfg.unique_key] + ([cfg.timestamp_col] if cfg.timestamp_col in df.columns else [])].copy()

    # Identify duplicate groups among valid keys
    dup_involved = dfx.duplicated(subset=[cfg.unique_key], keep=False)
    involved_idx = dfx.index[dup_involved]

    stats["duplicate_rows_involved"] = int(len(involved_idx))

    if stats["duplicate_rows_involved"] == 0:
        return pd.Series(False, index=df.index), stats

    stats["duplicate_ids_count"] = int(dfx.loc[dup_involved, cfg.unique_key].nunique())

    # Policy A: quarantine_all => quarantine all involved rows
    if dupe_policy == "quarantine_all":
        out = pd.Series(False, index=df.index)
        out.loc[involved_idx] = True
        stats["duplicate_rows_removed_extras"] = stats["duplicate_rows_involved"]
        return out, stats

    # Policies that quarantine only "extras" and keep one row per duplicated key
    if dupe_policy in {"quarantine_extras_keep_latest", "quarantine_extras_keep_first"}:
        # If no timestamp column, fall back to stable index order
        has_ts = cfg.timestamp_col in df.columns

        if has_ts:
            # Ensure timestamp is datetime (caller already coerced)
            dfx[cfg.timestamp_col] = pd.to_datetime(dfx[cfg.timestamp_col], errors="coerce")
            # Sort per policy
            ascending = True if dupe_policy == "quarantine_extras_keep_first" else False
            dfx_sorted = dfx.sort_values([cfg.unique_key, cfg.timestamp_col], ascending=[True, ascending], kind="mergesort")
        else:
            # No timestamp: keep_first means lowest index, keep_latest means highest index
            ascending = True if dupe_policy == "quarantine_extras_keep_first" else False
            dfx_sorted = dfx.sort_values([cfg.unique_key], kind="mergesort")
            # Within each key, index order is natural; flip if keeping "latest"
            if not ascending:
                dfx_sorted = dfx_sorted.iloc[::-1]

        # Keep one per key (the first row in sorted order), quarantine the rest
        keeper_idx = dfx_sorted.drop_duplicates(subset=[cfg.unique_key], keep="first").index
        extras_idx = dfx_sorted.index.difference(keeper_idx)

        out = pd.Series(False, index=df.index)
        out.loc[extras_idx] = True

        stats["duplicate_rows_removed_extras"] = int(len(extras_idx))
        return out, stats

    raise ValueError(
        f"Unknown dupe_policy: {dupe_policy}\n"
        "Allowed: quarantine_all, quarantine_extras_keep_latest, quarantine_extras_keep_first"
    )


def summarize_flags(flags: pd.DataFrame, n_rows: int) -> dict:
    flag_cols = [c for c in flags.columns if c.startswith("flag_") and c != "any_flag"]
    rates = {c: float(flags[c].mean()) for c in flag_cols} if flag_cols else {}

    any_rate = float(flags["any_flag"].mean()) if "any_flag" in flags.columns else 0.0
    top = sorted(rates.items(), key=lambda kv: kv[1], reverse=True)[:10]

    return {
        "rows_total": int(n_rows),
        "rows_flagged": int(flags["any_flag"].sum()) if "any_flag" in flags.columns else 0,
        "rows_clean": int((~flags["any_flag"]).sum()) if "any_flag" in flags.columns else int(n_rows),
        "quarantine_rate": any_rate,
        "top_flag_rates": [{"flag": k, "rate": v} for k, v in top],
        "all_flag_rates": rates,
    }


def run_integrity_gate(
    input_path: Path,
    out_dir: Path,
    cfg: IntegrityConfig,
    dupe_policy: str,
) -> dict:
    df_raw = load_table(input_path)
    df = df_raw.copy()

    # Coerce timestamp
    df = coerce_timestamp(df, col=cfg.timestamp_col)

    # Coerce binary-like columns to 0/1 if present
    for c in cfg.binary_like:
        if c in df.columns:
            df[c] = _coerce_flag(df[c])

    # Coerce CRT numeric if present
    crt_col = detect_crt_column(df)
    if crt_col is not None:
        df[crt_col] = pd.to_numeric(df[crt_col], errors="coerce")

    # Base flags
    flags = build_flags(df, cfg)

    # Apply dupe policy => set duplicate flag
    dupe_quarantine_mask, dupe_stats = apply_dupe_policy(df, cfg, dupe_policy=dupe_policy)
    flags["flag_duplicate_interaction_id"] = dupe_quarantine_mask

    # Master flag
    flag_cols = [c for c in flags.columns if c.startswith("flag_")]
    flags["any_flag"] = flags[flag_cols].any(axis=1) if flag_cols else False

    clean_df = df.loc[~flags["any_flag"]].copy()
    quarantine_df = df.loc[flags["any_flag"]].copy()
    quarantine_flags = flags.loc[flags["any_flag"]].copy()

    out_dir.mkdir(parents=True, exist_ok=True)

    clean_path = out_dir / "calls_clean.csv"
    quarantine_path = out_dir / "calls_quarantine.csv"
    flags_path = out_dir / "integrity_flags.csv"
    summary_path = out_dir / "integrity_summary.json"

    clean_df.to_csv(clean_path, index=False)
    quarantine_df.to_csv(quarantine_path, index=False)
    quarantine_flags.to_csv(flags_path, index=False)

    summary = summarize_flags(flags, n_rows=len(df))
    summary.update({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(input_path.name),
        "output_clean": str(clean_path.name),
        "output_quarantine": str(quarantine_path.name),
        "output_flags": str(flags_path.name),
        "crt_detected_column": crt_col,
        "crt_sanity_min_seconds": cfg.crt_min_seconds,
        "crt_sanity_max_seconds": cfg.crt_max_seconds,
        "required_columns": list(cfg.required_columns),
        "binary_like_columns": list(cfg.binary_like),
        "unique_key": cfg.unique_key,
        # dupe stats
        **dupe_stats,
    })

    save_json(summary_path, summary)

    return {
        "clean_path": clean_path,
        "quarantine_path": quarantine_path,
        "flags_path": flags_path,
        "summary_path": summary_path,
        "summary": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Integrity Control Layer (Synthetic) — NovaWireless Trust Signals")
    p.add_argument(
        "--file",
        default="novawireless_synthetic_calls.csv",
        help="Input filename inside /data (default: novawireless_synthetic_calls.csv)",
    )
    p.add_argument("--crt_min", type=float, default=0.0, help="CRT sanity minimum (seconds).")
    p.add_argument("--crt_max", type=float, default=6 * 60 * 60, help="CRT sanity maximum (seconds).")

    p.add_argument(
        "--dupe_policy",
        default="quarantine_all",
        choices=["quarantine_all", "quarantine_extras_keep_latest", "quarantine_extras_keep_first"],
        help="How to handle duplicate interaction_id values.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    repo_root = find_repo_root(Path.cwd())
    data_dir = repo_root / "data"
    out_dir = repo_root / "output"

    input_path = data_dir / args.file
    if not input_path.exists():
        raise FileNotFoundError(
            f"Expected input file not found: {input_path}\n"
            "Place your dataset in /data and rerun."
        )

    cfg = IntegrityConfig(
        crt_min_seconds=float(args.crt_min),
        crt_max_seconds=float(args.crt_max),
    )

    result = run_integrity_gate(
        input_path=input_path,
        out_dir=out_dir,
        cfg=cfg,
        dupe_policy=args.dupe_policy,
    )

    s = result["summary"]
    print("INTEGRITY CONTROL LAYER — COMPLETE")
    print("=" * 40)
    print(f"Input:              {input_path}")
    print(f"Rows total:         {s['rows_total']}")
    print(f"Rows clean:         {s['rows_clean']}")
    print(f"Rows quarantined:   {s['rows_flagged']}")
    print(f"Quarantine rate:    {s['quarantine_rate']:.2%}")
    if s.get("crt_detected_column"):
        print(f"CRT column detected: {s['crt_detected_column']}")
        print(f"CRT sanity range:    {s['crt_sanity_min_seconds']}..{s['crt_sanity_max_seconds']} seconds")
    print("")
    print("Duplicate handling:")
    print(f"- dupe_policy:                 {s.get('dupe_policy')}")
    print(f"- duplicate_ids_count:         {s.get('duplicate_ids_count')}")
    print(f"- duplicate_rows_involved:     {s.get('duplicate_rows_involved')}")
    print(f"- duplicate_rows_removed_extras:{s.get('duplicate_rows_removed_extras')}")
    print("")
    print("Wrote artifacts to /output:")
    print(f"- {Path(s['output_clean']).name}")
    print(f"- {Path(s['output_quarantine']).name}")
    print(f"- {Path(s['output_flags']).name}")
    print(f"- integrity_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
