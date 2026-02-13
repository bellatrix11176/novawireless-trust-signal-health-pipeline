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

Usage:
  python src/control_layer/integrity_control_layer.py
  python src/control_layer/integrity_control_layer.py --file novawireless_synthetic_calls.csv
  python src/control_layer/integrity_control_layer.py --crt_low 0 --crt_high 7200
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

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

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def coerce_timestamp(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        # keep naive timestamps; your framework uses utc=False later
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
    # Flags / binary indicator columns that should be 0/1 if present
    binary_like: tuple[str, ...] = ("callback_flag", "cleanup_flag", "port_out_flag", "flag_30DACC")

    # Optional ID column for duplicate checks
    unique_key: str = "interaction_id"

    # Timestamp sanity: allow null? (No—required.)
    timestamp_col: str = "timestamp"

    # CRT thresholds for sanity checks (not the same as your risk scaling; this is "is the value plausible?")
    crt_min_seconds: float = 0.0
    crt_max_seconds: float = 6 * 60 * 60  # 6 hours

    # If you have handle time or similar, you can add it here later.
    # Example:
    # handle_time_col: Optional[str] = "handle_time_sec"
    # handle_time_min: float = 0.0
    # handle_time_max: float = 4 * 60 * 60


def build_flags(df: pd.DataFrame, cfg: IntegrityConfig) -> pd.DataFrame:
    """
    Return a flags DataFrame with one row per record.
    Columns begin with 'flag_' and can be OR'ed into a master any_flag.
    """
    flags = pd.DataFrame(index=df.index)

    # Required column existence (dataset-level) => if missing, all rows are flagged
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

    # Duplicate unique key (if present)
    if cfg.unique_key in df.columns:
        flags["flag_duplicate_interaction_id"] = df.duplicated(subset=[cfg.unique_key], keep=False)
    else:
        flags["flag_duplicate_interaction_id"] = False

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

    # Master flag
    flag_cols = [c for c in flags.columns if c.startswith("flag_")]
    flags["any_flag"] = flags[flag_cols].any(axis=1) if flag_cols else False

    return flags


def summarize_flags(flags: pd.DataFrame, n_rows: int) -> dict:
    flag_cols = [c for c in flags.columns if c.startswith("flag_") and c != "any_flag"]
    rates = {c: float(flags[c].mean()) for c in flag_cols} if flag_cols else {}

    any_rate = float(flags["any_flag"].mean()) if "any_flag" in flags.columns else 0.0

    # top 10 flags by rate
    top = sorted(rates.items(), key=lambda kv: kv[1], reverse=True)[:10]

    return {
        "rows_total": int(n_rows),
        "rows_flagged": int(flags["any_flag"].sum()) if "any_flag" in flags.columns else 0,
        "rows_clean": int((~flags["any_flag"]).sum()) if "any_flag" in flags.columns else int(n_rows),
        "quarantine_rate": any_rate,
        "top_flag_rates": [{"flag": k, "rate": v} for k, v in top],
        "all_flag_rates": rates,
    }


# ---------------------------
# Main
# ---------------------------

def run_integrity_gate(
    input_path: Path,
    out_dir: Path,
    cfg: IntegrityConfig,
) -> dict:
    df_raw = load_table(input_path)

    # Basic cleaning: strip col names already done
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

    flags = build_flags(df, cfg)

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

    result = run_integrity_gate(input_path=input_path, out_dir=out_dir, cfg=cfg)

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
    print("Wrote artifacts to /output:")
    print(f"- {Path(s['output_clean']).name}")
    print(f"- {Path(s['output_quarantine']).name}")
    print(f"- {Path(s['output_flags']).name}")
    print(f"- {Path('integrity_summary.json').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
