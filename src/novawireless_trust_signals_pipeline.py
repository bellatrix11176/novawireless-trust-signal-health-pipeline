#!/usr/bin/env python3
"""
novawireless_trust_signals_pipeline.py

ONE SCRIPT PIPELINE (Integrity Gate + Trust Signal Analysis)

What it does (end-to-end):
1) Reads /data/<file> (default: novawireless_synthetic_calls.csv)
2) Runs an Integrity Gate:
   - timestamp parse
   - required fields present
   - binary flag coercion + non-binary detection
   - CRT sanity range (if CRT column exists)
   - duplicate interaction_id handling via --dupe_policy
   Outputs:
     /output/calls_clean.csv
     /output/calls_quarantine.csv
     /output/integrity_flags.csv
     /output/integrity_summary.json

3) Promotes the CLEAN file into analysis and runs Trust Signal scoring + story outputs:
   - optional PII scrubbing for GitHub safety
   - repeat-contact features
   - drift_score (with optional CRT risk)
   - rep/ivr/customer summaries
   - charts + correlation heatmaps
   - summary_report.txt
   Outputs:
     /output/calls_scrubbed.csv (optional if scrub enabled)
     /output/calls_scored_cleaned.csv
     /output/rep_summary.csv
     /output/ivr_summary.csv
     /output/customer_repeat_summary.csv
     /output/summary_report.txt
     /output/*.png charts

Repo-root style:
- Finds repo root by locating folders: data/ and src/
- Always writes to repo_root/output/

Usage:
  python src/novawireless_trust_signals_pipeline.py run
  python src/novawireless_trust_signals_pipeline.py run --file novawireless_synthetic_calls.csv
  python src/novawireless_trust_signals_pipeline.py run --dupe_policy quarantine_extras_keep_latest
  python src/novawireless_trust_signals_pipeline.py run --no_scrub_pii
  python src/novawireless_trust_signals_pipeline.py run --crt_low 480 --crt_high 900 --crt_min 0 --crt_max 21600

Notes:
- This script does NOT depend on analyze_trust_signals.py or integrity_control_layer.py.
- It is fully standalone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Repo-root helpers
# ============================================================

def find_repo_root(start: Path) -> Path:
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


def get_input_file(data_dir: Path, filename: str) -> Path:
    p = data_dir / filename
    if not p.exists():
        raise FileNotFoundError(
            f"Expected input file not found: {p}\n"
            "Place your dataset in /data and rerun."
        )
    return p


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ============================================================
# Shared coercions / detection
# ============================================================

def _coerce_flag(series: pd.Series) -> pd.Series:
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


def detect_30dacc_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    preferred = ["flag_30DACC", "30DACC_flag", "flag30DACC", "dacc_30_flag", "dacc30_flag"]
    for c in preferred:
        if c in cols:
            return c
    for c in cols:
        cl = c.lower()
        if "30" in cl and ("dacc" in cl or "acc" in cl) and ("flag" in cl or "indicator" in cl):
            return c
    return None


def detect_crt_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    preferred = ["CRT_seconds", "CRT", "crt", "crt_seconds", "crt_sec", "CRT_sec", "CRTSeconds", "crtSeconds"]
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


# ============================================================
# PART 1 — Integrity Gate
# ============================================================

@dataclass
class IntegrityConfig:
    required_columns: tuple[str, ...] = ("timestamp", "customer_phone")
    binary_like: tuple[str, ...] = ("callback_flag", "cleanup_flag", "port_out_flag", "flag_30DACC")
    unique_key: str = "interaction_id"
    timestamp_col: str = "timestamp"
    crt_min_seconds: float = 0.0
    crt_max_seconds: float = 6 * 60 * 60  # 6 hours


def coerce_timestamp(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _valid_key_mask(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    missing_like = {"", "nan", "none", "null", "UNKNOWN", "N/A"}
    return ~s.str.lower().isin({m.lower() for m in missing_like})


def build_flags(df: pd.DataFrame, cfg: IntegrityConfig) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)

    missing_cols = [c for c in cfg.required_columns if c not in df.columns]
    flags["flag_missing_required_column"] = False
    if missing_cols:
        flags["flag_missing_required_column"] = True

    for c in cfg.required_columns:
        if c in df.columns:
            flags[f"flag_missing_{c}"] = df[c].isna()

    if cfg.timestamp_col in df.columns:
        flags["flag_bad_timestamp_parse"] = df[cfg.timestamp_col].isna()

    for c in cfg.binary_like:
        if c in df.columns:
            flags[f"flag_nonbinary_{c}"] = ~df[c].isin([0, 1])

    crt_col = detect_crt_column(df)
    if crt_col is not None:
        crt_vals = pd.to_numeric(df[crt_col], errors="coerce")
        flags["flag_missing_crt"] = crt_vals.isna()
        flags["flag_crt_out_of_range"] = (crt_vals < cfg.crt_min_seconds) | (crt_vals > cfg.crt_max_seconds)
    else:
        flags["flag_missing_crt"] = False
        flags["flag_crt_out_of_range"] = False

    flags["flag_duplicate_interaction_id"] = False
    return flags


def apply_dupe_policy(df: pd.DataFrame, cfg: IntegrityConfig, dupe_policy: str) -> Tuple[pd.Series, dict]:
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

    cols = [cfg.unique_key] + ([cfg.timestamp_col] if cfg.timestamp_col in df.columns else [])
    dfx = df.loc[key_valid, cols].copy()

    dup_involved = dfx.duplicated(subset=[cfg.unique_key], keep=False)
    involved_idx = dfx.index[dup_involved]
    stats["duplicate_rows_involved"] = int(len(involved_idx))

    if stats["duplicate_rows_involved"] == 0:
        return pd.Series(False, index=df.index), stats

    stats["duplicate_ids_count"] = int(dfx.loc[dup_involved, cfg.unique_key].nunique())

    if dupe_policy == "quarantine_all":
        out = pd.Series(False, index=df.index)
        out.loc[involved_idx] = True
        stats["duplicate_rows_removed_extras"] = stats["duplicate_rows_involved"]
        return out, stats

    if dupe_policy in {"quarantine_extras_keep_latest", "quarantine_extras_keep_first"}:
        has_ts = cfg.timestamp_col in df.columns
        if has_ts:
            dfx[cfg.timestamp_col] = pd.to_datetime(dfx[cfg.timestamp_col], errors="coerce")
            ascending = True if dupe_policy == "quarantine_extras_keep_first" else False
            dfx_sorted = dfx.sort_values([cfg.unique_key, cfg.timestamp_col], ascending=[True, ascending], kind="mergesort")
        else:
            ascending = True if dupe_policy == "quarantine_extras_keep_first" else False
            dfx_sorted = dfx.sort_values([cfg.unique_key], kind="mergesort")
            if not ascending:
                dfx_sorted = dfx_sorted.iloc[::-1]

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


def run_integrity_gate(input_path: Path, out_dir: Path, cfg: IntegrityConfig, dupe_policy: str) -> dict:
    df_raw = load_table(input_path)
    df = df_raw.copy()

    df = coerce_timestamp(df, col=cfg.timestamp_col)

    for c in cfg.binary_like:
        if c in df.columns:
            df[c] = _coerce_flag(df[c])

    crt_col = detect_crt_column(df)
    if crt_col is not None:
        df[crt_col] = pd.to_numeric(df[crt_col], errors="coerce")

    flags = build_flags(df, cfg)

    dupe_quarantine_mask, dupe_stats = apply_dupe_policy(df, cfg, dupe_policy=dupe_policy)
    flags["flag_duplicate_interaction_id"] = dupe_quarantine_mask

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


# ============================================================
# PART 2 — Trust Signal Analysis (integrated)
# ============================================================

def _hash_series(series: pd.Series, salt: str) -> pd.Series:
    def h(v: object) -> str:
        s = "" if pd.isna(v) else str(v).strip()
        raw = (salt + "|" + s).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
    return series.apply(h)


def scrub_pii_df(
    df: pd.DataFrame,
    salt: str = "trust-signals-salt",
    hash_interaction_id: bool = True,
    keep_text: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]

    required = ["timestamp", "customer_phone"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for downstream analysis: {missing}")

    out["customer_phone"] = "cust_" + _hash_series(out["customer_phone"], salt=salt).str.slice(0, 16)

    if hash_interaction_id and "interaction_id" in out.columns:
        out["interaction_id"] = "int_" + _hash_series(out["interaction_id"], salt=salt).str.slice(0, 16)

    drop_cols: list[str] = []
    for c in ["account_number", "first_name", "last_name", "email", "address"]:
        if c in out.columns:
            drop_cols.append(c)

    if not keep_text:
        for c in ["rep_memo", "full_transcript"]:
            if c in out.columns:
                drop_cols.append(c)

    drop_cols = sorted(set(drop_cols))
    if drop_cols:
        out = out.drop(columns=drop_cols)

    return out, drop_cols


def validate_columns(df: pd.DataFrame) -> None:
    required = ["timestamp", "customer_phone"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "rep_id" in out.columns:
        out["rep_id"] = out["rep_id"].astype(str).str.strip()
        out.loc[out["rep_id"].isin(["", "nan", "none", "None"]), "rep_id"] = "UNKNOWN"

    if "ivr_reason" in out.columns:
        out["ivr_reason"] = out["ivr_reason"].astype(str).str.strip()
        out.loc[out["ivr_reason"].isin(["", "nan", "none", "None"]), "ivr_reason"] = "UNKNOWN"

    return out


def scale_high_is_worse(x: pd.Series, low: float, high: float) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    out = (s - low) / (high - low)
    out = out.clip(lower=0.0, upper=1.0)
    return out.fillna(0.0)


def engineer_features(df: pd.DataFrame, repeat_window_hours: int = 72) -> pd.DataFrame:
    out = df.copy()
    out["ts"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=False)

    for col in ["callback_flag", "cleanup_flag", "port_out_flag"]:
        if col in out.columns:
            out[col] = _coerce_flag(out[col])
        else:
            out[col] = 0

    dacc_col = detect_30dacc_column(out)
    if dacc_col is None:
        out["flag_30DACC"] = 0
    else:
        out["flag_30DACC"] = _coerce_flag(out[dacc_col])

    out = out.sort_values(["customer_phone", "ts"], kind="mergesort").reset_index(drop=True)

    window = pd.Timedelta(hours=repeat_window_hours)
    out["is_repeat_contact"] = 0
    out["repeat_count_in_window"] = 0

    for _, grp in out.groupby("customer_phone", sort=False):
        times = grp["ts"].values
        idxs = grp.index.values
        j = 0
        for i in range(len(times)):
            while j < i and pd.Timestamp(times[i]) - pd.Timestamp(times[j]) > window:
                j += 1
            prior_in_window = i - j
            out.loc[idxs[i], "repeat_count_in_window"] = prior_in_window
            out.loc[idxs[i], "is_repeat_contact"] = 1 if prior_in_window > 0 else 0

    def disposition_row(r) -> str:
        if r.get("port_out_flag", 0) == 1:
            return "Port-Out"
        if r.get("flag_30DACC", 0) == 1:
            return "Delayed Deactivation (30DACC)"
        if r.get("cleanup_flag", 0) == 1:
            return "Cleanup / Correction"
        if r.get("callback_flag", 0) == 1:
            return "Callback / Follow-up"
        return "Resolved / No Follow-up Observed"

    out["final_disposition"] = out.apply(disposition_row, axis=1)
    return out


def compute_trust_metrics(df: pd.DataFrame, crt_low: float = 480.0, crt_high: float = 900.0) -> pd.DataFrame:
    out = df.copy()
    for c in ["callback_flag", "cleanup_flag", "flag_30DACC", "port_out_flag", "is_repeat_contact"]:
        if c in out.columns:
            out[c] = _coerce_flag(out[c])

    crt_col = detect_crt_column(out)
    if crt_col is None:
        out["crt_value"] = np.nan
        out["crt_risk"] = 0.0
        has_crt = False
    else:
        out["crt_value"] = pd.to_numeric(out[crt_col], errors="coerce")
        out["crt_risk"] = scale_high_is_worse(out["crt_value"], low=crt_low, high=crt_high)
        has_crt = True

    if has_crt:
        out["drift_score"] = (
            0.30 * out["flag_30DACC"]
            + 0.20 * out["cleanup_flag"]
            + 0.20 * out["port_out_flag"]
            + 0.15 * out["is_repeat_contact"]
            + 0.15 * out["crt_risk"]
        )
    else:
        out["drift_score"] = (
            0.35 * out["flag_30DACC"]
            + 0.25 * out["cleanup_flag"]
            + 0.20 * out["port_out_flag"]
            + 0.20 * out["is_repeat_contact"]
        )
    return out


def summarize_by_rep(df: pd.DataFrame, crt_over: float = 900.0) -> pd.DataFrame:
    if "rep_id" not in df.columns:
        return pd.DataFrame()

    agg_map = {
        "calls": ("interaction_id", "count") if "interaction_id" in df.columns else ("ts", "count"),
        "cbr": ("callback_flag", "mean"),
        "cds": ("cleanup_flag", "mean"),
        "dacc30": ("flag_30DACC", "mean"),
        "por": ("port_out_flag", "mean"),
        "repeat_rate": ("is_repeat_contact", "mean"),
        "drift_score_avg": ("drift_score", "mean"),
    }

    has_crt = "crt_value" in df.columns and df["crt_value"].notna().any()
    if has_crt:
        dfx = df.copy()
        dfx["crt_over_flag"] = (pd.to_numeric(dfx["crt_value"], errors="coerce") > float(crt_over)).astype(int)
        agg_map.update({
            "crt_avg": ("crt_value", "mean"),
            "crt_p90": ("crt_value", lambda s: float(np.nanpercentile(pd.to_numeric(s, errors="coerce"), 90))),
            "crt_over_rate": ("crt_over_flag", "mean"),
            "crt_risk_avg": ("crt_risk", "mean"),
        })
        rep = dfx.groupby("rep_id", dropna=False).agg(**agg_map).reset_index()
    else:
        rep = df.groupby("rep_id", dropna=False).agg(**agg_map).reset_index()

    if has_crt and "crt_risk_avg" in rep.columns:
        rep["trust_score"] = (
            1.0
            - 0.25 * rep["dacc30"]
            - 0.20 * rep["cds"]
            - 0.25 * rep["por"]
            - 0.15 * rep["repeat_rate"]
            - 0.15 * rep["crt_risk_avg"]
        )
    else:
        rep["trust_score"] = (
            1.0
            - 0.30 * rep["dacc30"]
            - 0.25 * rep["cds"]
            - 0.25 * rep["por"]
            - 0.20 * rep["repeat_rate"]
        )

    rep = rep.sort_values(["trust_score", "calls"], ascending=[False, False]).reset_index(drop=True)
    return rep


def summarize_by_ivr(df: pd.DataFrame, crt_over: float = 900.0) -> pd.DataFrame:
    if "ivr_reason" not in df.columns:
        return pd.DataFrame()

    agg_map = {
        "calls": ("interaction_id", "count") if "interaction_id" in df.columns else ("ts", "count"),
        "cbr": ("callback_flag", "mean"),
        "cds": ("cleanup_flag", "mean"),
        "dacc30": ("flag_30DACC", "mean"),
        "por": ("port_out_flag", "mean"),
        "repeat_rate": ("is_repeat_contact", "mean"),
        "drift_score_avg": ("drift_score", "mean"),
    }

    has_crt = "crt_value" in df.columns and df["crt_value"].notna().any()
    if has_crt:
        dfx = df.copy()
        dfx["crt_over_flag"] = (pd.to_numeric(dfx["crt_value"], errors="coerce") > float(crt_over)).astype(int)
        agg_map.update({
            "crt_avg": ("crt_value", "mean"),
            "crt_p90": ("crt_value", lambda s: float(np.nanpercentile(pd.to_numeric(s, errors="coerce"), 90))),
            "crt_over_rate": ("crt_over_flag", "mean"),
            "crt_risk_avg": ("crt_risk", "mean"),
        })
        ivr = dfx.groupby("ivr_reason", dropna=False).agg(**agg_map).reset_index()
    else:
        ivr = df.groupby("ivr_reason", dropna=False).agg(**agg_map).reset_index()

    ivr = ivr.sort_values(["drift_score_avg", "calls"], ascending=[False, False]).reset_index(drop=True)
    return ivr


def summarize_repeat_customers(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("customer_phone", dropna=False)
    cust = g.agg(
        calls=("interaction_id", "count") if "interaction_id" in df.columns else ("ts", "count"),
        any_port_out=("port_out_flag", "max"),
        any_30dacc=("flag_30DACC", "max"),
        any_cleanup=("cleanup_flag", "max"),
        any_callback=("callback_flag", "max"),
        avg_drift=("drift_score", "mean"),
        first_ts=("ts", "min"),
        last_ts=("ts", "max"),
    ).reset_index()
    cust = cust.sort_values(["calls", "avg_drift"], ascending=[False, False]).reset_index(drop=True)
    return cust


# ---------------- Charts ----------------

def save_barh(df: pd.DataFrame, x: str, y: str, title: str, outpath: Path, top_n: int = 15) -> None:
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return
    d = df.head(top_n).copy().sort_values(x, ascending=True)
    plt.figure(figsize=(10, max(4, 0.35 * len(d) + 1)))
    plt.barh(d[y].astype(str), d[x].astype(float))
    plt.title(title)
    plt.xlabel(x)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def save_hist(series: pd.Series, title: str, outpath: Path, bins: int = 20) -> None:
    if series is None or series.dropna().empty:
        return
    plt.figure(figsize=(10, 5))
    plt.hist(series.dropna().astype(float), bins=bins)
    plt.title(title)
    plt.xlabel(series.name or "value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def make_story_charts(rep: pd.DataFrame, ivr: pd.DataFrame, cust: pd.DataFrame, out_dir: Path) -> None:
    if ivr is not None and not ivr.empty and {"calls", "drift_score_avg"}.issubset(ivr.columns):
        plt.figure(figsize=(10, 6))
        plt.scatter(ivr["calls"], ivr["drift_score_avg"])
        plt.title("IVR hot zones: Drift vs Volume")
        plt.xlabel("Calls (volume)")
        plt.ylabel("Average drift_score")
        plt.tight_layout()
        plt.savefig(out_dir / "story_ivr_drift_vs_volume.png", dpi=160)
        plt.close()

    if rep is not None and not rep.empty and {"calls", "trust_score"}.issubset(rep.columns):
        plt.figure(figsize=(10, 6))
        plt.scatter(rep["calls"], rep["trust_score"])
        plt.title("Rep landscape: Trust Score vs Volume")
        plt.xlabel("Calls handled")
        plt.ylabel("Trust score (higher is better)")
        plt.tight_layout()
        plt.savefig(out_dir / "story_rep_trust_vs_volume.png", dpi=160)
        plt.close()

    if cust is not None and not cust.empty and "calls" in cust.columns:
        cust_sorted = cust.sort_values("calls", ascending=False).reset_index(drop=True)
        cust_sorted["cum_calls"] = cust_sorted["calls"].cumsum()
        cust_sorted["cum_calls_pct"] = cust_sorted["cum_calls"] / max(cust_sorted["calls"].sum(), 1)
        cust_sorted["cust_pct"] = (cust_sorted.index + 1) / max(len(cust_sorted), 1)

        plt.figure(figsize=(10, 6))
        plt.plot(cust_sorted["cust_pct"], cust_sorted["cum_calls_pct"])
        plt.title("Pareto: % of customers vs % of calls")
        plt.xlabel("Fraction of customers")
        plt.ylabel("Cumulative fraction of calls")
        plt.tight_layout()
        plt.savefig(out_dir / "story_customer_pareto_calls.png", dpi=160)
        plt.close()


def plot_corr_heatmap(corr: pd.DataFrame, title: str, outpath: Path) -> None:
    if corr is None or corr.empty:
        return
    cols = list(corr.columns)
    data = corr.values.astype(float)

    plt.figure(figsize=(max(8, 0.6 * len(cols) + 2), max(6, 0.6 * len(cols) + 2)))
    plt.imshow(data, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            v = data[i, j]
            if np.isfinite(v):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def make_corr_heatmaps(rep_summary: pd.DataFrame, out_dir: Path) -> None:
    if rep_summary is None or rep_summary.empty:
        return
    numeric_cols = [c for c in rep_summary.columns if c != "rep_id" and pd.api.types.is_numeric_dtype(rep_summary[c])]
    if len(numeric_cols) < 2:
        return
    signals = [c for c in numeric_cols if c != "trust_score"]
    if len(signals) >= 2:
        plot_corr_heatmap(rep_summary[signals].corr(numeric_only=True),
                          "Rep-level correlation heatmap (signals only)",
                          out_dir / "rep_corr_heatmap_signals_only.png")
    if "trust_score" in rep_summary.columns:
        plot_corr_heatmap(rep_summary[numeric_cols].corr(numeric_only=True),
                          "Rep-level correlation heatmap (including derived trust_score)",
                          out_dir / "rep_corr_heatmap_with_trust.png")


# ---------------- Reporting ----------------

def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def _drift_concentration_calls(df: pd.DataFrame, top_frac: float = 0.10) -> float:
    if len(df) == 0:
        return 0.0
    s = df["drift_score"].fillna(0).astype(float).sort_values(ascending=False)
    k = max(1, int(np.ceil(top_frac * len(s))))
    return _safe_div(s.head(k).sum(), s.sum())


def _drift_concentration_customers(df: pd.DataFrame, top_frac: float = 0.10) -> float:
    if len(df) == 0:
        return 0.0
    g = df.groupby("customer_phone", dropna=False).agg(
        calls=("interaction_id", "count") if "interaction_id" in df.columns else ("ts", "count"),
        avg_drift=("drift_score", "mean"),
    )
    g["total_drift_proxy"] = g["calls"].astype(float) * g["avg_drift"].astype(float)
    g = g.sort_values("total_drift_proxy", ascending=False)
    k = max(1, int(np.ceil(top_frac * len(g))))
    return _safe_div(g.head(k)["total_drift_proxy"].sum(), g["total_drift_proxy"].sum())


def write_summary_report(
    df: pd.DataFrame,
    rep_summary: pd.DataFrame,
    ivr_summary: pd.DataFrame,
    outpath: Path,
    min_calls_for_rank: int = 20,
    crt_low: float = 480.0,
    crt_high: float = 900.0,
    crt_over: float = 900.0,
) -> None:
    n = len(df)
    if n == 0:
        outpath.write_text("No rows found.\n", encoding="utf-8")
        return

    def pct(x: float) -> str:
        return f"{100*x:.1f}%"

    overall = {
        "CBR (callback rate)": df["callback_flag"].mean(),
        "CDS (cleanup rate)": df["cleanup_flag"].mean(),
        "30DACC rate": df["flag_30DACC"].mean(),
        "POR (port-out rate)": df["port_out_flag"].mean(),
        "Repeat contact rate": df["is_repeat_contact"].mean(),
        "Avg drift_score": df["drift_score"].mean(),
    }

    has_crt = "crt_value" in df.columns and df["crt_value"].notna().any()
    if has_crt:
        crt_vals = pd.to_numeric(df["crt_value"], errors="coerce")
        overall.update({
            "CRT avg (seconds)": float(np.nanmean(crt_vals)),
            "CRT p90 (seconds)": float(np.nanpercentile(crt_vals, 90)),
            f"CRT > {int(crt_over)}s rate": float(np.nanmean((crt_vals > crt_over).astype(float))),
            f"CRT risk avg (scaled {int(crt_low)}–{int(crt_high)}s)": float(np.nanmean(df["crt_risk"].astype(float))),
        })

    top10_calls_share = _drift_concentration_calls(df, top_frac=0.10)
    top10_cust_share = _drift_concentration_customers(df, top_frac=0.10)

    lines: list[str] = []
    lines.append("TRUST SIGNAL HEALTH — SUMMARY REPORT")
    lines.append("=" * 40)
    lines.append(f"Rows analyzed: {n}")
    lines.append("")

    lines.append("OVERALL RATES")
    for k, v in overall.items():
        if "Avg" in k or "CRT" in k or "p90" in k or "seconds" in k or "risk" in k:
            if "rate" in k.lower() and "CRT" in k:
                lines.append(f"- {k}: {pct(v)}")
            else:
                lines.append(f"- {k}: {v:.3f}" if "risk" in k else f"- {k}: {v:.1f}")
        else:
            lines.append(f"- {k}: {pct(v)}")
    lines.append("")

    if has_crt:
        lines.append("CRT INTERPRETATION")
        lines.append("- CRT is treated as a friction cost signal (higher is worse).")
        lines.append(f"- CRT risk is scaled: <= {int(crt_low)}s → 0, >= {int(crt_high)}s → 1.")
        lines.append("")

    lines.append("DRIFT CONCENTRATION (where risk clusters)")
    lines.append(f"- Top 10% of calls account for: {pct(top10_calls_share)} of total drift_score")
    lines.append(f"- Top 10% of customers account for: {pct(top10_cust_share)} of total drift_score (proxy)")
    lines.append("")

    lines.append("TOP 5 IVR REASONS BY DRIFT (avg)")
    if ivr_summary is None or ivr_summary.empty:
        lines.append("- (No ivr_reason column found; skipping IVR section)")
    else:
        base_cols = ["ivr_reason", "calls", "drift_score_avg", "por", "dacc30", "cds", "cbr"]
        extra_cols = ["crt_avg", "crt_p90", "crt_over_rate"] if "crt_avg" in ivr_summary.columns else []
        top_ivr = ivr_summary.head(5)[base_cols + extra_cols]
        for _, r in top_ivr.iterrows():
            msg = (
                f"- {r['ivr_reason']} | calls={int(r['calls'])} | drift={r['drift_score_avg']:.3f} "
                f"| POR={pct(r['por'])} | 30DACC={pct(r['dacc30'])} | CDS={pct(r['cds'])} | CBR={pct(r['cbr'])}"
            )
            if "crt_avg" in ivr_summary.columns:
                msg += f" | CRT_avg={r['crt_avg']:.0f}s | CRT_p90={r['crt_p90']:.0f}s | CRT>thr={pct(r['crt_over_rate'])}"
            lines.append(msg)
    lines.append("")

    if rep_summary is None or rep_summary.empty:
        lines.append("NO REP SUMMARY FOUND (missing rep_id?)")
        lines.append("")
        rep_rank = rep_summary
    else:
        qualified = rep_summary[rep_summary["calls"] >= min_calls_for_rank].copy()
        rep_rank = qualified if len(qualified) >= 5 else rep_summary.copy()

        lines.append(f"REPS FOUND: {rep_summary['rep_id'].nunique(dropna=False)}")
        lines.append(f"REPS WITH >= {min_calls_for_rank} CALLS: {(rep_summary['calls'] >= min_calls_for_rank).sum()}")
        lines.append("")

    lines.append("TOP 5 REPS BY TRUST_SCORE (higher is better)")
    if rep_rank is None or rep_rank.empty:
        lines.append("- No reps available to rank.")
    else:
        cols = ["rep_id", "calls", "trust_score", "drift_score_avg", "por", "dacc30", "cds", "cbr"]
        extra = ["crt_avg", "crt_p90", "crt_over_rate"] if "crt_avg" in rep_rank.columns else []
        top_rep = rep_rank.sort_values(["trust_score", "calls"], ascending=[False, False]).head(5)[cols + extra]
        for _, r in top_rep.iterrows():
            msg = (
                f"- {r['rep_id']} | calls={int(r['calls'])} | trust_score={r['trust_score']:.3f} "
                f"| drift={r['drift_score_avg']:.3f} | POR={pct(r['por'])} | 30DACC={pct(r['dacc30'])} "
                f"| CDS={pct(r['cds'])} | CBR={pct(r['cbr'])}"
            )
            if "crt_avg" in rep_rank.columns:
                msg += f" | CRT_avg={r['crt_avg']:.0f}s | CRT_p90={r['crt_p90']:.0f}s | CRT>thr={pct(r['crt_over_rate'])}"
            lines.append(msg)

    lines.append("")
    lines.append("INTERPRETATION HINTS")
    lines.append("- Headline KPIs can look stable while drift indicators rise (30DACC/port-outs/repeats/CRT).")
    lines.append("- Cleanup isn’t failure; it’s often proof you’re fixing upstream mistakes (or catching bad inputs).")
    lines.append("- CRT is a cost signal: even if outcomes are stable, rising CRT suggests growing friction/complexity.")
    lines.append("- Concentration tells you whether to fix hotspots or address broad process issues.")
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# PIPELINE COMMAND
# ============================================================

def cmd_run(args: argparse.Namespace) -> int:
    repo_root = find_repo_root(Path.cwd())
    data_dir = repo_root / "data"
    out_dir = repo_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_path = get_input_file(data_dir, args.file)

    # --- Integrity gate
    cfg = IntegrityConfig(crt_min_seconds=float(args.crt_min), crt_max_seconds=float(args.crt_max))
    gate = run_integrity_gate(in_path, out_dir, cfg=cfg, dupe_policy=args.dupe_policy)
    s = gate["summary"]

    # --- Promote clean file into analysis
    clean_path = gate["clean_path"]
    df_clean = load_table(clean_path)

    # --- Optional scrub for GitHub safety
    if args.scrub_pii:
        df_work, dropped = scrub_pii_df(
            df_clean,
            salt=args.salt,
            hash_interaction_id=True,
            keep_text=args.keep_text,
        )
        scrubbed_out = out_dir / "calls_scrubbed.csv"
        df_work.to_csv(scrubbed_out, index=False)
    else:
        df_work = df_clean.copy()
        dropped = []
        scrubbed_out = None

    df_work = clean_ids(df_work)
    validate_columns(df_work)

    df2 = engineer_features(df_work, repeat_window_hours=args.repeat_window_hours)
    df2 = compute_trust_metrics(df2, crt_low=args.crt_low, crt_high=args.crt_high)

    calls_scored_path = out_dir / "calls_scored_cleaned.csv"
    df2.to_csv(calls_scored_path, index=False)

    rep_summary = summarize_by_rep(df2, crt_over=args.crt_over)
    ivr_summary = summarize_by_ivr(df2, crt_over=args.crt_over)
    cust_summary = summarize_repeat_customers(df2)

    rep_path = out_dir / "rep_summary.csv"
    ivr_path = out_dir / "ivr_summary.csv"
    cust_path = out_dir / "customer_repeat_summary.csv"
    rep_summary.to_csv(rep_path, index=False)
    ivr_summary.to_csv(ivr_path, index=False)
    cust_summary.to_csv(cust_path, index=False)

    report_path = out_dir / "summary_report.txt"
    write_summary_report(
        df2,
        rep_summary=rep_summary,
        ivr_summary=ivr_summary,
        outpath=report_path,
        min_calls_for_rank=args.min_calls_for_rank,
        crt_low=args.crt_low,
        crt_high=args.crt_high,
        crt_over=args.crt_over,
    )

    save_barh(rep_summary, "trust_score", "rep_id", "Top reps by trust_score (higher is better)", out_dir / "rep_trust_score_top.png", top_n=15)
    save_barh(ivr_summary, "drift_score_avg", "ivr_reason", "IVR reasons with highest average drift_score", out_dir / "ivr_drift_top.png", top_n=15)
    save_hist(df2["drift_score"], "Distribution of drift_score (row-level)", out_dir / "drift_score_hist.png", bins=20)

    if "crt_value" in df2.columns and df2["crt_value"].notna().any():
        save_hist(df2["crt_value"], "Distribution of CRT (seconds)", out_dir / "crt_seconds_hist.png", bins=25)

    make_story_charts(rep_summary, ivr_summary, cust_summary, out_dir)
    make_corr_heatmaps(rep_summary, out_dir)

    # --- Console story (non-robot version)
    print("NOVA TRUST SIGNALS PIPELINE — COMPLETE")
    print("=" * 48)
    print(f"Input file:                 {in_path.name}")
    print(f"Integrity clean rows:       {s['rows_clean']} / {s['rows_total']}  (quarantine {s['quarantine_rate']:.2%})")
    if s.get("crt_detected_column"):
        print(f"CRT column detected:        {s['crt_detected_column']}  (sanity {s['crt_sanity_min_seconds']}..{s['crt_sanity_max_seconds']} sec)")
    print(f"Duplicate policy:           {s.get('dupe_policy')}")
    print(f"Duplicates involved rows:   {s.get('duplicate_rows_involved')}")
    print("")
    print("Artifacts written to /output:")
    print(f"- calls_clean.csv")
    print(f"- calls_quarantine.csv")
    print(f"- integrity_flags.csv")
    print(f"- integrity_summary.json")
    if scrubbed_out is not None:
        print(f"- calls_scrubbed.csv   (PII-safe-ish for GitHub)")
        if dropped:
            print(f"  dropped columns: {dropped}")
    print(f"- calls_scored_cleaned.csv")
    print(f"- rep_summary.csv")
    print(f"- ivr_summary.csv")
    print(f"- customer_repeat_summary.csv")
    print(f"- summary_report.txt")
    print(f"- *.png charts")
    print("Done.")
    return 0


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NovaWireless Trust Signals — ONE SCRIPT Pipeline")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="Run integrity gate + trust-signal analysis end-to-end.")
    r.add_argument("--file", default="novawireless_synthetic_calls.csv")
    r.add_argument("--repeat_window_hours", type=int, default=72)
    r.add_argument("--min_calls_for_rank", type=int, default=20)

    # Integrity CRT sanity
    r.add_argument("--crt_min", type=float, default=0.0)
    r.add_argument("--crt_max", type=float, default=6 * 60 * 60)

    # Analysis CRT risk thresholds
    r.add_argument("--crt_low", type=float, default=480.0)
    r.add_argument("--crt_high", type=float, default=900.0)
    r.add_argument("--crt_over", type=float, default=900.0)

    # Duplicates policy
    r.add_argument(
        "--dupe_policy",
        default="quarantine_all",
        choices=["quarantine_all", "quarantine_extras_keep_latest", "quarantine_extras_keep_first"],
    )

    # PII scrubbing
    r.add_argument("--no_scrub_pii", dest="scrub_pii", action="store_false")
    r.set_defaults(scrub_pii=True)
    r.add_argument("--salt", default="trust-signals-salt")
    r.add_argument("--keep_text", action="store_true")

    r.set_defaults(func=cmd_run)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    argv2 = argv if argv is not None else (sys.argv[1:] or ["run"])
    args = parser.parse_args(argv2)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
