#!/usr/bin/env python3
"""Analyze local sweep results from DCE_NIK runs directory.

Usage:
    python analyze_sweeps.py                          # all runs
    python analyze_sweeps.py --runs-dir runs          # custom dir
    python analyze_sweeps.py --filter-family siren    # single family
    python analyze_sweeps.py --filter-date 20260313   # single date
    python analyze_sweeps.py --csv results.csv        # export to CSV
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---- Parsing ----

# New format: Model: family=siren, hidden=256, depth=4, w0=30.0, k_freq=64, k_sigma=6.0, wd=0.0, params=...
_MODEL_NEW_RE = re.compile(
    r"Model: family=(\w+), hidden=(\d+), depth=(\d+), "
    r"w0=([\d.]+)(?:, k_freq=(\d+))?(?:, k_sigma=([\d.]+))?(?:, wd=([\d.eE+-]+))?"
)
# Old format: Model: hidden=128, depth=5, w0=15.0, params=...
_MODEL_OLD_RE = re.compile(
    r"Model: hidden=(\d+), depth=(\d+), w0=([\d.]+)"
)
_BEST_RE = re.compile(r"best_val_loss=([\d.eE+-]+)")
_SKIP_RE = re.compile(r"Skipping: (.+)")
# Extract model_family from config line (old sweeps that had no family= in Model line)
_FAMILY_RE = re.compile(r"'model_family':\s*'(\w+)'")
# Image-space metrics
_CONJ_RE = re.compile(r"Constraints:.*conj_weight=([\d.eE+-]+)")
_DENSITY_RE = re.compile(r"Constraints:.*density_weight=([\d.eE+-]+)")
_PSNR_MEAS_RE = re.compile(r"Metrics vs measured:.*PSNR=([\d.eE+-]+) dB")
_SSIM_MEAS_RE = re.compile(r"Metrics vs measured:.*SSIM=([\d.eE+-]+)")
_NRMSE_MEAS_RE = re.compile(r"Metrics vs measured:.*NRMSE=([\d.eE+-]+)")
_PSNR_GT_RE = re.compile(r"Metrics vs GT:.*PSNR=([\d.eE+-]+) dB")
_SSIM_GT_RE = re.compile(r"Metrics vs GT:.*SSIM=([\d.eE+-]+)")
_NRMSE_GT_RE = re.compile(r"Metrics vs GT:.*NRMSE=([\d.eE+-]+)")


def _parse_image_metrics(text: str) -> dict:
    """Extract image-space metrics from log text. Returns dict with None for missing."""
    def _match_float(regex, txt):
        m = regex.search(txt)
        return float(m.group(1)) if m else None

    return {
        "psnr_meas": _match_float(_PSNR_MEAS_RE, text),
        "ssim_meas": _match_float(_SSIM_MEAS_RE, text),
        "nrmse_meas": _match_float(_NRMSE_MEAS_RE, text),
        "psnr_gt": _match_float(_PSNR_GT_RE, text),
        "ssim_gt": _match_float(_SSIM_GT_RE, text),
        "nrmse_gt": _match_float(_NRMSE_GT_RE, text),
    }


def parse_run(run_dir: Path):
    log = run_dir / "training.log"
    if not log.exists():
        return None

    text = log.read_text()

    # Check skipped
    m_skip = _SKIP_RE.search(text)
    if m_skip:
        return {"skipped": True, "skip_reason": m_skip.group(1), "run": run_dir.name}

    # Parse best val loss
    m_best = _BEST_RE.search(text)
    if not m_best:
        return None

    img_metrics = _parse_image_metrics(text)

    # Parse constraint weights (new step 13+ logs)
    m_conj = _CONJ_RE.search(text)
    m_dens = _DENSITY_RE.search(text)
    conj_weight = float(m_conj.group(1)) if m_conj else 0.0
    density_weight = float(m_dens.group(1)) if m_dens else 0.0

    # Try new format first
    m_model = _MODEL_NEW_RE.search(text)
    if m_model:
        return {
            "skipped": False,
            "run": run_dir.name,
            "date": run_dir.name[:8],
            "family": m_model.group(1),
            "hidden": int(m_model.group(2)),
            "depth": int(m_model.group(3)),
            "w0": float(m_model.group(4)),
            "k_freq": int(m_model.group(5)) if m_model.group(5) else None,
            "k_sigma": float(m_model.group(6)) if m_model.group(6) else None,
            "weight_decay": float(m_model.group(7)) if m_model.group(7) else 0.0,
            "conj_weight": conj_weight,
            "density_weight": density_weight,
            "best_val_loss": float(m_best.group(1)),
            **img_metrics,
        }

    # Try old format
    m_old = _MODEL_OLD_RE.search(text)
    if m_old:
        # Try to get family from config line, default to "siren" (old runs were all siren)
        m_fam = _FAMILY_RE.search(text)
        family = m_fam.group(1) if m_fam else "siren"
        return {
            "skipped": False,
            "run": run_dir.name,
            "date": run_dir.name[:8],
            "family": family,
            "hidden": int(m_old.group(1)),
            "depth": int(m_old.group(2)),
            "w0": float(m_old.group(3)),
            "k_freq": None,
            "k_sigma": None,
            "weight_decay": 0.0,
            "best_val_loss": float(m_best.group(1)),
            **img_metrics,
        }

    return None


# ---- Display ----

def print_table(rows: list, columns: list, title: str = ""):
    if not rows:
        print(f"  (no data)\n")
        return
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    # Column widths
    widths = {c: max(len(c), max(len(fmt_val(r.get(c))) for r in rows)) for c in columns}
    header = " | ".join(c.rjust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    print(f"  {header}")
    print(f"  {sep}")
    for r in rows:
        line = " | ".join(fmt_val(r.get(c)).rjust(widths[c]) for c in columns)
        print(f"  {line}")
    print()


def fmt_val(v):
    if v is None:
        return "--"
    if isinstance(v, float):
        if abs(v) < 0.01 or abs(v) > 1e4:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def analyze(runs, filter_family=None):
    completed = [r for r in runs if not r["skipped"]]
    skipped = [r for r in runs if r["skipped"]]

    if filter_family:
        completed = [r for r in completed if r["family"] == filter_family]

    print(f"\nTotal runs: {len(runs)}  |  Completed: {len(completed)}  |  Skipped: {len(skipped)}")

    # ---- Best per family ----
    by_family = defaultdict(list)
    for r in completed:
        by_family[r["family"]].append(r)

    best_per_family = []
    for fam, fam_runs in sorted(by_family.items()):
        best = min(fam_runs, key=lambda r: r["best_val_loss"])
        best_per_family.append(best)
    best_per_family.sort(key=lambda r: r["best_val_loss"])

    cols = ["family", "hidden", "depth", "w0", "k_sigma", "best_val_loss",
            "psnr_meas", "ssim_meas", "nrmse_meas", "date"]
    print_table(best_per_family, cols, "Best Run Per Family (by k-space val loss)")

    # ---- Best per family by image PSNR ----
    has_img = [r for r in completed if r.get("psnr_meas") is not None]
    if has_img:
        by_family_img = defaultdict(list)
        for r in has_img:
            by_family_img[r["family"]].append(r)
        best_img = []
        for fam, fam_runs in sorted(by_family_img.items()):
            best = max(fam_runs, key=lambda r: r["psnr_meas"])
            best_img.append(best)
        best_img.sort(key=lambda r: r["psnr_meas"], reverse=True)
        print_table(best_img, cols, "Best Run Per Family (by image PSNR vs measured)")

    # ---- Top 10 overall ----
    top10 = sorted(completed, key=lambda r: r["best_val_loss"])[:10]
    print_table(top10, cols, "Top 10 Runs Overall (by k-space val loss)")

    if has_img:
        top10_img = sorted(has_img, key=lambda r: r["psnr_meas"], reverse=True)[:10]
        print_table(top10_img, cols, "Top 10 Runs Overall (by image PSNR vs measured)")

    # ---- Depth analysis per family ----
    for fam in sorted(by_family):
        fam_runs = by_family[fam]
        depth_groups = defaultdict(list)
        for r in fam_runs:
            depth_groups[r["depth"]].append(r)

        depth_rows = []
        for d in sorted(depth_groups):
            group = depth_groups[d]
            losses = [r["best_val_loss"] for r in group]
            best = min(group, key=lambda r: r["best_val_loss"])
            depth_rows.append({
                "depth": d,
                "n_runs": len(group),
                "min_loss": min(losses),
                "mean_loss": sum(losses) / len(losses),
                "best_hidden": best["hidden"],
                "best_w0": best["w0"],
                "best_k_sigma": best.get("k_sigma"),
            })
        dcols = ["depth", "n_runs", "min_loss", "mean_loss", "best_hidden", "best_w0", "best_k_sigma"]
        print_table(depth_rows, dcols, f"Depth Analysis: {fam}")

    # ---- k_sigma analysis for FF families ----
    for fam in sorted(by_family):
        if not fam.startswith("ff_"):
            continue
        fam_runs = by_family[fam]
        sigma_groups = defaultdict(list)
        for r in fam_runs:
            if r.get("k_sigma") is not None:
                sigma_groups[r["k_sigma"]].append(r)

        if not sigma_groups:
            continue

        sigma_rows = []
        for s in sorted(sigma_groups):
            group = sigma_groups[s]
            losses = [r["best_val_loss"] for r in group]
            sigma_rows.append({
                "k_sigma": s,
                "n_runs": len(group),
                "min_loss": min(losses),
                "mean_loss": sum(losses) / len(losses),
            })
        scols = ["k_sigma", "n_runs", "min_loss", "mean_loss"]
        print_table(sigma_rows, scols, f"k_sigma Analysis: {fam}")

    # ---- w0 analysis for siren ----
    if "siren" in by_family:
        siren_runs = by_family["siren"]
        w0_groups = defaultdict(list)
        for r in siren_runs:
            w0_groups[r["w0"]].append(r)

        if len(w0_groups) > 1:
            w0_rows = []
            for w in sorted(w0_groups):
                group = w0_groups[w]
                losses = [r["best_val_loss"] for r in group]
                w0_rows.append({
                    "w0": w,
                    "n_runs": len(group),
                    "min_loss": min(losses),
                    "mean_loss": sum(losses) / len(losses),
                })
            print_table(w0_rows, ["w0", "n_runs", "min_loss", "mean_loss"],
                        "w0 Analysis: siren")

    # ---- Weight decay analysis per family ----
    wd_families = defaultdict(lambda: defaultdict(list))
    for r in completed:
        wd_families[r["family"]][r.get("weight_decay", 0.0)].append(r)

    for fam in sorted(wd_families):
        wd_groups = wd_families[fam]
        if len(wd_groups) <= 1:
            continue
        wd_rows = []
        for wd in sorted(wd_groups):
            group = wd_groups[wd]
            losses = [r["best_val_loss"] for r in group]
            psnrs = [r["psnr_meas"] for r in group if r.get("psnr_meas") is not None]
            best = min(group, key=lambda r: r["best_val_loss"])
            row = {
                "weight_decay": wd,
                "n_runs": len(group),
                "min_loss": min(losses),
                "mean_loss": sum(losses) / len(losses),
            }
            if psnrs:
                row["max_psnr"] = max(psnrs)
                row["mean_psnr"] = sum(psnrs) / len(psnrs)
            wd_rows.append(row)
        wd_cols = ["weight_decay", "n_runs", "min_loss", "mean_loss"]
        if any("max_psnr" in r for r in wd_rows):
            wd_cols += ["max_psnr", "mean_psnr"]
        print_table(wd_rows, wd_cols, f"Weight Decay Analysis: {fam}")


def export_csv(runs: list, path: str):
    completed = [r for r in runs if not r["skipped"]]
    if not completed:
        print("No completed runs to export.")
        return
    cols = ["run", "date", "family", "hidden", "depth", "w0", "k_freq", "k_sigma",
            "weight_decay", "best_val_loss", "psnr_meas", "ssim_meas", "nrmse_meas",
            "psnr_gt", "ssim_gt", "nrmse_gt"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in sorted(completed, key=lambda r: r["best_val_loss"]):
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"Exported {len(completed)} runs to {path}")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Analyze DCE_NIK sweep results")
    parser.add_argument("--runs-dir", default="runs", help="Path to runs directory")
    parser.add_argument("--filter-family", help="Show only this model family")
    parser.add_argument("--filter-date", help="Show only runs from this date (YYYYMMDD)")
    parser.add_argument("--csv", help="Export results to CSV file")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"Error: {runs_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Parse all runs
    runs = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        if args.filter_date and not d.name.startswith(args.filter_date):
            continue
        result = parse_run(d)
        if result:
            runs.append(result)

    if not runs:
        print("No runs found.")
        sys.exit(0)

    if args.csv:
        export_csv(runs, args.csv)

    analyze(runs, filter_family=args.filter_family)


if __name__ == "__main__":
    main()
