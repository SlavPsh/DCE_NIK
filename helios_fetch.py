#!/usr/bin/env python
"""laptop side of the helios loop: git pull the three repos, jobs/done table, wandb api summaries into results/_wandb/.
usage: python helios_fetch.py [--no-git] [--no-wandb] [--since 2026-09-01] [--history] [--entity E] [--limit 30]
wandb needs `wandb login` once on the laptop (or WANDB_API_KEY); without it only the git part runs."""
import argparse, csv, json, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPOS = ["DCE_NIK", "grasp_v2", "grasp_pro_py"]
OUT = os.path.join(HERE, "results", "_wandb")
SUMMARY_KEYS = ["loss", "val_knmse", "best_val_knmse", "heldout_mse", "best_heldout_mse", "best_step",
                "wall_s", "wall_total_s", "peak_gpu_mb", "params"]
CONFIG_KEYS = ["model", "sim", "slice", "steps", "hidden_width", "hidden", "depth", "k_sigma", "t_sigma",
               "rank", "seed", "save_dir", "spoke_keep_file"]


def sh(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def git_pull():
    for r in REPOS:
        d = os.path.join(ROOT, r)
        if not os.path.isdir(os.path.join(d, ".git")): print(f"{r:14s} no repo at {d}"); continue
        _, old = sh(["git", "rev-parse", "HEAD"], d)
        rc, out = sh(["git", "pull", "--ff-only", "-q", "origin"], d)
        _, new = sh(["git", "rev-parse", "HEAD"], d)
        if rc: print(f"{r:14s} pull FAILED: {out}"); continue
        if old == new: print(f"{r:14s} up to date ({new[:7]})"); continue
        _, lg = sh(["git", "log", "--oneline", f"{old}..{new}"], d)
        print(f"{r:14s} {old[:7]} -> {new[:7]}, {len(lg.splitlines())} commits\n  " + lg.replace("\n", "\n  "))


def show_jobs():
    done = os.path.join(HERE, "jobs", "done")
    names = sorted(n for n in os.listdir(done)) if os.path.isdir(done) else []
    names = [n for n in names if not n.startswith(".")]
    if names:
        print("\njobs/done:")
        for n in names:
            kv = {}
            with open(os.path.join(done, n), errors="replace") as f:
                for line in f:
                    k, _, v = line.rstrip("\n").partition(" "); kv[k] = v
            state = kv.get("state") or (f"exit {kv['exit']}" if "exit" in kv else "")
            print(f"  {n:34s} {kv.get('jid', ''):>9s}  {state:30s} {kv.get('submitted', kv.get('start', ''))}")
    st = os.path.join(HERE, "jobs", "agent", "status.md")
    if os.path.exists(st):
        with open(st) as f: lines = f.read().splitlines()
        if len(lines) > 2: print("  agent: " + lines[2])


def fetch_wandb(a):
    try:
        import wandb
        api = wandb.Api(timeout=60)
    except Exception as e:
        print(f"\nwandb api unavailable: {str(e).splitlines()[0][:160]}\n  fix: wandb login   (or set WANDB_API_KEY)"); return
    entity = a.entity or os.environ.get("WANDB_ENTITY") or api.default_entity
    path = f"{entity}/{a.project}" if entity else a.project
    filt = {"createdAt": {"$gte": a.since}} if a.since else None
    try:
        runs = list(api.runs(path, filters=filt, order="-created_at"))
    except Exception as e:
        print(f"\nwandb query {path} failed: {str(e).splitlines()[0][:160]}"); return
    os.makedirs(OUT, exist_ok=True); rows = []
    for r in runs:
        try: s = dict(r.summary)
        except Exception: s = getattr(r.summary, "_json_dict", {})
        rows.append(dict(id=r.id, name=r.name, group=r.group, tags=list(r.tags), state=r.state, created=str(r.created_at),
                         url=r.url, summary={k: v for k, v in s.items() if not k.startswith("_")},
                         config={k: v for k, v in r.config.items() if not k.startswith("_")}))
    with open(os.path.join(OUT, "runs.json"), "w") as f: json.dump(rows, f, indent=1, default=str)
    cols = ["name", "group", "state", "created"] + SUMMARY_KEYS + CONFIG_KEYS
    with open(os.path.join(OUT, "runs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r.get(k) if k in r else r["summary"].get(k) if k in SUMMARY_KEYS else r["config"].get(k) for k in cols])
    print(f"\nwandb {path}: {len(rows)} runs -> {os.path.relpath(OUT, HERE)}/runs.json, runs.csv")

    def fmt(v):
        if v is None: return ""
        if isinstance(v, float): return f"{v:.0f}" if abs(v) >= 100 else f"{v:.3e}"
        return str(v)
    print(f"  {'name':36s} {'state':9s} {'val_knmse':>10s} {'best':>10s} {'best_step':>9s} {'wall_s':>8s} {'gpu_mb':>7s}")
    for r in rows[:a.limit]:
        s = r["summary"]; best = s.get("best_val_knmse", s.get("best_heldout_mse"))
        print(f"  {r['name'][:36]:36s} {r['state']:9s} {fmt(s.get('val_knmse')):>10s} {fmt(best):>10s} "
              f"{fmt(s.get('best_step')):>9s} {fmt(s.get('wall_s')):>8s} {fmt(s.get('peak_gpu_mb')):>7s}")
    if a.history:
        hd = os.path.join(OUT, "history"); os.makedirs(hd, exist_ok=True)
        for r, run in zip(rows, runs):
            keys = ["_step"] + [k for k in SUMMARY_KEYS if k in r["summary"]]
            try: hist = run.history(keys=keys, pandas=False, samples=a.samples)
            except Exception as e: print(f"  history {r['name']} failed: {e}"); continue
            with open(os.path.join(hd, f"{r['name']}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(hist)
        print(f"  history csv for {len(rows)} runs -> {os.path.relpath(hd, HERE)}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-git", action="store_true"); ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--since", default=None, help="iso date, wandb createdAt >=")
    ap.add_argument("--history", action="store_true", help="per-run history csv")
    ap.add_argument("--samples", type=int, default=500); ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--entity", default=None); ap.add_argument("--project", default="dce_nik")
    a = ap.parse_args()
    if not a.no_git: git_pull()
    show_jobs()
    if not a.no_wandb: fetch_wandb(a)


if __name__ == "__main__": main()
