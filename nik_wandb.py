"""wandb helper for the trainers: project dce_nik, offline fallback, no-op without wandb.
env: WANDB_MODE (online | offline | disabled), WANDB_ENTITY, WANDB_DIR. summary mirrored to a local json."""
import json, os, time

PROJECT = "dce_nik"


def peak_gpu_mb():
    import torch
    return float(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0


def _num(v):
    if isinstance(v, (bool, int, str)) or v is None: return v
    try: return float(v)
    except Exception: return v


class Run:
    """one training run. log() also updates the local summary, finish() writes it"""

    def __init__(self, name, config, group=None, tags=None, local_json=None, enabled=True):
        self.name, self.local_json, self.t0 = name, local_json, time.time()
        self.summary, self.wb, self.mode = {}, None, "disabled"
        want = os.environ.get("WANDB_MODE", "online")
        if not enabled or want == "disabled":
            return
        try:
            import wandb
        except Exception as e:
            print(f"[wandb] import failed ({e}); local json only", flush=True); return
        kw = dict(project=PROJECT, name=name, group=group, tags=tags, config=config)
        if os.environ.get("WANDB_ENTITY"): kw["entity"] = os.environ["WANDB_ENTITY"]
        if local_json and not os.environ.get("WANDB_DIR"):
            kw["dir"] = os.path.dirname(local_json); os.makedirs(kw["dir"], exist_ok=True)
        try: st = wandb.Settings(init_timeout=60)
        except Exception: st = None
        for mode in dict.fromkeys([want, "offline"]):            # online first, offline fallback
            try:
                self.wb = wandb.init(mode=mode, settings=st, **kw); self.mode = mode; break
            except Exception as e:
                print(f"[wandb] init mode={mode} failed: {str(e).splitlines()[0][:200]}", flush=True)
        print(f"[wandb] {name}: mode {self.mode}", flush=True)

    def log(self, d, step=None):
        d = {k: _num(v) for k, v in d.items()}
        self.summary.update(d)
        if self.wb is not None:
            try: self.wb.log(d, step=step)
            except Exception as e: print(f"[wandb] log failed: {e}", flush=True)

    def finish(self, **final):
        self.summary.update({k: _num(v) for k, v in final.items()})
        self.summary["wall_total_s"] = time.time() - self.t0
        if self.local_json:
            os.makedirs(os.path.dirname(self.local_json), exist_ok=True)
            with open(self.local_json, "w") as f:
                json.dump(dict(name=self.name, mode=self.mode, **self.summary), f, indent=1, default=str)
        if self.wb is not None:
            try:
                for k, v in self.summary.items(): self.wb.summary[k] = v
                self.wb.finish()
            except Exception as e: print(f"[wandb] finish failed: {e}", flush=True)
        self.wb = None
