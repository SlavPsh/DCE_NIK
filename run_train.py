# run_train.py
import argparse
import numpy as np
import torch

from nik_io import h5_tree, load_event
from nik_model import NIK_SIREN
from nik_train import prepare_tensors, make_sampler, fit_one_scan

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=131072)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--data-device", choices=["cpu","cuda"], default="cpu")
    p.add_argument("--device", choices=["cpu","cuda"], default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--print-h5", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.print_h5:
        h5_tree(args.file, max_items=500)
        return

    event = load_event(args.file, load_images=False)
    k_np, traj_np = event["k"], event["traj"]
    T, S, C, RO = k_np.shape
    print("Loaded:", k_np.shape, traj_np.shape)

    k_t, traj_t, scales, dims = prepare_tensors(k_np, traj_np, data_device=args.data_device)
    sampler = make_sampler(k_t, traj_t, scales, dims, compute_device=args.device)

    model = NIK_SIREN(n_coils=C).to(args.device)
    if args.compile:
        try:
            model = torch.compile(model, mode="max-autotune")
            print("torch.compile enabled")
        except Exception as e:
            print("torch.compile failed:", e)

    model, info = fit_one_scan(
        model, sampler,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        amp=args.amp,
        device=args.device,
        use_tqdm=False
    )
    print("Best loss:", info["best_loss"])

if __name__ == "__main__":
    main()


