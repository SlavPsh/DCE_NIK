"""re-execute NIK_vs_GRASPv2_combined_report.ipynb in place (all cells, outputs cleared first). no prose edits, no rebuild:
section 10 and the 2026-09-08/10 prose live in the ipynb only, build_gv2_nb.py would regenerate old text.
usage: python refresh_gv2_nb.py [--nb path] [--timeout 1800]"""
import argparse, os, shutil, time
import nbformat as nbf
from nbclient import NotebookClient

B = "/net/beegfs/users/P101440/DCE_NIK"
NB = f"{B}/results/xcat_physical_nomotion_nik_vs_grasp/NIK_vs_GRASPv2_combined_report.ipynb"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nb", default=NB); ap.add_argument("--timeout", type=int, default=1800); a = ap.parse_args()
    bak = a.nb + ".bak_" + time.strftime("%Y%m%d"); shutil.copy2(a.nb, bak); print("backup", bak, flush=True)
    nb = nbf.read(a.nb, as_version=4)
    for c in nb.cells:
        if c.cell_type == "code": c.outputs = []; c.execution_count = None
    os.chdir(B); t0 = time.time()
    NotebookClient(nb, timeout=a.timeout, kernel_name="python3", resources={"metadata": {"path": B}}).execute()
    nbf.write(nb, a.nb)
    err = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.get("output_type") == "error")
    print(f"REFRESHED {a.nb}: {len(nb.cells)} cells, {err} error outputs, {time.time() - t0:.0f} s", flush=True)

if __name__ == "__main__": main()
