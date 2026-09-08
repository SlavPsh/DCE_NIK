"""re-execute NIK_vs_CS_combined_report.ipynb in place so it picks up regenerated jsons/figures.
does NOT rebuild cells from build_report_nb.py (that script is stale and no longer emits sections 4-8).
also rewrites the two bits of prose that stated the now-corrected leaked-basis conclusion."""
import json, shutil, sys, os
import nbformat as nbf
from nbclient import NotebookClient

B = "/scratch/rnga/vvpshenov/DCE_NIK"
NB = f"{B}/results/xcat_physical_nomotion_nik_vs_grasp/NIK_vs_CS_combined_report.ipynb"

OLD_FRONTIER = "### spoke-fraction frontier. cs-favored, nuanced. nik haarpsi vs cs flat ~0.71, below cs every fraction. cs-likeness, no gt"
NEW_FRONTIER = ("### spoke-fraction frontier, fair temporal basis\n\n"
                "cs pca basis now from the retained spokes only. previously from all 14 spokes/frame while the "
                "data term used m, which leaked held-out spokes into the cs recon and lifted it at every reduced "
                "fraction. nik trains on the reduced spokes only, so the old comparison was asymmetric.\n\n"
                "fair result: nik and cs level across fractions, nik ahead at 50%. cs drops 0.15-0.18 haarpsi "
                "once the basis is honest. ruler is cs-likeness not accuracy, so this reads conservative for nik.")
OLD_SUM = "| real spoke frontier | cs-favored, no gt |"
NEW_SUM = "| real spoke frontier | level after removing the cs temporal-basis leak; nik ahead at 50%; was wrongly cs-favored |"

def main():
    shutil.copy2(NB, NB + ".leaked_backup")
    nb = nbf.read(NB, as_version=4)
    n = 0
    for c in nb.cells:
        s = c.source
        if OLD_FRONTIER in s:
            c.source = s.replace(OLD_FRONTIER, NEW_FRONTIER); n += 1
        elif OLD_SUM in s:
            c.source = s.replace(OLD_SUM, NEW_SUM); n += 1
        if c.cell_type == "code":
            c.outputs = []; c.execution_count = None
    print(f"prose cells updated: {n} (expect 2)")
    if n != 2:
        print("WARNING: expected 2 prose edits, notebook wording may have drifted")
    os.chdir(B)
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": B}}).execute()
    nbf.write(nb, NB)
    print("REFRESHED", NB)

if __name__ == "__main__":
    main()
