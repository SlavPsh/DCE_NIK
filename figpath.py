"""timestamped figure paths, figures/<yyyymmdd_hhmm>_<name>.png"""
import os
import time

FIGDIR = "/scratch/rnga/vvpshenov/DCE_NIK/figures"


def fig(name, figdir=FIGDIR):
    base, ext = os.path.splitext(str(name))
    os.makedirs(figdir, exist_ok=True)
    return os.path.join(figdir, f"{time.strftime('%Y%m%d_%H%M')}_{base}{ext or '.png'}")
