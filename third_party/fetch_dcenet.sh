#!/bin/bash
# fetch DCENET at the pinned SHA. upstream source is NOT modified; adapters live in DCE_NIK/dcenet_adapter.py
set -euo pipefail
cd "$(dirname "$0")"
. <(grep -E '^(url|sha)=' DCENET.lock)
[ -d DCENET/.git ] || git clone -q "$url" DCENET
cd DCENET && git fetch -q && git checkout -q "$sha" && echo "DCENET at $(git rev-parse HEAD)"
