#!/bin/bash
# fetch_villa_extractor.sh <kit>/x_recipe_erwin : put villa's surface-track extractor (spiral-fitting, MIT) and
# local_tracks.py into <kit>/x_recipe_erwin/tracks_gpu/, which the patched erwin_data.sh runs for TRACKS=local.
# Pinned to the villa commit the extractor was checked at.
set -euo pipefail
D=${1:?usage: fetch_villa_extractor.sh <kit>/x_recipe_erwin}/tracks_gpu
C=baf5e605bb42a3d55a62b7c5c728e0908137225c
mkdir -p "$D"
cp "$(dirname "$0")/local_tracks.py" "$D/"
for f in extract_surface_tracks.py surface_tracks_gpu.py fast_tracks.py; do
  curl -fsSL -o "$D/$f" "https://raw.githubusercontent.com/ScrollPrize/villa/$C/spiral-fitting/$f"
done
ls "$D"
