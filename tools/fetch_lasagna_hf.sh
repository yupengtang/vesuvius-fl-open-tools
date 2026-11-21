#!/bin/bash
# fetch_lasagna_hf.sh SCROLL Z0 Z1 OUT_DIR : Lasagna nx, ny and grad_mag (group 2) for one band, from SharpMeow's whole-scan
# fields on Hugging Face (datasets SharpMeow/<scroll>-lasagna: villa lasagna predict3d with scrollprize/lasagna de45f011 on
# the listed volume, the bucket's layout except that group 2 chunks are 64, not 32). They cover the scrolls the bucket has
# no Lasagna for (0175A/B, 0483A/B, 0490A/B, 0846A/B). The parts are 1024-slice tiles: only the tiles that meet
# [Z0-256, Z1+256) are fetched (sha256 from the dataset's index.json), and only the three channels are kept.
# Writes OUT_DIR/las_008_{nx,ny,grad_mag}.ome.zarr. Exit 3: no fields published for SCROLL.
set -euo pipefail
S=$1 Z0=$2 Z1=$3 OUT=$4
HFL=https://huggingface.co/datasets/SharpMeow/$(echo $S | tr A-Z a-z)-lasagna/resolve/main
TMP=$OUT/.hf_parts; rm -rf $TMP; mkdir -p $TMP
curl -sfL --retry 3 -o $TMP/index.json $HFL/index.json || { rm -rf $TMP; exit 3; }
python3 -c "
import json
for p in json.load(open('$TMP/index.json'))['parts']:
    if p['crop_zyx'][0][0] < $Z1 + 256 and p['crop_zyx'][0][1] > $Z0 - 256:
        print(p['tar'], p['sha256'])" > $TMP/tiles.txt
# one tile at a time ran at ~4 MB/s (PHerc0846B band: 6 tiles in 1122 s), so the tiles download side by side
while read t sha; do curl -sfL --retry 5 -C - -o $TMP/$t $HFL/parts/$t & done < $TMP/tiles.txt; wait
while read t sha; do
  echo "$sha  $TMP/$t" | sha256sum -c --quiet
  tar -x -C $TMP -f $TMP/$t --wildcards "${S}_nx.ome.zarr/*" "${S}_ny.ome.zarr/*" "${S}_grad_mag.ome.zarr/*"
  rm $TMP/$t
done < $TMP/tiles.txt
for ch in nx ny grad_mag; do rm -rf $OUT/las_008_$ch.ome.zarr; mv $TMP/${S}_$ch.ome.zarr $OUT/las_008_$ch.ome.zarr; done
echo "Lasagna for $S z $Z0-$Z1: community fields (SharpMeow on Hugging Face), $(wc -l < $TMP/tiles.txt) tiles, $(du -sh $OUT | cut -f1)"
rm -rf $TMP
