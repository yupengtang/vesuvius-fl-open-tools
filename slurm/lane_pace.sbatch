#!/bin/bash
#SBATCH -J fl-lane
#SBATCH -p ice-gpu        # PACE-ICE names; adapt partition / -C features to your cluster
#SBATCH -N 1 --ntasks-per-node=1 -c 16
#SBATCH --gres=gpu:1 -C "gpu-h100|gpu-h200|gpu-l40s"
#SBATCH --mem=192G
#SBATCH -t 16:00:00
#SBATCH -o fl_lane_%j.log
# One First Letters screening band end to end (the PHerc. 0306B team's lane_box.sh) on one PACE GPU node.
# Everything runs in node-local /tmp; only the results tar and logs come back to shared storage.
# usage: sbatch lane_pace.sbatch <lane id> <scroll> <z0> <z1> <um> <CW|ACW|AB> <umbilicus json (kit-relative)> [wcenter] [maxw]
set -uo pipefail
ID=$1 SCROLL=$2 Z0=$3 Z1=$4 UM=$5 SENSE=$6 UMBREL=$7 WC=${8:-} MAXW=${9:-40}
B=${FL_BASE:?set FL_BASE to a shared directory that holds fl/kit}; KIT=$B/fl/kit; OUT=$B/fl/lanes/$ID; mkdir -p $OUT
T=/tmp/$USER/lane_$SLURM_JOB_ID; W=$T/vw; V=$T/villa; mkdir -p $T
trap 'cp $T/lane/logs/*.log $W/x_recipe_erwin/native/*.log $T/lane/ds/*/local_tracks.log $T/lane/timing.tsv $T/lane/FAILED $T/lane/sense.txt $OUT/ 2>/dev/null; rm -rf $T' EXIT
echo "node $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) tmp $(df -h /tmp | tail -1 | awk '{print $4}') free"
python3 - <<'PY' || exit 1
import subprocess; subprocess.run(["nvidia-smi", "-L"], check=True)
PY

# The kit's scripts hard-code the original authors' paths; point them at this job's /tmp tree.
cp -r $KIT/vw $W
grep -rlE "/home/liam/projects|/opt/squashfs-root|cd /opt" $W --include=*.sh --include=*.py --include=*.json | xargs sed -i \
  -e "s#/home/liam/projects/vesuvius-work#$W#g" -e "s#/home/liam/projects/villa#$V#g" \
  -e "s#/opt/squashfs-root#$T/squashfs-root#g" -e "s#cd /opt \&\&#cd $T \&\&#g" -e "s#/opt/VC3D.AppImage#$T/VC3D.AppImage#g"

# uv, its cache and its Pythons on /tmp (home is near its quota); gcc 12 for the C++23 native modules
# (the gcc 13 and 14 installs have no cc1plus on compute nodes); ninja from pip.
export UV_CACHE_DIR=$T/uvcache UV_PYTHON_INSTALL_DIR=$T/uvpy XDG_CACHE_HOME=$T/xdg HF_HOME=$T/hf TMPDIR=$T/tmp
export TRITON_CACHE_DIR=$T/triton TORCHINDUCTOR_CACHE_DIR=$T/inductor; mkdir -p $TMPDIR
# FL_GCC: a gcc 12 prefix (on PACE-ICE /usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/
# gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz); leave it unset where the default compiler builds C++23.
[ -n "${FL_GCC:-}" ] && export CXX=$FL_GCC/bin/g++ LD_LIBRARY_PATH=$FL_GCC/lib64:${LD_LIBRARY_PATH:-}
export PATH=$HOME/.local/bin:$T/bin:$PATH; mkdir -p $T/bin
uv venv -q -p 3.12 $T/tools && uv pip install -q -p $T/tools/bin/python ninja && ln -sf $T/tools/bin/ninja $T/bin/ninja

# Scrolls with no published spiral-dataset tracks (PHerc0846B, 1203, 1218, 1545): erwin_data.sh extracts this band's tracks
# on this GPU with the published tracks' parameters (x_recipe_erwin/tracks_gpu).
VID=$(python3 -c "import json;print(json.load(open('$KIT/vw/fl/scrolls.json'))['$SCROLL']['volume_url'].rstrip('/').split('/')[-1].split('-')[0])")
TRACKS=published; curl -sf -o /dev/null https://dl.ash2txt.org/datasets/spiral_datasets/$SCROLL/$VID/tracks/ || TRACKS=local
cat > $T/lane.env <<EOF
SCROLL=$SCROLL
Z0=$Z0
Z1=$Z1
TB=$ID
UM=$UM
SENSE=$SENSE
TRACKS=$TRACKS
UMB=$W/$UMBREL
MAXW=$MAXW
WCENTER=$WC
CACHE_GB=60
EOF
cat $T/lane.env
LANE=$T/lane LANE_ENV=$T/lane.env THREADS=$SLURM_CPUS_PER_TASK bash $W/x_vast_plan/lane_box.sh
rc=$?
echo "lane_box rc=$rc"; cat $T/lane/timing.tsv 2>/dev/null; cat $T/lane/FAILED 2>/dev/null
for f in setup erwin_setup native data; do echo "== $f"; tail -15 $T/lane/logs/$f.log 2>/dev/null; done
[ -f $T/lane/results.tar ] && cp $T/lane/results.tar $OUT/ && du -sh $OUT/results.tar

# Review, which the published lane_box.sh leaves to a later local step: the glyph CNN (x_glyph gA.pt) per window and the
# single-line finder on the fused ink maps. Those maps exist only in node /tmp, so run both here and keep their JSON.
if [ -f $T/lane/best_tag ] && [ -d $T/lane/rv2 ]; then
  RV=$W/x_vast_plan/results/$ID; mkdir -p $RV && ln -sfn $T/lane/results $RV/results && ln -sfn $T/lane/rv2 $RV/maps
  VPY=$W/envs/vcpy/bin/python
  $VPY $W/fl/scripts/glyph_win.py $T/lane/rv2 --tag $(cat $T/lane/best_tag) --um $UM > $T/lane/logs/glyph.log 2>&1
  $VPY $W/fl/scripts/line_find.py --lane $ID --faces both --out $OUT/line_find.json > $T/lane/logs/line_find.log 2>&1
  mkdir -p $OUT/glyph; for d in $T/lane/rv2/*/; do [ -f $d/glyph_win.json ] && cp $d/glyph_win.json $OUT/glyph/$(basename $d).json; done
  echo "review: $(ls $OUT/glyph | wc -l) windings glyph-scored"; tail -3 $T/lane/logs/glyph.log $T/lane/logs/line_find.log
fi
exit $rc
