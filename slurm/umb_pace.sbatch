#!/bin/bash
#SBATCH -J fl-umb
#SBATCH -p ice-cpu        # PACE-ICE name; adapt to your cluster
#SBATCH -N 1 --ntasks-per-node=1 -c 16
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o fl_umb_%j.log
# Scroll axes (umbilicus) for the sweep scrolls with the kit's build_umb.py, on node-local /tmp.
# usage: sbatch umb_pace.sbatch <scroll> [<scroll> ...]
set -uo pipefail
B=${FL_BASE:?set FL_BASE to a shared directory that holds fl/kit}; KIT=$B/fl/kit; OUT=$KIT/vw/x_recipe_erwin/umb
T=/tmp/$USER/umb_$SLURM_JOB_ID; W=$T/vw; mkdir -p $T
trap 'rm -rf $T' EXIT
export UV_CACHE_DIR=$T/uvcache UV_PYTHON_INSTALL_DIR=$T/uvpy TMPDIR=$T/tmp PATH=$HOME/.local/bin:$T/py/bin:$PATH; mkdir -p $TMPDIR
mkdir -p $W/fl $W/x_recipe_erwin/tools $W/vc_build_main/bin
cp $KIT/vw/fl/scrolls.json $W/fl/; cp $KIT/vw/x_recipe_erwin/tools/build_umb.py $W/x_recipe_erwin/tools/
sed -i "s#/home/liam/projects/vesuvius-work#$W#g" $W/x_recipe_erwin/tools/build_umb.py
uv venv -q -p 3.12 $T/py && uv pip install -q -p $T/py/bin/python numpy scipy awscli || exit 1
cd $T && curl -fsSL -o VC3D.AppImage "$(curl -fsSL https://api.github.com/repos/ScrollPrize/villa/releases/tags/latest \
  | python3 -c "import json,sys;print(next(a['browser_download_url'] for a in json.load(sys.stdin)['assets'] if a['name'].endswith('linux-x86_64.AppImage')))")" \
  && chmod +x VC3D.AppImage && ./VC3D.AppImage --appimage-extract >/dev/null || exit 1
printf '#!/bin/bash\nexec %s/squashfs-root/AppRun vc_gen_umbilicus "$@"\n' $T > $W/vc_build_main/bin/vc_gen_umbilicus; chmod +x $W/vc_build_main/bin/vc_gen_umbilicus
$W/vc_build_main/bin/vc_gen_umbilicus --help 2>&1 | head -3
for S in "$@"; do
  Z1=$(python3 -c "import json;print(json.load(open('$W/fl/scrolls.json'))['$S']['shape'][0]-1)")
  # same sampling as umb_PHerc0306B.json: every 100 slices over the scroll, the edges trimmed by the tool's outlier pass
  [ -s $OUT/umb_$S.json ] && { echo "$S exists"; continue; }
  cd $W/x_recipe_erwin/tools && $T/py/bin/python build_umb.py $S 0 $Z1 100 $T/umb_$S.json \
    && cp $T/umb_$S.json $OUT/ || echo "$S FAILED"
  cd $T; rm -rf $W/x_recipe_erwin/ngrid_$S
done
ls -la $OUT
