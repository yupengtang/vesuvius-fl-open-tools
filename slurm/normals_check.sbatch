#!/bin/bash
#SBATCH -J fl-nrm
#SBATCH -p ice-cpu        # PACE-ICE name; adapt to your cluster
#SBATCH -N 1 --ntasks-per-node=1 -c 16
#SBATCH --mem=96G
#SBATCH -t 2:00:00
#SBATCH -o fl_nrm_%j.log
# usage: sbatch normals_check.sbatch SCROLL Z0 Z1 [make_normals args]
set -uo pipefail
K=${FL_BASE:?set FL_BASE}/fl/kit; T=/tmp/$USER/nrm_$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT
export UV_CACHE_DIR=$T/uvc UV_PYTHON_INSTALL_DIR=$T/uvpy TMPDIR=$T PATH=$HOME/.local/bin:$PATH
uv venv -q -p 3.12 $T/py && uv pip install -q -p $T/py/bin/python numpy numcodecs requests torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple || exit 1
mkdir -p $T/k/vw/fl $T/k/vw/x_recipe_erwin/tools && cp $K/normals_check.py $T/k/ && cp $K/vw/fl/scrolls.json $T/k/vw/fl/ && cp $K/vw/x_recipe_erwin/tools/make_normals.py $T/k/vw/x_recipe_erwin/tools/
$T/py/bin/python -I $T/k/normals_check.py "$1" "$2" "$3" $T/work "${@:4}"
