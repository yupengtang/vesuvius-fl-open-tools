"""local_tracks.py PREDICTIONS_ZARR_ARRAY TRACKS_DBM Z0 Z1

Surface tracks for one band of a scroll that has no published spiral-dataset tracks (PHerc0846B, 1203, 1218, 1545), made
the way the published ones were: villa's extractor with the parameters recorded in the published <name>.extract.json
(downsample 2, horizontal ribbons 2 deep every 8 slices, vertical slabs 4 thick every 16 voxels, dust 400, max area
640000, 'interjoint' paths). extract_surface_tracks.py, surface_tracks_gpu.py and fast_tracks.py next to this file are
villa main baf5e605 (spiral-fitting/), run here with the GPU (Brook) backend; the lane's old fitter copy only has the
Kimimaro CPU path, which is far slower on a full band.

Only the DBM is written here. erwin_data.sh then builds the packed store and the crossings cache with the fitter's own
tracks.py, so the fitter accepts them (store version 1 and the DBM signature check are the same in both copies).
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_surface_tracks as est  # noqa: E402

published = dict(downsample_factor=2, z_chunk_depth_h=2, z_chunk_stride_h=8, yx_chunk_thickness_v=4, yx_stride_v=16,
                 dust_threshold=400, max_area_threshold=640_000, path_mode='interjoint')
for k, v in published.items():   # module level, so worker processes that re-import this file see them too
    setattr(est, k, v)


def open_zarr_array(path):
    # only reached when the direct reader does not apply (it decodes local uint8 blosc zarr v2 chunks itself)
    import zarr
    return zarr.open_array(path, mode='r')


est.open_predictions = open_zarr_array


def main():
    pred, dbm_path, z0, z1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    est.z_min, est.z_max = z0, z1
    backend = est.select_backend(os.environ.get('LOCAL_TRACKS_BACKEND', 'gpu'))   # cpu (Kimimaro) only for checks off a GPU
    Path(dbm_path).parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    (est.run_gpu if backend == 'gpu' else est.run_cpu)(pred, dbm_path)
    print(f'tracks z {z0}-{z1} extracted in {time.time() - t:.0f} s', flush=True)


# the GPU backend spawns its workers and Kimimaro forks a pool (forkserver by default on Python 3.14): both re-import this
# file, so the run itself must stay behind the main guard
if __name__ == '__main__':
    main()
