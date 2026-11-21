# Running the PHerc. 0306B First Letters lane on any scroll

The PHerc. 0306B team (Liam Warner and Nicola Bodill) published the screening lane that found their line:
[SirProdigle/pherc0306b-first-letters](https://github.com/SirProdigle/pherc0306b-first-letters). We tried to run it
on the other eligible scrolls and found that, as published, it only runs on scrolls that have a hosted Lasagna
prediction, and even there it needs a module that is not in the kit. This repo fills those gaps, with a check for
each piece, and adds what we needed to run it on a Slurm cluster.

The step after the lane (reading orientation, the crack test, claim imaging, a reading benchmark and a screening
atlas of 57 lanes) is in the companion repository
[yupengtang/vesuvius-fl-followup](https://github.com/yupengtang/vesuvius-fl-followup).

Nothing here changes their method. The fit, snapping, flattening, inking and triage are theirs (and Erwin
Nieuwlaar's PHerc. 343 recipe underneath).

## What the published kit is missing

| missing piece | effect | here |
|---|---|---|
| `vc3d_fiber_format` (villa `vesuvius/src`, needed by the old spiral fitter in `x_recipe_erwin/spiral_old`) | `erwin_setup.sh` stops with `No module named 'vc3d_fiber_format'` | copy it from villa commit `0fb38c45` into `x_recipe_erwin/spiral_old/vc3d_fiber_format/`; it is stdlib only |
| `x_recipe_erwin/tools/make_normals.py` (synthetic normals from m7, called by `erwin_data.sh` when a scroll has no Lasagna) | the data stage finishes without normals and the fit dies on a missing `las_008_nx.ome.zarr` | `tools/make_normals.py`, written from scratch and checked against real Lasagna normals (below) |
| `x_recipe_erwin/tools/sense_geo.py` (used for `SENSE=AB`) | the lane falls back to its weaker sheet-jump rule | `tools/sense_from_catalog.py`: the sense straight from the open-data catalog, so most scrolls need no AB run |
| the VC3D AppImage named in `fl/remote/setup.sh` | that URL is gone, and the fallback takes the newest build: the one it fetched on 3 November refused the lane's normal grids, the one it fetches on 14 November no longer takes the `--prefetch-remote` option of the claim jobs | `patches/setup_pin_vc3d.patch` pins the archive build VC3D c9f8c57 (2 November 2026) that every lane here ran on (`patch -p1 -d FL_BASE/fl/kit/vw < patches/setup_pin_vc3d.patch`); a copy of the AppImage in `FL_BASE/fl/cache` is used before any download |

9 of the 21 eligible scrolls have no hosted Lasagna (0175A, 0175B, 0306B, 0483A, 0483B, 0490A, 0490B, 0846A,
0846B), so without `make_normals.py` the lane cannot screen them at all. PHerc. 0306B is one of them, which is
why we could not reproduce their own band and used PHerc. 343 as the positive control instead.

Since then SharpMeow has published whole-scan Lasagna fields for eight of them on Hugging Face (datasets
`SharpMeow/<scroll>-lasagna`: 0175A/B, 0483A/B, 0490A/B, 0846A/B, villa `predict3d` on the listed volumes).
`tools/fetch_lasagna_hf.sh SCROLL Z0 Z1 OUT_DIR` fetches only the 1024-slice tiles that meet a band (sha256 from the
dataset's `index.json`) and keeps nx, ny and grad_mag in the layout `erwin_data.sh` expects (their group 2 chunks are 64,
not 32; the fitter's pool packer takes either). On the PHerc. 0846B band z 11500-13000 that is 6 tiles, 2.8 GB. Where
these exist, use them rather than the synthetic normals below.

## Synthetic normals (`tools/make_normals.py`)

The fitter reads Lasagna normals as two uint8 volumes on the 4x grid, decoded as `(u8 - 128) / 127` with
`nz = +sqrt(1 - nx^2 - ny^2)`, 0 meaning no data, and only uses `|n . n_model|`. `make_normals.py` mean-pools the
m7 surface prediction (level 1, as `erwin_data.sh` downloads it) onto that grid, takes the principal eigenvector of
the structure tensor, canonicalises it to `nz >= 0` and writes the same layout (`las_008_nx.ome.zarr/2`,
`las_008_ny.ome.zarr/2`, zarr v2, chunks 32). It runs on the GPU in the data stage, a few minutes per band.

Check (`tools/normals_check.py`): on PHerc. 343 z 10000-10256, which has real Lasagna normals, the angle between
the two normals where both exist:

| gradient sigma / tensor sigma (4x voxels) | median | within 10 deg | within 20 deg | p90 |
|---|---|---|---|---|
| 1 / 2.5 | 7.8 | 59% | 78.5% | 40 |
| **1 / 4 (default)** | **7.7** | **60%** | **81%** | **35** |
| 1 / 6 | 8.6 | 56% | 79% | 35.5 |
| 1.5 / 4 | 9.7 | 51% | 74% | 45.5 |
| 2 / 6 | 12.6 | 41% | 67% | 54 |

Coverage is close to Lasagna's (in that band 135M voxels have both, 5.6M Lasagna only, 33M synthetic only).

## Spiral sense from the catalog (`tools/sense_from_catalog.py`)

villa's spiral fitter derives the outward sense from two catalog properties of the volume (ACW when
`z_direction_is_top_to_bottom` differs from `left_handed_coordinates`, see
`spiral-fitting/surface_orientation.py`). The script reads them from the bucket's `metadata.json`. It gives ACW for
PHerc. 0306B and CW for PHerc. 343, the senses both winning teams used. For the volumes the lane uses:

| scroll | volume | um | sense |
|---|---|---|---|
| PHerc0125 | 20250821151825 | 9.362 | unknown (run AB) |
| PHerc0175A | 20250521115057 | 8.64 | ACW |
| PHerc0175B | 20250521125822 | 8.64 | ACW |
| PHerc0191 | 20250821151635 | 9.362 | CW |
| PHerc0211 | 20250821151803 | 9.362 | ACW |
| PHerc0257 | 20250821151750 | 9.362 | CW |
| PHerc0268 | 20251110183117 | 8.64 | CW |
| PHerc0306B | 20250521133212 | 8.64 | ACW |
| PHerc0358 | 20250821151737 | 9.362 | ACW |
| PHerc0483A | 20250521140913 | 8.64 | ACW |
| PHerc0483B | 20251124083638 | 8.64 | CW |
| PHerc0490A | 20250521151210 | 8.64 | unknown (run AB) |
| PHerc0490B | 20250521151215 | 8.64 | unknown (run AB) |
| PHerc0800 | 20250521135224 | 8.64 | CW |
| PHerc0813 | 20250821151723 | 9.362 | CW |
| PHerc0826 | 20250821151701 | 9.362 | ACW |
| PHerc0846A | 20250728152254 | 9.362 | unknown (run AB) |
| PHerc0846B | 20250804142305 | 9.362 | CW |
| PHerc1203 | 20250820131727 | 9.362 | unknown (run AB) |
| PHerc1218 | 20250521120456 | 8.64 | CW |
| PHerc1545 | 20250821151648 | 9.362 | ACW |

## Scroll axes for all 21 eligible scrolls (`umbilicus/`)

Built with the kit's own `x_recipe_erwin/tools/build_umb.py` (normal grids every 100 slices, `vc_gen_umbilicus`,
outlier drop, Savitzky-Golay), on CPU in about 4 minutes per scroll. `umb_PHerc0306B.json` is the team's own file.
The axis step is how far the axis moves per 100 slices; the 0306B axis the lane ran on has a median of 43 and a
p95 of 82, so bands where the local step stays near that are the safest to screen.

| scroll | um | points | kept/total | z range | median / p95 axis step (vox per 100 slices) |
|---|---|---|---|---|---|
| PHerc0125 | 9.362 | 182 | 182/205 | 2000-20500 | 30 / 84 |
| PHerc0175A | 8.64 | 97 | 97/128 | 1700-12700 | 36 / 171 |
| PHerc0175B | 8.64 | 120 | 120/159 | 1500-14400 | 83 / 226 |
| PHerc0191 | 9.362 | 150 | 150/186 | 1700-18700 | 68 / 157 |
| PHerc0211 | 9.362 | 166 | 166/191 | 1700-19400 | 50 / 166 |
| PHerc0257 | 9.362 | 156 | 156/185 | 1400-18500 | 44 / 139 |
| PHerc0268 | 8.64 | 84 | 84/149 | 100-12600 | 107 / 344 |
| PHerc0306B | 8.64 | 124 | 124/148 | 1100-14400 | 43 / 82 |
| PHerc0358 | 9.362 | 117 | 117/148 | 0-14700 | 69 / 292 |
| PHerc0483A | 8.64 | 118 | 118/159 | 200-15800 | 56 / 178 |
| PHerc0483B | 8.64 | 81 | 81/113 | 900-11800 | 70 / 145 |
| PHerc0490A | 8.64 | 70 | 70/117 | 900-11100 | 81 / 244 |
| PHerc0490B | 8.64 | 54 | 54/107 | 200-9400 | 64 / 202 |
| PHerc0800 | 8.64 | 192 | 192/243 | 1000-24200 | 54 / 184 |
| PHerc0813 | 9.362 | 132 | 132/158 | 1200-16300 | 51 / 181 |
| PHerc0826 | 9.362 | 129 | 129/170 | 1400-16600 | 65 / 165 |
| PHerc0846A | 9.362 | 97 | 97/141 | 300-13700 | 48 / 131 |
| PHerc0846B | 9.362 | 82 | 82/136 | 400-12800 | 54 / 278 |
| PHerc1203 | 9.362 | 166 | 166/186 | 1700-18400 | 40 / 82 |
| PHerc1218 | 8.64 | 185 | 185/233 | 800-23100 | 32 / 90 |
| PHerc1545 | 9.362 | 186 | 186/206 | 400-20900 | 58 / 178 |

## Tracks for the scrolls that have none (`tracks/`)

The fitter's data stage downloads surface tracks from `spiral_datasets/<scroll>/<volume>/tracks/`. Four eligible
scrolls have none there (PHerc. 0846B, 1203, 1218 and 1545), so the published lane stops at the data stage on them.
`tracks/local_tracks.py` makes them for one band the way the published ones were made: villa's extractor
(`spiral-fitting/extract_surface_tracks.py`, `surface_tracks_gpu.py`, `fast_tracks.py`, fetched at a pinned commit by
`tracks/fetch_villa_extractor.sh`) with the parameters recorded in the published stores' `extract.json` (downsample 2,
horizontal ribbons 2 deep every 8 slices, vertical slabs 4 thick every 16 voxels, dust 400, max area 640000,
`interjoint` paths). It writes only the DBM; `erwin_data.sh` (`patches/erwin_data_local_tracks.patch`, `TRACKS=local`)
then builds the packed store and the crossings with the fitter's own `tracks.py`, so the fitter takes them as it takes
the published ones.

Check (`tracks/tracks_compare.py`): on PHerc. 343 z 10004-10404 against the published store, on CPU (Kimimaro, 53 min
on 8 cores):

| family | ours | published | ratio |
|---|---|---|---|
| horizontal ribbons, all 48 between z 10012 and 10396 | 268,409 | 268,409 | 1.000, ribbon by ribbon |
| vertical (zy slabs) inside z 10068-10340 | 273,503 (mean 20.2 points) | 209,852 (mean 20.4 points) | 1.30 |

The horizontal tracks match exactly. We get 30% more vertical tracks of the same length; we have not found which
setting accounts for it, and the fitter used them without trouble. In a lane on PHerc. 1545 (z 2300-3800) the GPU path
made 2.34 M tracks inside the data stage (6 min for the whole stage), and the fit converged as on the scrolls with
published tracks.

Note for anyone running it: the extractor's GPU backend spawns worker processes and Kimimaro forks a pool, and
Python 3.14 starts both by re-importing the main script, so the run has to sit under `if __name__ == '__main__'`
(our first version looped on bootstrap errors for two hours).

## The review step: rebuilding the glyph CNN (`glyph/`)

The lane's review (`fl/scripts/glyph_win.py` and the single-line finder `fl/scripts/line_find.py`) scores windows
with a small CNN, `x_glyph/models/gA.pt`, the model that ranked the 0306B line first of 72 windows. The kit ships the
weights but not `x_glyph/train.py`, which defines `GlyphNet` and `load_model`, nor `x_glyph/synth.py`, which `data.py`
imports. Both review scripts therefore stop at the import, and nothing else in the lane says so: the triage and the
cards still come out. All our first screens ran without it.

`glyph/train.py` rebuilds the network. The state dict fixes most of it: four blocks of two 3x3 convolutions without
bias, each followed by BatchNorm (24, 48, 96 and 192 channels, two input channels for ink and mask), and a head
Linear(384, 64) then Linear(64, 1), so the head reads two pooled copies of the last block. The shapes leave open the
activation, the pooling between blocks and the head's pooling. For the first two we used the BatchNorm running
statistics as a fingerprint (`glyph/fingerprint.py`, `glyph/deep_fp.py fp`): push 640 real lane windows through each
candidate wiring and compare the channel means arriving at each of the 8 BatchNorm layers with the means stored in
them.

| wiring | summed mean gap over the 8 BatchNorms (in running sd) | lowest channel correlation |
|---|---|---|
| **ReLU, 2x2 max-pool** | **1.95** | **0.88** |
| leaky ReLU (0.1), max-pool | 2.29 | 0.87 |
| ReLU, average pool | 2.54 | 0.74 |
| SiLU or GELU, either pool | 3.9 to 4.2 | 0.36 to 0.55 |
| ELU, either pool | 12.2 to 12.7 | below 0 |

For the head, the order of the two pooled halves decides everything: with (max, mean) every window scores below -4,
with (mean, max) the published scale comes back (`glyph/deep_fp.py scale`). On our PHerc. 343 control lane the
windows over Erwin's text score 8.8 to 9.9 against 72 windows from six bands without text (p90 4.45, max 7.07); the
team's notes give 10.8 for 0343 text and 9.06 for the top 1 % of text-free windows on their own maps. `glyph/synth.py`
is a stub (only training calls it) and `glyph/evaluate_cpu.patch` lets `score_windows` run without a GPU.

The lane deletes its fused z maps with the node's `/tmp`, but `results.tar` keeps the overlay JPGs, and fastpipe
draws them so that the ink term can be read back: `a = clip((z - 0.9) / 1.8, 0, 1) = (R - B) / (0.6 x 255)`, at 1/6 of
the voxel size once packed, right next to the CNN's 50 um grid. `glyph/glyph_overlay.py` scores the triage windows of
a finished lane from them the way `glyph_win.py` does, and `glyph/line_overlay.py` runs `line_find.py` on z rebuilt
from `a` (exact around its mark threshold, z = 2). On the 343 control the CNN puts the three text windows first of
216 and the line finder puts the text line first. [Re-review of the screened bands.]

## Positive control on PHerc. 343

We ran the whole lane, unchanged apart from the fixes above, on the band of Erwin Nieuwlaar's claim (z 9500-11000,
CW, his own axis, 40 windings around w047) on one H100: 3 h 49 min end to end (setup 2 min, data 5 min, fit 2 h 21
min, render and ink 55 min, triage 24 min). The triage ranked w046, w045 and w047 first; his text is on w047/w048
of his fit, and winding numbers shift by about one between fits. w047 and w048 come back again and again in the top
13 (`results/control_PHerc0343_tri_rank.txt`). For scale, the best window of a band without text scored 2.7 to 4.6
against 24.4 here.

## Running it on Slurm (`slurm/`)

`lane_pace.sbatch` runs one band end to end in node-local `/tmp` (the kit writes tens of GB), rewriting the
kit's hard-coded `/home/liam/...` and `/opt` paths to the job's directory, and copies back only the results tar and
the logs. Set `FL_BASE` to a shared directory holding the kit under `fl/kit`.

```
sbatch --export=ALL,FL_BASE=/shared/you lane_pace.sbatch <id> <scroll> <z0> <z1> <um> <CW|ACW|AB> <umbilicus json, kit-relative> [wcenter] [maxw]
```

Things that cost us runs:

- The fit is CPU-bound. 16 cores are enough; on our cluster the A40 nodes (slower CPUs) could not finish a 30k-step
  fit in 6 h, the H100/H200/L40S nodes did it in about 2 h 10 min to 2 h 20 min.
- The native modules need C++23. Our cluster's gcc 13/14 modules were incomplete on compute nodes (no `cc1plus`),
  gcc 12 worked. `FL_GCC` names the gcc 12 prefix for the job, which then sets `CXX` only; setting `CC` to a broken
  compiler makes `posix-ipc` fail to build.
- `umb_pace.sbatch` builds axes for a list of scrolls on CPU; `normals_check.sbatch` reruns the normals check on any
  scroll that has Lasagna.

## Credits and licence

The lane, the kit and the triage are by Liam Warner and Nicola Bodill (MIT); the spiral recipe they build on is
Erwin Nieuwlaar's (MIT); the fitter, `vc_gen_umbilicus`, `vc3d_fiber_format` and the catalog are ScrollPrize/villa
and the Vesuvius Challenge open data. The code added here is MIT as well.
