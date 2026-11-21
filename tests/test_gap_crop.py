"""gap_stats(cols=...) must return exactly what the whole-render call returns for a window mask."""
import sys
import time

import numpy as np
import zarr

sys.path.insert(0, sys.argv[1])  # fl/scripts
from gap_gate import gap_stats  # noqa: E402

rng = np.random.default_rng(0)
H, W, um, win = 900, 16000, 9.362, 2600
yy, xx = np.mgrid[0:H, 0:W]
base = 120 + 40 * np.sin(xx / 37.0) * np.cos(yy / 53.0) + rng.normal(0, 12, (H, W))
for _ in range(400):                                   # dark cracks, some long
    x0, y0, L, ang = rng.integers(0, W), rng.integers(0, H), rng.integers(50, 900), rng.uniform(0, np.pi)
    t = np.arange(L)
    xs = np.clip((x0 + t * np.cos(ang)).astype(int), 0, W - 1); ys = np.clip((y0 + t * np.sin(ang)).astype(int), 0, H - 1)
    for o in range(-6, 7):
        base[np.clip(ys + o, 0, H - 1), xs] *= 0.25
base[:, :300] = 0; base[:40, :] = 0; base[:, W - 500:] = 0  # off-sheet borders and a hole
base[300:420, 7000:7600] = 0
vol = np.repeat(np.clip(base, 0, 255).astype(np.uint8)[None], 31, 0) + rng.integers(0, 3, (31, H, W)).astype(np.uint8)
vol[:, base == 0] = 0
root = zarr.open_group(sys.argv[2], mode='w')
root.create_array('0', data=vol, chunks=(31, 256, 256))
ink = (rng.normal(0, 1, (H, W)) + 3 * (np.sin(xx / 90.0) > 0.97)).astype(np.float32)

worst = 0.0
for x0 in (0, 1300, 2600, 6500, 9100, W - win, W - 1800):
    m = np.zeros((H, W), bool); m[:, x0:x0 + win] = True
    t = time.time(); full = gap_stats(sys.argv[2], ink, d=0, um=um, mask=m); tf = time.time() - t
    t = time.time(); crop = gap_stats(sys.argv[2], ink, d=0, um=um, mask=m, cols=(x0, min(W, x0 + win))); tc = time.time() - t
    same = all(full[k] == crop[k] or (np.isnan(full[k]) and np.isnan(crop[k])) for k in full)
    worst = max(worst, max(abs(full[k] - crop[k]) for k in full if not (np.isnan(full[k]) and np.isnan(crop[k]))))
    print(f'x0 {x0:5d}: identical {same}  full {tf:.2f}s crop {tc:.2f}s  {full}' + ('' if same else f'\n          crop {crop}'))
print('max abs difference over all outputs:', worst)
