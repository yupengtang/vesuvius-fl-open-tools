"""tracks_compare.py OURS.dbm.vctracks PUBLISHED_VCTRACKS_URL Z0 Z1 [--z-origin Z]

Compare a locally extracted packed track store with the published store of the same scroll over [Z0, Z1): horizontal
tracks (family 0) ribbon by ribbon, and vertical tracks (families 1, 2) that lie entirely inside the window shrunk by
64 slices. Only the per-track arrays are read (offsets, z_bounds, family_codes), not the coordinates.
"""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('ours'); ap.add_argument('url'); ap.add_argument('z0', type=int); ap.add_argument('z1', type=int)
ap.add_argument('--z-origin', type=int, default=0, help='z_min of the published extraction (ribbon grid origin)')
a = ap.parse_args()


def load(get):
    n = json.loads(get('metadata.json'))['track_count']
    off = np.frombuffer(get('offsets.i64'), '<i8')
    zb = np.frombuffer(get('z_bounds.i32'), '<i4').reshape(-1, 2)
    fam = np.frombuffer(get('family_codes.i8'), 'i1')
    assert len(off) == n + 1 and len(zb) == n and len(fam) == n, (len(off), len(zb), len(fam), n)
    return np.diff(off), zb, fam


ours = load(lambda f: (Path(a.ours) / f).read_bytes())
pub = load(lambda f: urllib.request.urlopen(f'{a.url.rstrip("/")}/{f}', timeout=600).read())

lo, hi = a.z0 + 8, a.z1 - 8
print(f'horizontal ribbons in [{lo}, {hi}): ribbon  ours n/mean pts  published n/mean pts')
rows = []
for name, (npts, zb, fam) in (('ours', ours), ('pub', pub)):
    h = fam == 0
    r = zb[h, 0] - (zb[h, 0] - a.z_origin) % 8
    keep = (r >= lo) & (r < hi)
    rows.append({int(z): (int((r[keep] == z).sum()), float(npts[h][keep][r[keep] == z].mean())) for z in np.unique(r[keep])})
for z in sorted(set(rows[0]) | set(rows[1])):
    o, p = rows[0].get(z, (0, 0.0)), rows[1].get(z, (0, 0.0))
    print(f'  {z:6d}  {o[0]:6d} / {o[1]:6.1f}   {p[0]:6d} / {p[1]:6.1f}')
no, npub = sum(v[0] for v in rows[0].values()), sum(v[0] for v in rows[1].values())
print(f'horizontal total: ours {no}, published {npub}, ratio {no / max(npub, 1):.3f}')

vlo, vhi = a.z0 + 64, a.z1 - 64
for code in (1, 2):
    c = []
    for npts, zb, fam in (ours, pub):
        m = (fam == code) & (zb[:, 0] >= vlo) & (zb[:, 1] < vhi)
        c.append((int(m.sum()), float(npts[m].mean()) if m.any() else 0.0))
    print(f'vertical family {code} inside [{vlo}, {vhi}): ours {c[0][0]} (mean {c[0][1]:.1f} pts), '
          f'published {c[1][0]} (mean {c[1][1]:.1f} pts), ratio {c[0][0] / max(c[1][0], 1):.3f}')
