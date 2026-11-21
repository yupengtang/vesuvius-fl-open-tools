"""normals_check.py SCROLL Z0 Z1 WORK : synthetic normals (make_normals.py) vs the hosted Lasagna nx/ny on a band.

Downloads m7 level 1 for the band (+ halo) and Lasagna nx/ny group 2 for the same 4x chunks, runs make_normals.py,
and reports the angle between the two sheet normals (sign-free, as the fitter uses them) where both are defined."""
import json, math, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numcodecs, numpy as np, requests

S, Z0, Z1, WORK = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
EXTRA = sys.argv[5:]
B = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
HERE = Path(__file__).resolve().parent
cfg = json.load(open(HERE / 'vw/fl/scrolls.json'))[S]
surf = cfg['surface_url'].split('amazonaws.com/')[1]
lst = requests.get(f'{B}/?list-type=2&prefix={S}/representations/predictions/lasagna/&delimiter=/').text
las = lst.split('<Prefix>')[2].split('</Prefix>')[0].rstrip('/')


def get(url, dst):
    if dst.exists():
        return
    for t in range(5):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 404:
                return
            r.raise_for_status()
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(r.content)
            return
        except requests.RequestException:
            if t == 4:
                raise


M = WORK / 'm7'
(M / '1').mkdir(parents=True, exist_ok=True)
get(f'{B}/{surf}/1/.zarray', M / '1/.zarray')
meta = json.load(open(M / '1/.zarray'))
c = meta['chunks'][0]; sy, sx = meta['shape'][1:]
z1a, z1b = (Z0 // 2 - 40) // c, (Z1 // 2 + 40) // c
jobs = [(f'{B}/{surf}/1/{kz}/{ky}/{kx}', M / f'1/{kz}/{ky}/{kx}') for kz in range(z1a, z1b + 1)
        for ky in range(math.ceil(sy / c)) for kx in range(math.ceil(sx / c))]
with ThreadPoolExecutor(32) as ex:
    list(ex.map(lambda j: get(*j), jobs))
print('m7 chunks', len(jobs), flush=True)

OUT = WORK / 'syn'
subprocess.run([sys.executable, str(HERE / 'vw/x_recipe_erwin/tools/make_normals.py'), str(M), str(OUT),
                '--z', str(Z0), str(Z1), *EXTRA], check=True)

z4a, z4b = (Z0 // 4) // 32, math.ceil(math.ceil(Z1 / 4) / 32)
L = WORK / 'las'
jobs = []
for ch in ('nx', 'ny'):
    get(f'{B}/{las}/{S}_{ch}.ome.zarr/2/.zarray', L / f'{ch}/.zarray')
lm = json.load(open(L / 'nx/.zarray'))
ly, lx = lm['shape'][1:]
for ch in ('nx', 'ny'):
    jobs += [(f'{B}/{las}/{S}_{ch}.ome.zarr/2/{kz}/{ky}/{kx}', L / f'{ch}/{kz}/{ky}/{kx}') for kz in range(z4a, z4b)
             for ky in range(math.ceil(ly / 32)) for kx in range(math.ceil(lx / 32))]
with ThreadPoolExecutor(32) as ex:
    list(ex.map(lambda j: get(*j), jobs))

lcod = numcodecs.get_codec(lm['compressor'])
scod = numcodecs.get_codec(json.load(open(OUT / 'las_008_nx.ome.zarr/2/.zarray'))['compressor'])
ang, ang_w = [], []
both = only_las = only_syn = 0
for kz in range(z4a, z4b):
    for ky in range(math.ceil(ly / 32)):
        for kx in range(math.ceil(lx / 32)):
            def rd(root, ch, cod):
                p = root / ch / str(kz) / str(ky) / str(kx)
                return np.frombuffer(cod.decode(p.read_bytes()), np.uint8).astype(np.float32) if p.exists() else None
            lx_, ly_ = rd(L, 'nx', lcod), rd(L, 'ny', lcod)
            sx_, sy_ = rd(OUT / 'las_008_nx.ome.zarr', '2', scod), rd(OUT / 'las_008_ny.ome.zarr', '2', scod)
            if lx_ is None and sx_ is None:
                continue
            z = np.zeros(32 ** 3, np.float32)
            lx_ = z if lx_ is None else lx_; ly_ = z if ly_ is None else ly_
            sx_ = z if sx_ is None else sx_; sy_ = z if sy_ is None else sy_
            lv = (lx_ != 0) | (ly_ != 0); sv = (sx_ != 0) | (sy_ != 0)
            both += int((lv & sv).sum()); only_las += int((lv & ~sv).sum()); only_syn += int((sv & ~lv).sum())
            m = lv & sv
            if not m.any():
                continue
            def vec(a, b):
                nx = (a[m] - 128) / 127; ny = (b[m] - 128) / 127
                nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, None))
                v = np.stack([nx, ny, nz], 1)
                return v / np.linalg.norm(v, axis=1, keepdims=True)
            d = np.abs((vec(lx_, ly_) * vec(sx_, sy_)).sum(1)).clip(0, 1)
            ang.append(np.degrees(np.arccos(d)))
a = np.concatenate(ang)
print(f'{S} z{Z0}-{Z1}: voxels both {both}, lasagna only {only_las}, synthetic only {only_syn}')
print('angle deg: median %.1f  mean %.1f  p75 %.1f  p90 %.1f  share<10deg %.1f%%  share<20deg %.1f%%' % (
    np.median(a), a.mean(), np.percentile(a, 75), np.percentile(a, 90), (a < 10).mean() * 100, (a < 20).mean() * 100))
