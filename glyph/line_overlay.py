"""line_overlay.py RESULTS_DIR... [--out F.json] [--device cpu|cuda] : line_find.py (single lines of letter-sized marks, glyph
CNN along each chain) on finished lanes, from the overlay JPGs that results.tar keeps instead of the deleted z maps.

The overlay holds a = clip((z - 0.9) / 1.8, 0, 1) at 1/6 native resolution (fastpipe.overlay, halved when packed), so z is
known exactly on 0.9 < z < 2.7, which contains line_find's mark threshold Z_HI = 2.0: z = 0.9 + 1.8 a where a > 0. On-sheet
pixels with a = 0 (z <= 0.9) get uniform 0.3-0.85 (any value under 0.9 is the same to the marks and the glyph input; a constant
would read as skipped tiles). The map is upsampled 3x to 2x the native voxel size so that line_find's glyph input (a_from_z:
area mean over 3 px, then clip) lands back on the overlay's a. The CT layer is the lane's ct/<winding>.jpg (1/3 resolution).
Scores therefore sit near, not on, line_find's calibration (known text lines 8.2-11.2, reviewed-null lanes max 6.8)."""
import argparse, glob, json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import line_find as LF                                                  # noqa: E402


def overlay_z(path, seed=0):
    o = cv2.imread(path).astype(np.float32)
    a6 = np.clip((o[..., 2] - o[..., 0]) / (0.6 * 255), 0, 1)
    v6 = cv2.morphologyEx((o.max(-1) > 2).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    rng = np.random.default_rng(seed)
    z6 = np.where(v6, np.where(a6 > 0, 0.9 + 1.8 * a6, rng.uniform(0.3, 0.85, a6.shape)), 0).astype(np.float32)
    h, w = z6.shape
    z = cv2.resize(z6, (w * 3, h * 3), interpolation=cv2.INTER_LINEAR)
    valid = cv2.resize(v6.astype(np.uint8), (w * 3, h * 3), interpolation=cv2.INTER_NEAREST) > 0
    z[~valid] = 0
    return z, valid


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('results', nargs='+'); ap.add_argument('--out', default='line_overlay.json')
    ap.add_argument('--device', default='cpu'); ap.add_argument('--top', type=int, default=3)
    a = ap.parse_args()
    gl = LF.Glyph('gA', a.device)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for r in a.results:
        r = r.rstrip('/')
        lane = os.path.basename(os.path.dirname(r)) if os.path.basename(r) == 'results' else os.path.basename(r)
        if lane in res:
            continue
        env = dict(l.strip().split('=', 1) for l in open(f'{r}/lane.env') if '=' in l); um = float(env['UM'])
        maps = []
        for p in sorted(glob.glob(f'{r}/maps/*/*_overlay.jpg')):
            name = os.path.basename(p).replace('_overlay.jpg', ''); w = name.split('_d0_')[0].rsplit('_', 1)[-1]
            side = 'rev' if '_reverse' in name else 'fwd'
            z, valid = overlay_z(p)
            ctp = f'{r}/ct/{w}.jpg'; ct = cv2.imread(ctp, cv2.IMREAD_GRAYSCALE) if os.path.exists(ctp) else None
            cands, meta = LF.run_map(z, valid, um * 2, gl, ct, um * 3, LF.ANGLES, recipe='lane')
            for c in cands:
                c.pop('geom', None)
            maps.append(dict(name=name, winding=w, side=side, **{k: v for k, v in meta.items() if k != 'glyph_null'},
                             candidates=cands[:a.top]))
        res[lane] = dict(um=um, scroll=env.get('SCROLL'), maps=maps)
        json.dump(res, open(a.out, 'w'))
        best = sorted(((c['score'], m['name'], c) for m in maps for c in m['candidates']), key=lambda t: -t[0])[:6]
        print(f'== {lane} ({env.get("SCROLL")}, {um} um): {len(maps)} maps, {sum(len(m["candidates"]) for m in maps)} kept chains', flush=True)
        for s, n, c in best:
            x = np.mean([q[0] for q in c['poly']]) * 2; y = np.mean([q[1] for q in c['poly']]) * 2   # poly is in map px (2x the native voxel)
            print(f'   {s:6.2f}  {n}  ~native x {x:.0f} y {y:.0f}  slots {c.get("n_slots")} len {c.get("len_mm")} mm  '
                  f'glyph_p {c.get("glyph_p")} contrast {c.get("contrast")}', flush=True)


if __name__ == '__main__':
    main()
