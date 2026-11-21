"""glyph_overlay.py RESULTS_DIR... [--out F.json] [--device cpu|cuda] : glyph CNN (x_glyph gA) scores for finished lanes, read
from the overlay JPGs that results.tar keeps (the full z maps die with the node's /tmp, so glyph_win.py cannot be rerun).

fastpipe.overlay() draws BGR = g(1-a) + a(.35, .85, .95) with a = clip((z - 0.9) / 1.8, 0, 1) at 1/3 resolution, so
a = (R - B) / (0.6 * 255) is the a3 of x_glyph.common.a_from_z; packed at half size it sits at 1/6 native resolution
(52-58 um), next to the CNN's 50 um grid, and goes through a_from_a3 as glyph_win.py's maps do. On-sheet = any overlay
pixel above 2 (CT or ink), closed over 5 px. Skipped tiles cannot be told from blank papyrus here and count as on-sheet.

Per lane: glyph_win.py's windows (2600 native px at a 1300 step, max over 128 px sub-windows at stride 16, >= 70 %
on-sheet, 5-fold TTA), plus the line_find.py grid null (all on-sheet windows on a 32 px grid: median and MAD-sd) to set
against GLYPH_REF lane gA (0.586, 1.406). Writes {lane: {"um", "null", "maps": {map: [[x0, score], ...]}}} and prints each
lane's top windows with their z against the lane's own windows."""
import argparse, glob, json, os, sys
import numpy as np, cv2, torch
cv2.setNumThreads(int(os.environ.get('OMP_NUM_THREADS', '4')))
W = '/home/liam/projects/vesuvius-work'
sys.path.insert(0, f'{W}/x_glyph')
from common import a_from_a3, onsheet_windows, WIN as GWIN, UM          # noqa: E402  x_glyph helpers
from evaluate import score_windows                                      # noqa: E402
from train import load_model                                            # noqa: E402

WIN, STEP = 2600, 1300


def overlay_item(path, um):
    o = cv2.imread(path).astype(np.float32)
    a6 = np.clip((o[..., 2] - o[..., 0]) / (0.6 * 255), 0, 1)
    v6 = cv2.morphologyEx((o.max(-1) > 2).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(np.float32)
    a, m = a_from_a3(a6 * (v6 > 0.5), v6, um * 6)
    return a, m, o.shape[1] * 6


def window_scores(a, m, wn, um, net):
    """glyph_win.window_scores on an overlay item: [[x0, score], ...]"""
    f = um / UM; out = []
    for x0 in range(0, max(1, wn - WIN + 1), STEP):
        x50, w50 = int(round(x0 * f)), int(round(min(WIN, wn - x0) * f))
        it = dict(a=a[:, x50:x50 + w50], mask=m[:, x50:x50 + w50])
        if it['mask'].mean() < 0.3:
            continue
        ws = onsheet_windows(it, GWIN, 16, 0.7)
        out.append([x0, round(float(score_windows(net, it, ws).max()), 3) if ws else None])
    return out


def grid_scores(a, m, net, step=32):
    ws = onsheet_windows(dict(mask=m), GWIN, step, 0.7)
    return score_windows(net, dict(a=a, mask=m), ws) if ws else np.zeros(0, np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('results', nargs='+'); ap.add_argument('--out', default='glyph_overlay.json')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    net = load_model(f'{W}/x_glyph/models/gA.pt', a.device)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for r in a.results:
        lane = os.path.basename(os.path.dirname(r.rstrip('/'))) if os.path.basename(r.rstrip('/')) == 'results' else os.path.basename(r.rstrip('/'))
        if lane in res:
            continue
        env = dict(l.strip().split('=', 1) for l in open(f'{r}/lane.env') if '=' in l); um = float(env['UM'])
        maps, grid = {}, []
        for p in sorted(glob.glob(f'{r}/maps/*/*_overlay.jpg')):
            am, mm, wn = overlay_item(p, um)
            maps[os.path.basename(p).replace('_overlay.jpg', '')] = window_scores(am, mm, wn, um, net)
            grid.append(grid_scores(am, mm, net))
        g = np.concatenate(grid) if grid else np.zeros(0)
        med = float(np.median(g)) if g.size else None
        null = dict(n=int(g.size), median=med, mad_sd=float(1.4826 * np.median(np.abs(g - med))) if g.size else None)
        res[lane] = dict(um=um, scroll=env.get('SCROLL'), null=null, maps=maps)
        json.dump(res, open(a.out, 'w'))
        wins = [(k, x0, s) for k, v in maps.items() for x0, s in v if s is not None]
        sv = np.array([s for _, _, s in wins])
        mu, sd = (sv.mean(), sv.std() + 1e-6) if sv.size else (0, 1)
        print(f'== {lane} ({env.get("SCROLL")}, {um} um): {len(wins)} windows, window max p50 {np.median(sv):.2f} p99 {np.percentile(sv, 99):.2f}; '
              f'grid null median {null["median"]:.3f} MAD-sd {null["mad_sd"]:.3f} (GLYPH_REF lane gA 0.586 / 1.406)', flush=True)
        for k, x0, s in sorted(wins, key=lambda t: -t[2])[:6]:
            print(f'   {s:6.2f}  z {(s - mu) / sd:+.2f}  {k} x{x0}{"  >= 9.06" if s >= 9.06 else ""}', flush=True)


if __name__ == '__main__':
    main()
