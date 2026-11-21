"""Deeper BN fingerprint for the GlyphNet candidates, then the documented scale check.

1. For each (activation, pooling) wiring, run the whole conv trunk on lane windows and compare, at all 8 BNs, the batch
   channel means with the stored running means: mean |gap| / sd and the across-channel correlation of the two vectors
   (the pattern survives the shift between our blank-heavy windows and the text-heavy training mix).
2. Head readouts ([avg,max], [max,avg], masked) scored like glyph_win.py (max over 128 px sub-windows, stride 16, 5-fold
   TTA, inside 2600 px tri_rank windows): the published scale is "top 1 % of negative windows 9.06 (8.15-10.3 between
   lanes), 0343 lane text 10.8".
"""
import glob, os, sys
import numpy as np, cv2, torch, torch.nn.functional as F
from fingerprint import collect, item_from_overlay, onsheet_windows, lane_um, ACTS, CK, LANES, WIN, UM

torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
sd = torch.load(CK, map_location='cpu', weights_only=False)['state']
BNS = ['b1.1', 'b1.4', 'b2.1', 'b2.4', 'b3.1', 'b3.4', 'b4.1', 'b4.4']
POOLS = {'max2': lambda t: F.max_pool2d(t, 2), 'avg2': lambda t: F.avg_pool2d(t, 2)}


def bn(name, x):
    return F.batch_norm(x, sd[f'{name}.running_mean'], sd[f'{name}.running_var'], sd[f'{name}.weight'], sd[f'{name}.bias'], False)


def trunk(x, act, pool, taps=None):
    for i, b in enumerate(('b1', 'b2', 'b3', 'b4')):
        if i:
            x = pool(x)
        for c, n in ((f'{b}.0', f'{b}.1'), (f'{b}.3', f'{b}.4')):
            x = F.conv2d(x, sd[f'{c}.weight'], None, 1, 1)
            if taps is not None:
                taps.setdefault(n, []).append((x.sum((0, 2, 3)), (x * x).sum((0, 2, 3)), x.numel() / x.shape[1]))
            x = act(bn(n, x))
    return x


def head(x, m, act, mode):
    if mode == 'avgmax':
        f = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)
    elif mode == 'maxavg':
        f = torch.cat([x.amax((2, 3)), x.mean((2, 3))], 1)
    else:   # masked mean and max over on-sheet cells
        mm = F.adaptive_avg_pool2d(m, x.shape[-2:]) > 0.5
        f = torch.cat([(x * mm).sum((2, 3)) / mm.sum((2, 3)).clamp_min(1), x.masked_fill(~mm, -1e4).amax((2, 3))], 1)
    h = act(F.linear(f, sd['head.0.weight'], sd['head.0.bias']))
    return F.linear(h, sd['head.3.weight'], sd['head.3.bias']).squeeze(1)


def net(a, m, act, pool, modes=('avgmax', 'maxavg', 'masked')):
    """5-fold TTA logits for every head readout at once (the trunk is shared): {mode: tensor}"""
    s = {k: 0 for k in modes}
    views = [(torch.rot90(a, k, (2, 3)), torch.rot90(m, k, (2, 3))) for k in range(4)] + [(a.flip(3), m.flip(3))]
    for aa, mm in views:
        x = trunk(torch.cat([aa, mm], 1), act, pool)
        for k in modes:
            s[k] = s[k] + head(x, mm, act, k)
    return {k: v / 5 for k, v in s.items()}


def fingerprint(a, m, wirings):
    for an, pn in wirings:
        taps = {}
        with torch.no_grad():
            for i in range(0, len(a), 64):
                trunk(torch.cat([a[i:i + 64], m[i:i + 64]], 1), ACTS[an], POOLS[pn], taps)
        gaps, cors = [], []
        for n in BNS:
            s1 = sum(t[0] for t in taps[n]); s2 = sum(t[1] for t in taps[n]); cnt = sum(t[2] for t in taps[n])
            mu = s1 / cnt; rm, rv = sd[f'{n}.running_mean'], sd[f'{n}.running_var']
            gaps.append(float(((mu - rm).abs() / rv.sqrt()).mean())); cors.append(float(np.corrcoef(mu.numpy(), rm.numpy())[0, 1]))
        print(f'  {an:6s} {pn}  gap ' + ' '.join(f'{g:.2f}' for g in gaps) + f'  | sum {sum(gaps):.2f}   corr ' +
              ' '.join(f'{c:.2f}' for c in cors) + f'  | min {min(cors):.2f}')


def lane_windows(lane, face_filter=None, step=16, keep=None, seed=0):
    """glyph_win.window_scores windows (2600 native px at a 1300 step) of LANE's overlays, yielded one at a time as
    (name, x0, a stack, m stack); keep=N yields a random N of them"""
    um = lane_um(lane); todo = []
    for p in sorted(glob.glob(f'{LANES}/{lane}/results/maps/*/*_overlay.jpg')):
        name = os.path.basename(p).replace('_overlay.jpg', '')
        if face_filter and not face_filter(name):
            continue
        a, m = item_from_overlay(p, um)
        f = um / UM; wn = a.shape[1] / f
        for x0 in range(0, max(1, int(wn) - 2600 + 1), 1300):
            todo.append((p, name, x0))
    if keep:
        rng = np.random.default_rng(seed); todo = [todo[i] for i in sorted(rng.choice(len(todo), min(keep, len(todo)), replace=False))]
    for p, name, x0 in todo:
        a, m = item_from_overlay(p, um); f = um / UM; wn = a.shape[1] / f
        x50, w50 = int(round(x0 * f)), int(round(min(2600, wn - x0) * f))
        aw, mw = a[:, x50:x50 + w50], m[:, x50:x50 + w50]
        if mw.mean() < 0.3:
            continue
        ws = onsheet_windows(mw, step=step)
        if ws:
            yield (name, x0, np.stack([aw[y:y + WIN, x:x + WIN] for y, x in ws]),
                   np.stack([mw[y:y + WIN, x:x + WIN] for y, x in ws]).astype(np.float32))


def score(wins, an, pn):
    """{mode: [(name, x0, max logit), ...]}"""
    res = {}
    with torch.no_grad():
        for name, x0, A, M in wins:
            parts = [net(torch.from_numpy(A[i:i + 64, None]), torch.from_numpy(M[i:i + 64, None]), ACTS[an], POOLS[pn])
                     for i in range(0, len(A), 64)]
            for k in parts[0]:
                res.setdefault(k, []).append((name, x0, float(torch.cat([q[k] for q in parts]).max())))
    return res


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'fp':
        a, m = collect(['c343', 's0175b', 's0191c', 's0257b', 's0358c', 's0483Bb', 's0813c', 's0175c'])
        fingerprint(a, m, [(an, pn) for an in ACTS for pn in POOLS])
    else:   # scale: 'scale relu max2 lane,lane,...'
        an, pn = sys.argv[2], sys.argv[3]
        t = score(lane_windows('c343', lambda n: any(w in n for w in ('w045_', 'w046_', 'w047_', 'w048_')) and 'reverse' in n), an, pn)
        lanes = sys.argv[4].split(',')
        n = score((w for lane in lanes for w in lane_windows(lane, step=32, keep=12)), an, pn)
        for mode in t:
            nv = np.array([r[2] for r in n[mode]]); top = sorted(t[mode], key=lambda r: -r[2])[:4]
            print(f'  {an} {pn} {mode:7s} {len(nv)} negatives p50 {np.percentile(nv, 50):.2f} p90 {np.percentile(nv, 90):.2f} '
                  f'max {nv.max():.2f} | 0343 top of {len(t[mode])}: ' + ', '.join(f'{r[0][-17:]} x{r[1]} {r[2]:.2f}' for r in top), flush=True)
