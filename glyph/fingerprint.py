"""Rebuild the x_glyph GlyphNet forward pass from gA.pt alone (its train.py was never published).

Inputs: the half-size lane overlay JPGs kept in results.tar. fastpipe.overlay() writes, per 1/3-res pixel,
BGR = g(1-a) + a(.35, .85, .95) with a = clip((z - 0.9) / 1.8, 0, 1), so a = (R - B) / (0.6 * 255): exactly the a3 of
x_glyph.common.a_from_z, at 1/6 native resolution here. Windows are cut as glyph_win.py cuts them (128 px at 50 um).

Fingerprint: every BatchNorm keeps the running mean/var of its input over training batches. For each candidate wiring
(input channel order, activation, pooling) the batch statistics our windows produce at each BN input are compared with
the stored ones; the wiring the net was trained with reproduces them, a wrong one drifts. Greedy, layer by layer.
"""
import glob, os, sys, itertools
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

LANES = os.environ.get('FL_LANES', 'lanes')   # <lane>/results/{lane.env,maps/*/*_overlay.jpg}, as results.tar unpacks
CK = os.environ.get('GLYPH_CK', 'x_glyph/models/gA.pt')
UM, WIN = 50.0, 128


def lane_um(lane):
    for l in open(f'{LANES}/{lane}/results/lane.env'):
        if l.startswith('UM='):
            return float(l.split('=')[1])


def item_from_overlay(path, um_native):
    o = cv2.imread(path).astype(np.float32)
    a6 = np.clip((o[..., 2] - o[..., 0]) / (0.6 * 255), 0, 1)
    v6 = (o.max(-1) > 2).astype(np.uint8)
    v6 = cv2.morphologyEx(v6, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(np.float32)
    um6 = um_native * 6
    f = um6 / UM
    size = (max(1, int(round(a6.shape[1] * f))), max(1, int(round(a6.shape[0] * f))))
    a = cv2.resize(a6 * (v6 > 0.5), size, interpolation=cv2.INTER_AREA)
    m = cv2.resize(v6, size, interpolation=cv2.INTER_AREA) > 0.6
    a[~m] = 0
    return a.astype(np.float32), m


def onsheet_windows(m, win=WIN, step=16, min_frac=0.7):
    ii = cv2.integral(m.astype(np.uint8)); H, Wd = m.shape; out = []
    for y in range(0, max(1, H - win + 1), step):
        for x in range(0, max(1, Wd - win + 1), step):
            if ii[y + win, x + win] - ii[y, x + win] - ii[y + win, x] + ii[y, x] >= min_frac * win * win:
                out.append((y, x))
    return out


def collect(lanes, per_map=1, seed=0):
    rng = np.random.default_rng(seed); A, M = [], []
    for lane in lanes:
        um = lane_um(lane)
        for p in sorted(glob.glob(f'{LANES}/{lane}/results/maps/*/*_overlay.jpg')):
            a, m = item_from_overlay(p, um)
            ws = onsheet_windows(m)
            if not ws:
                continue
            for i in rng.choice(len(ws), min(per_map, len(ws)), replace=False):
                y, x = ws[i]; A.append(a[y:y + WIN, x:x + WIN]); M.append(m[y:y + WIN, x:x + WIN])
    return torch.from_numpy(np.stack(A)[:, None]), torch.from_numpy(np.stack(M)[:, None].astype(np.float32))


ACTS = {'relu': F.relu, 'silu': F.silu, 'gelu': F.gelu, 'lrelu': lambda x: F.leaky_relu(x, 0.1), 'elu': F.elu}


def bn_gap(x, bn):
    """distance between the batch statistics of a BN input and the BN's running statistics (per channel, averaged)"""
    mu = x.mean((0, 2, 3)); var = x.var((0, 2, 3))
    rm, rv = bn.running_mean, bn.running_var
    return float(((mu - rm).abs() / rv.sqrt()).mean()), float((var / rv).log().abs().mean())


def main():
    ck = torch.load(CK, map_location='cpu', weights_only=False); sd = ck['state']
    lanes = sys.argv[1].split(',') if len(sys.argv) > 1 else ['c343', 's0175b', 's0191c', 's0257b', 's0358c', 's0483Bb', 's0813c', 's0175c']
    a, m = collect(lanes)
    print(f'{len(a)} windows, mean a {a.mean():.4f}, mean m {m.mean():.4f}')

    def conv(name, x, stride=1):
        return F.conv2d(x, sd[f'{name}.weight'], None, stride, 1)

    def bn(name, x):
        return F.batch_norm(x, sd[f'{name}.running_mean'], sd[f'{name}.running_var'], sd[f'{name}.weight'], sd[f'{name}.bias'], False)

    class S:   # running stats holder for bn_gap
        def __init__(s, name): s.running_mean, s.running_var = sd[f'{name}.running_mean'], sd[f'{name}.running_var']

    with torch.no_grad():
        print('\n== b1.1 (input wiring)')
        inputs = {'[a,m]': torch.cat([a, m], 1), '[m,a]': torch.cat([m, a], 1), '[a,1-m]': torch.cat([a, 1 - m], 1),
                  '[2a-1,m]': torch.cat([2 * a - 1, m], 1), '[a,0]': torch.cat([a, 0 * m], 1)}
        for k, x in inputs.items():
            print(f'  {k:10s} mean gap {bn_gap(conv("b1.0", x), S("b1.1"))[0]:.3f}  log-var gap {bn_gap(conv("b1.0", x), S("b1.1"))[1]:.3f}')
        x0 = inputs['[a,m]']
        print('\n== b1.4 (activation)')
        for an, act in ACTS.items():
            h = act(bn('b1.1', conv('b1.0', x0)))
            g = bn_gap(conv('b1.3', h), S('b1.4')); print(f'  {an:6s} mean gap {g[0]:.3f}  log-var gap {g[1]:.3f}')
        best = {}
        for an, act in ACTS.items():
            h1 = act(bn('b1.4', conv('b1.3', act(bn('b1.1', conv('b1.0', x0))))))
            for pn, pool in {'max2': lambda t: F.max_pool2d(t, 2), 'avg2': lambda t: F.avg_pool2d(t, 2), 'none': lambda t: t}.items():
                g = bn_gap(conv('b2.0', pool(h1)), S('b2.1')); best[(an, pn)] = g
            g = bn_gap(conv('b2.0', h1, stride=2), S('b2.1')); best[(an, 'stride2')] = g
        print('\n== b2.1 (activation x pooling between blocks)')
        for k, g in sorted(best.items(), key=lambda kv: kv[1][0] + kv[1][1]):
            print(f'  {k[0]:6s} {k[1]:8s} mean gap {g[0]:.3f}  log-var gap {g[1]:.3f}')


if __name__ == '__main__':
    main()
