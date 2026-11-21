"""make_normals.py M7_DIR OUT_DIR --z Z0 Z1 [--tile N] [--sigma-grad 1.0] [--sigma-tensor 4.0] [--device cuda]

Synthetic Lasagna-style sheet normals for scrolls that have no hosted Lasagna prediction. Writes
OUT_DIR/las_008_nx.ome.zarr and las_008_ny.ome.zarr, group 2 (4x the base grid, chunks 32, uint8), in the
encoding the old spiral fitter decodes: n = (u8 - 128) / 127 with nz = +sqrt(1 - nx^2 - ny^2), 0 = no data.

Source: the m7 surface prediction at level 1 (2x), as erwin_data.sh downloads it (band +- the SDT halo).
Level 1 is mean-pooled 2x onto the 4x grid; the normal is the principal eigenvector of the structure tensor
(Gaussian derivative at --sigma-grad, tensor smoothed at --sigma-tensor, both in 4x voxels). The fitter only
uses |n . n_model|, so the sign is canonicalised to nz >= 0. Voxels with almost no gradient energy (air,
the far inside of thick fill) get 0 and are ignored by the loss.

Checked against the hosted Lasagna normals on PHerc0343 z10000-10256 (normals_check.py): median angle 7.7 deg,
60% within 10 deg, 81% within 20 deg at the defaults (sigma-grad 1, sigma-tensor 4); larger scales did worse.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numcodecs
import numpy as np
import torch
import torch.nn.functional as F

OUT_CHUNK = 32
OUT_CODEC = {'id': 'blosc', 'cname': 'lz4', 'clevel': 5, 'shuffle': 1, 'blocksize': 0}


def read_meta(array_dir):
    with open(array_dir / '.zarray') as f:
        return json.load(f)


def load_pooled(src_dir, z1_lo, z1_hi):
    """Level-1 slab [z1_lo, z1_hi) (z1_lo even), mean-pooled 2x -> float16 (Z4, Y4, X4) on the 4x grid."""
    meta = read_meta(src_dir)
    cz, cy, cx = meta['chunks']
    sz, sy, sx = meta['shape']
    codec = numcodecs.get_codec(meta['compressor']) if meta['compressor'] else None
    sep = meta.get('dimension_separator', '.')
    z1_hi = min(z1_hi, sz)
    y4, x4 = math.ceil(sy / 2), math.ceil(sx / 2)
    out = np.zeros(((z1_hi - z1_lo + 1) // 2, y4, x4), np.float16)
    found = missing = 0
    for kz in range(z1_lo // cz, (z1_hi - 1) // cz + 1):
        for ky in range(math.ceil(sy / cy)):
            for kx in range(math.ceil(sx / cx)):
                p = src_dir / (f'{kz}/{ky}/{kx}' if sep == '/' else f'{kz}.{ky}.{kx}')
                if not p.exists():
                    missing += 1
                    continue
                found += 1
                raw = p.read_bytes()
                a = np.frombuffer(codec.decode(raw) if codec else raw, np.uint8).reshape(cz, cy, cx)
                z0c = kz * cz
                lo, hi = max(z1_lo, z0c), min(z1_hi, z0c + cz)
                a = a[lo - z0c:hi - z0c].astype(np.float32)
                # every chunk origin is even (192), so 2x pooling never straddles chunks in y/x;
                # z is pooled on the absolute grid, pad a trailing odd slice
                if (hi - lo) % 2:
                    a = np.concatenate([a, a[-1:]], 0)
                a = a.reshape(a.shape[0] // 2, 2, cy // 2, 2, cx // 2, 2).mean((1, 3, 5)) / 255.0
                oz = (lo - z1_lo) // 2
                oy, ox = ky * cy // 2, kx * cx // 2
                ny_, nx_ = min(cy // 2, y4 - oy), min(cx // 2, x4 - ox)
                out[oz:oz + a.shape[0], oy:oy + ny_, ox:ox + nx_] = a[:, :ny_, :nx_]
    print(f'level-1 chunks: {found} read, {missing} absent (zero)', flush=True)
    return out


def gauss_kernel(sigma, device, order=0):
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    if order == 1:
        g = -x / sigma ** 2 * g
    return g


def sep_filter(v, kz, ky, kx):
    """v: (Z, Y, X) float32; separable correlation with replicate padding."""
    v = v[None, None]
    for k, axis in ((kz, 0), (ky, 1), (kx, 2)):
        r = (k.numel() - 1) // 2
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = k.numel()
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = pad[2 * (2 - axis) + 1] = r
        v = F.conv3d(F.pad(v, pad, mode='replicate'), k.view(shape))
    return v[0, 0]


def principal_normals(block, sg, st, energy_min):
    """block: (Z, Y, X) float32 on device -> (nx, ny, nz, valid) for every voxel of the block."""
    dev = block.device
    g0 = gauss_kernel(sg, dev)
    d1 = gauss_kernel(sg, dev, order=1)
    # derivative of Gaussian; correlation with -x*g gives +d/dx
    gz = sep_filter(block, -d1, g0, g0)
    gy = sep_filter(block, g0, -d1, g0)
    gx = sep_filter(block, g0, g0, -d1)
    t = gauss_kernel(st, dev)
    T = {}
    for name, a, b in (('zz', gz, gz), ('yy', gy, gy), ('xx', gx, gx), ('zy', gz, gy), ('zx', gz, gx), ('yx', gy, gx)):
        T[name] = sep_filter(a * b, t, t, t)
    del gz, gy, gx
    trace = T['zz'] + T['yy'] + T['xx']
    # power iteration from the tensor column with the largest diagonal entry
    cols = torch.stack([T['xx'], T['yy'], T['zz']], 0).argmax(0)
    vx = torch.where(cols == 0, T['xx'], torch.where(cols == 1, T['yx'], T['zx']))
    vy = torch.where(cols == 0, T['yx'], torch.where(cols == 1, T['yy'], T['zy']))
    vz = torch.where(cols == 0, T['zx'], torch.where(cols == 1, T['zy'], T['zz']))
    for _ in range(8):
        nx = T['xx'] * vx + T['yx'] * vy + T['zx'] * vz
        ny = T['yx'] * vx + T['yy'] * vy + T['zy'] * vz
        nz = T['zx'] * vx + T['zy'] * vy + T['zz'] * vz
        norm = torch.sqrt(nx * nx + ny * ny + nz * nz).clamp(min=1e-20)
        vx, vy, vz = nx / norm, ny / norm, nz / norm
    flip = vz < 0
    vx, vy, vz = torch.where(flip, -vx, vx), torch.where(flip, -vy, vy), torch.where(flip, -vz, vz)
    valid = trace > energy_min
    return vx, vy, vz, valid


def encode(c):
    return torch.clamp(torch.round(c * 127.0 + 128.0), 1, 255).to(torch.uint8)


def write_store(path, name, shape4):
    path.mkdir(parents=True, exist_ok=True)
    (path / '2').mkdir(exist_ok=True)
    (path / '.zgroup').write_text(json.dumps({'zarr_format': 2}))
    (path / '.zattrs').write_text(json.dumps({'multiscales': [{
        'version': '0.4', 'name': name,
        'axes': [{'name': a, 'type': 'space', 'unit': 'pixel'} for a in 'zyx'],
        'datasets': [{'path': '2', 'coordinateTransformations': [{'type': 'scale', 'scale': [4.0, 4.0, 4.0]}]}]}],
        'synthetic_normals': {'tool': 'make_normals.py', 'source': 'm7 level 1, structure tensor'}}, indent=1))
    (path / '2' / '.zarray').write_text(json.dumps({
        'shape': list(shape4), 'chunks': [OUT_CHUNK] * 3, 'dtype': '|u1', 'fill_value': 0, 'order': 'C',
        'filters': None, 'dimension_separator': '/', 'compressor': OUT_CODEC, 'zarr_format': 2}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('m7')
    ap.add_argument('out')
    ap.add_argument('--z', type=int, nargs=2, required=True, help='base-grid z range')
    ap.add_argument('--tile', type=int, default=512, help='unused; kept for erwin_data.sh compatibility')
    ap.add_argument('--sigma-grad', type=float, default=1.0)
    ap.add_argument('--sigma-tensor', type=float, default=4.0)
    ap.add_argument('--energy-min', type=float, default=1e-5)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    t0 = time.time()
    src = Path(a.m7) / '1'
    meta = read_meta(src)
    s1 = meta['shape']
    shape4 = [math.ceil(s / 2) for s in s1]
    halo = int(math.ceil(3 * a.sigma_grad) + math.ceil(3 * a.sigma_tensor)) + 2
    z4_lo = max(0, (a.z[0] // 4) // OUT_CHUNK * OUT_CHUNK)
    z4_hi = min(shape4[0], math.ceil(math.ceil(a.z[1] / 4) / OUT_CHUNK) * OUT_CHUNK)
    l4_lo, l4_hi = max(0, z4_lo - halo), min(shape4[0], z4_hi + halo)
    slab = load_pooled(src, 2 * l4_lo, 2 * l4_hi)
    print(f'pooled slab {slab.shape} ({time.time() - t0:.0f}s)', flush=True)
    codec = numcodecs.get_codec(OUT_CODEC)
    out = Path(a.out)
    stores = {'nx': out / 'las_008_nx.ome.zarr', 'ny': out / 'las_008_ny.ome.zarr'}
    for k, p in stores.items():
        write_store(p, k, shape4)
    n_valid = n_total = n_chunks = 0
    for zb in range(z4_lo, z4_hi, OUT_CHUNK):
        ze = min(zb + OUT_CHUNK, shape4[0])
        lo, hi = max(l4_lo, zb - halo), min(l4_hi, ze + halo)
        block = torch.from_numpy(slab[lo - l4_lo:hi - l4_lo].astype(np.float32)).to(a.device)
        vx, vy, vz, valid = principal_normals(block, a.sigma_grad, a.sigma_tensor, a.energy_min)
        sl = slice(zb - lo, ze - lo)
        ex, ey = encode(vx[sl]), encode(vy[sl])
        m = valid[sl]
        ex[~m] = 0
        ey[~m] = 0
        n_valid += int(m.sum())
        n_total += m.numel()
        ex, ey = ex.cpu().numpy(), ey.cpu().numpy()
        kz = zb // OUT_CHUNK
        for ky in range(math.ceil(shape4[1] / OUT_CHUNK)):
            for kx in range(math.ceil(shape4[2] / OUT_CHUNK)):
                ys, xs = slice(ky * OUT_CHUNK, (ky + 1) * OUT_CHUNK), slice(kx * OUT_CHUNK, (kx + 1) * OUT_CHUNK)
                cxa, cya = ex[:, ys, xs], ey[:, ys, xs]
                if not cxa.any():
                    continue
                for k, c in (('nx', cxa), ('ny', cya)):
                    full = np.zeros((OUT_CHUNK,) * 3, np.uint8)
                    full[:c.shape[0], :c.shape[1], :c.shape[2]] = c
                    d = stores[k] / '2' / str(kz) / str(ky)
                    d.mkdir(parents=True, exist_ok=True)
                    (d / str(kx)).write_bytes(codec.encode(full))
                n_chunks += 1
        del block, vx, vy, vz, valid
        print(f'z4 {zb}-{ze}: chunks so far {n_chunks} ({time.time() - t0:.0f}s)', flush=True)
    print(f'make_normals done: z4 {z4_lo}-{z4_hi}, valid {n_valid / max(1, n_total):.1%}, {n_chunks} chunks/channel, '
          f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
