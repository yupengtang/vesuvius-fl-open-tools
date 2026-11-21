"""sense_from_catalog.py [SCROLL ...] : spiral outward sense (CW/ACW) of scroll volumes from the open-data catalog.

The open-data bucket's metadata.json carries two volume properties, z_direction_is_top_to_bottom and
left_handed_coordinates. villa's spiral fitter (spiral-fitting/surface_orientation.py, spiral_outward_sense_for)
derives the outward sense from them: ACW when they differ, CW when they are equal. This prints that sense per
volume, or "unknown" when the catalog leaves either property empty.

Checked against the two published First Letters lanes: PHerc0306B -> ACW (the 0306B team's lane) and
PHerc0343 -> CW (Erwin Nieuwlaar's recipe).
"""
import gzip
import json
import sys
import urllib.request

URL = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json'


def load_catalog(url=URL):
    with urllib.request.urlopen(url, timeout=120) as r:
        raw = r.read()
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    return json.loads(raw)


def volumes(node):
    """Yield every catalog entry that carries volume properties."""
    if isinstance(node, dict):
        props = node.get('properties')
        if isinstance(props, dict) and 'z_direction_is_top_to_bottom' in props and node.get('id'):
            yield node
        for v in node.values():
            yield from volumes(v)
    elif isinstance(node, list):
        for v in node:
            yield from volumes(v)


def sense(z_top_to_bottom, left_handed):
    if z_top_to_bottom is None or left_handed is None:
        return 'unknown'
    return 'ACW' if z_top_to_bottom != left_handed else 'CW'


def main(scrolls):
    rows = {}
    for v in volumes(load_catalog()):
        p = v['properties']
        sample = v.get('sample_id') or ''
        if scrolls and sample not in scrolls:
            continue
        rows[(sample, v['id'])] = (p.get('z_direction_is_top_to_bottom'), p.get('left_handed_coordinates'),
                                  p.get('pixel_size_um'))
    for (sample, vid), (z, lh, um) in sorted(rows.items()):
        print(f'{sample}\t{vid}\t{um}\t{sense(z, lh)}\tz_top_to_bottom={z}\tleft_handed={lh}')


if __name__ == '__main__':
    main(set(sys.argv[1:]))
