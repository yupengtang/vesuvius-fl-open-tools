"""GlyphNet and load_model, rebuilt from models/gA.pt: the published x_glyph ships the weights but not its train.py, which
glyph_win.py, line_find.py and zoom_lead.py import. Inference only (the training loop is not reproduced).

Shapes from the state dict: four blocks of two 3x3 conv (no bias) + BatchNorm (24-48-96-192 channels, 2 input channels =
cat(a, mask)), then Linear(384, 64) and Linear(64, 1) on the concatenated global mean and max of the last block. The rest
was fixed against the stored BatchNorm statistics: lane windows pushed through each candidate wiring reproduce the
running means of all 8 BatchNorms (channel correlation 0.88-0.99) only with ReLU and a 2x2 max-pool between blocks
(SiLU / GELU / ELU / average pooling: 0.36-0.74 at the deepest layers). Head readout cat(mean, max): on overlay maps of
our PHerc0343 control lane the w045-w048 text windows score 8.8-9.9 against 72 negative-lane windows at p90 4.45, max
7.07 (published: 0343 lane text 10.8, negatives p99 9.06); cat(max, mean) puts every window below -4 (glyph_rebuild/deep_fp.py)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class GlyphNet(nn.Module):
    def __init__(self, c=24, hidden=64):
        super().__init__()
        self.b1, self.b2, self.b3, self.b4 = _block(2, c), _block(c, 2 * c), _block(2 * c, 4 * c), _block(4 * c, 8 * c)
        self.head = nn.Sequential(nn.Linear(16 * c, hidden), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, a, m):
        """a: overlay ink (0-1), m: on-sheet mask, both [B, 1, 128, 128] at 50 um -> text logit [B]"""
        x = self.b1(torch.cat([a, m], 1))
        x = self.b2(F.max_pool2d(x, 2))
        x = self.b3(F.max_pool2d(x, 2))
        x = self.b4(F.max_pool2d(x, 2))
        return self.head(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)).squeeze(1)


def load_model(path, device='cuda'):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = GlyphNet(**ck.get('arch', {})).to(device).eval()
    net.load_state_dict(ck['state'])
    return net
