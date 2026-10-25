"""Residual-encoder 3D U-Net (nnU-Net ResEnc style) with an auxiliary tumour-volume head.

Instance normalisation + LeakyReLU, strided-conv downsampling, transposed-conv upsampling,
deep supervision outputs. The volume head regresses log1p(whole-tumour volume in the patch, in mL)
from the bottleneck; it is used as a label-free anchor during test-time adaptation (VolTTA).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {
    #      features per stage                    residual blocks per encoder stage
    "S": ([16, 32, 64, 128],                     [1, 1, 1, 1]),           # smoke tests only
    "M": ([32, 64, 128, 256, 320, 320],          [1, 3, 4, 6, 6, 6]),     # ~ nnU-Net ResEnc-M
    "L": ([32, 64, 128, 256, 320, 320],          [1, 3, 4, 6, 6, 6]),     # ResEnc-L uses same net, larger patch/batch
}


def norm(c):
    return nn.InstanceNorm3d(c, affine=True, eps=1e-5)


class ConvNormAct(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(nn.Conv3d(cin, cout, 3, stride, 1, bias=False), norm(cout), nn.LeakyReLU(0.01, inplace=True))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = norm(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = norm(cout)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), norm(cout))

    def forward(self, x):
        r = x if self.skip is None else self.skip(x)
        x = self.act(self.n1(self.c1(x)))
        x = self.n2(self.c2(x))
        return self.act(x + r)


class ResEncUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, preset="M", deep_supervision=True):
        super().__init__()
        feats, blocks = PRESETS[preset]
        self.deep_supervision = deep_supervision
        self.stem = ConvNormAct(in_ch, feats[0])
        self.enc = nn.ModuleList()
        cin = feats[0]
        for i, (f, b) in enumerate(zip(feats, blocks)):
            stage = [ResBlock(cin, f, stride=1 if i == 0 else 2)] + [ResBlock(f, f) for _ in range(b - 1)]
            self.enc.append(nn.Sequential(*stage))
            cin = f
        self.up, self.dec, self.heads = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(len(feats) - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(feats[i], feats[i - 1], 2, 2, bias=False))
            self.dec.append(nn.Sequential(ConvNormAct(2 * feats[i - 1], feats[i - 1]), ConvNormAct(feats[i - 1], feats[i - 1])))
            self.heads.append(nn.Conv3d(feats[i - 1], out_ch, 1))
        self.vol_head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(feats[-1], 64),
                                      nn.LeakyReLU(0.01), nn.Linear(64, 1))
        self.n_down = len(feats) - 1

    def forward(self, x, return_volume=False):
        skips = []
        x = self.stem(x)
        for st in self.enc:
            x = st(x)
            skips.append(x)
        vol = self.vol_head(x).squeeze(1)
        outs = []
        for i, (up, dec, head) in enumerate(zip(self.up, self.dec, self.heads)):
            x = up(x)
            x = dec(torch.cat([x, skips[-(i + 2)]], 1))
            outs.append(head(x))
        outs = outs[::-1]  # full resolution first
        seg = outs if (self.deep_supervision and self.training) else outs[0]
        return (seg, vol) if return_volume else seg


def norm_parameters(model):
    return [p for m in model.modules() if isinstance(m, nn.InstanceNorm3d) for p in m.parameters()]


def select_parameters(model, which: str):
    """Parameters updated during test-time adaptation."""
    for p in model.parameters():
        p.requires_grad_(False)
    if which == "norm":
        params = norm_parameters(model)
    elif which == "encoder1":
        params = list(model.stem.parameters()) + list(model.enc[0].parameters()) + norm_parameters(model)
    elif which == "all":
        params = [p for n, p in model.named_parameters() if not n.startswith("vol_head")]
    else:
        raise ValueError(which)
    params = list({id(p): p for p in params}.values())
    for p in params:
        p.requires_grad_(True)
    return params


def patch_volume_target(seg: torch.Tensor, voxel_ml: float = 0.001) -> torch.Tensor:
    """log1p(tumour volume in mL) per patch; seg is (B,1,X,Y,Z) in [0,1]."""
    return torch.log1p(seg.flatten(1).sum(1) * voxel_ml)


def dice_bce_loss(logits, target, smooth=1e-5):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    dice = 1 - (2 * inter + smooth) / (p.sum() + target.sum() + smooth)
    return bce + dice


def deep_supervision_loss(outs, target):
    if not isinstance(outs, (list, tuple)):
        return dice_bce_loss(outs, target)
    weights = torch.tensor([1 / 2 ** i for i in range(len(outs))])
    weights[-1] = 0  # lowest resolution not used (as in nnU-Net)
    weights = weights / weights.sum()
    loss = 0
    for w, o in zip(weights, outs):
        if w == 0:
            continue
        t = target if o.shape[2:] == target.shape[2:] else F.interpolate(target, size=o.shape[2:], mode="nearest")
        loss = loss + w * dice_bce_loss(o.float(), t)
    return loss
