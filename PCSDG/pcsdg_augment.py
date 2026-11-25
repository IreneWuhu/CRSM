"""Standalone task adaptation of PCSDG's FDA, global LSA, and SABA views."""
import random
import math
import numpy as np
import torch

def _normalize(a):
    a = a.astype(np.float32, copy=True)
    for b in range(a.shape[0]):
        for c in range(a.shape[1]):
            plane = a[b, c]
            a[b, c] = (plane - plane.mean()) / (plane.std() + 1e-5)
    return a

def normalize_tensor(x):
    return (x - x.mean(dim=(-2, -1), keepdim=True)) / (x.std(dim=(-2, -1), keepdim=True, unbiased=False) + 1e-5)

def _fda(x, beta_max=.1):
    donor = x[::-1].copy()
    out = np.empty_like(x, dtype=np.float32)
    for b in range(len(x)):
        L = round(.05 + random.random() * beta_max, 2)
        xf, yf = np.fft.fft2(x[b]), np.fft.fft2(donor[b])
        amp, phase = np.abs(xf), np.angle(xf)
        amp_t = np.abs(yf)
        src = np.fft.fftshift(amp, axes=(-2, -1)); trg = np.fft.fftshift(amp_t, axes=(-2, -1))
        h, w = src.shape[-2:]; radius = int(np.floor(min(h, w) * L))
        cy, cx = h // 2, w // 2
        src[:, cy-radius:cy+radius+1, cx-radius:cx+radius+1] = trg[:, cy-radius:cy+radius+1, cx-radius:cx+radius+1]
        amp = np.fft.ifftshift(src, axes=(-2, -1))
        out[b] = np.fft.ifft2(amp * np.exp(1j * phase)).real
    return out

def _bezier_map(values, n_points=4, n_times=2048):
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-8: return values.copy()
    xs, ys = [lo, hi], [lo, hi]
    for _ in range(n_points - 2):
        xs.insert(1, random.uniform(xs[0], xs[-1])); ys.insert(1, random.uniform(ys[0], ys[-1]))
    t = np.linspace(0, 1, n_times, dtype=np.float32)
    basis = np.stack([math.comb(n_points-1, i) * t**(n_points-1-i) * (1-t)**i for i in range(n_points)])
    xv, yv = np.dot(np.asarray(xs), basis), np.dot(np.asarray(ys), basis)
    xv, yv = np.sort(xv), np.sort(yv)
    return np.interp(values, xv, yv).astype(np.float32)

def _location_scale(x, vrange=(0., 255.)):
    scale = max(min(random.gauss(1, .1), 1.1), .9)
    location = random.gauss(0, .5)
    lo = vrange[0] - np.percentile(x, 20); hi = vrange[1] - np.percentile(x, 80)
    location = np.clip(location, lo, hi)
    return np.clip(x * scale + location, *vrange).astype(np.float32)

def _gla(x):
    out = np.empty_like(x, dtype=np.float32)
    for b in range(len(x)):
        image = _bezier_map(x[b])
        image = _location_scale(image)
        image[ x[b] <= .01 ] = x[b][ x[b] <= .01 ]
        out[b] = image
    return out

def _saba(x, mask):
    out = np.zeros_like(x, dtype=np.float32)
    labels = (np.where(mask[:, 1] > .5, 2, np.where(mask[:, 0] > .5, 1, 0))
              if mask.shape[1] > 1 else (mask[:, 0] > .5).astype(np.uint8))
    for b in range(len(x)):
        img = x[b]; brightness = .299 * img[0] + .587 * img[1] + .114 * img[2]
        for region_id in np.unique(labels[b]):
            region = labels[b] == region_id
            v = brightness[region]
            if not v.size: continue
            alpha = _bezier_map(v) / 255. + .5
            beta = random.gauss(0, .1) * 255
            for c in range(3):
                values = img[c][region]
                out[b, c][region] = np.clip(np.abs(values * alpha + beta), 0, 255)
    return out

def make_views(image, mask):
    """Return (clean,FDA,GLA,SABA), each using PCSDG per-image z-score input."""
    x = image.detach().cpu().numpy().astype(np.float32)
    y = mask.detach().cpu().numpy().astype(np.float32)
    raw = [x, _fda(x), _gla(x), _saba(x, y)]
    return [torch.from_numpy(_normalize(view)) for view in raw]
