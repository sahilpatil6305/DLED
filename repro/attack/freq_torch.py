"""
Differentiable re-implementation of DLED's input pipeline, for white-box attacks.

Official pipeline (training/dataset/frequency_dateset.py, load_rgb + __getitem__),
applied to the 224x224 RGB image AFTER cv2.resize(INTER_CUBIC):

  spatial : ToTensor -> Normalize(mean 0.5, std 0.5)
  freq    : per channel  fft2 -> fftshift -> log(|F| + 1e-3)
            -> cv2.normalize(NORM_MINMAX, 0..255)   (per channel, per image)
            -> np.uint8(...)                         (TRUNCATION, not rounding)
            -> ToTensor -> Normalize(0.5, 0.5)

Threat model used by the harness: the attacker perturbs the 224x224 image x in
[0, 1] (pixel value / 255). cv2.resize to the same size is an exact identity, so
an adversarial 224x224 PNG fed back through the official loader yields the same
tensors (checked in test_freq_equivalence.py).

Only non-differentiable step: the uint8 truncation. Options (quantize=):
  'ste'   forward = floor (bit-exact with the loader), backward = identity (BPDA)
  'floor' forward = floor, zero gradient (what naive autograd would give)
  'none'  no quantisation (smooth surrogate; NOT what the model sees)
min/max in NORM_MINMAX are differentiable a.e. (sub-gradient flows to the argmin
/ argmax coefficient only). The max is almost always the DC term, i.e. the mean
intensity of the channel.

Agreement with the NumPy/OpenCV loader: <= 1 grey level, on ~1e-5 of pixels
(floating-point ties at integer boundaries), measured on DF40 frames by
test_freq_equivalence.py.
"""
import torch

EPS_LOG = 1e-3
MEAN, STD = 0.5, 0.5


def spectrum_255(x, dtype=torch.float64):
    """x: (B,3,H,W) in [0,1]. Returns the pre-truncation spectrum in [0,255], (B,3,H,W)."""
    v = x.to(dtype) * 255.0
    f = torch.fft.fftshift(torch.fft.fft2(v), dim=(-2, -1))
    m = torch.log(f.abs() + EPS_LOG)
    mn = m.amin(dim=(-2, -1), keepdim=True)
    mx = m.amax(dim=(-2, -1), keepdim=True)
    scale = 255.0 / (mx - mn)
    return m * scale + (0.0 - mn * scale)          # cv2's arithmetic order: src*scale + shift


def quantize_uint8(s, mode='ste'):
    if mode == 'none':
        return s
    q = torch.floor(s).clamp(0, 255)               # np.uint8(float) truncates
    if mode == 'floor':
        return q
    if mode == 'ste':
        return s + (q - s).detach()
    raise ValueError(mode)


def freq_map(x, quantize='ste', dtype=torch.float64):
    """x: (B,3,H,W) in [0,1] -> normalised frequency tensor as the model receives it."""
    s = quantize_uint8(spectrum_255(x, dtype), quantize)
    return ((s / 255.0 - MEAN) / STD).to(x.dtype)


def spatial_map(x):
    return (x - MEAN) / STD


def dled_inputs(x, quantize='ste'):
    """Build the data_dict DLED's forward() reads ('image', 'freq') from a [0,1] image batch."""
    return {'image': spatial_map(x), 'freq': freq_map(x, quantize)}


def load_resized(path, size=224):
    """Read an image exactly as load_rgb does, up to the resize. Returns (3,H,W) float in [0,1]."""
    import cv2
    import numpy as np
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f'Loaded image is None: {path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0


def save_png(x, path):
    """Save a (3,H,W) [0,1] image as an 8-bit PNG (rounded) -- the realisable adversarial example."""
    import cv2
    img = (x.detach().clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
