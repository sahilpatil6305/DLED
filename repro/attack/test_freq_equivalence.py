"""
CPU-only check that repro/attack/freq_torch.py reproduces DLED's official loader.

Run from the repo root (after `source repro/env.sh`), through run.sh for a log:
  repro/run.sh freq_equiv python repro/attack/test_freq_equivalence.py

Never touches the GPU (CUDA_VISIBLE_DEVICES is cleared before torch loads) and
uses 2 CPU threads by default (ATTACK_THREADS), so it is safe next to training.

Checks, on real DF40 frames (a few per class, first files in sorted order):
  1. spatial tensor identical to the official loader's
  2. frequency map: |torch - official| <= 1 grey level, mismatch rate reported
  3. round trip: the image saved as a 224x224 PNG and re-read by the OFFICIAL
     load_rgb gives the same spatial and frequency inputs (attack is realisable)
  4. gradients: gradcheck of the smooth spectrum; STE gives finite, non-zero
     gradients; naive floor gives zero gradients (gradient masking if unhandled)
Writes freq_equivalence.json to $DLED_LOG_DIR (the run folder under run.sh).
"""
import argparse
import json
import os
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get('ATTACK_THREADS', '2')))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'training'), os.path.dirname(__file__)]
import freq_torch as ft  # noqa: E402

DATA = os.environ.get('DLED_DATA', os.path.expanduser('~/DLED/data'))
DEFAULT_DIRS = {   # NOEFS test label: 0 real, 1 FS, 2 FR, 3 EFS (unknown)
    'real_ff++': 'FaceForensics++/original_sequences/youtube/c23/frames',
    'FS_simswap': 'DF40/simswap/ff/frames',
    'FS_uniface': 'DF40/uniface/ff/frames',
    'FR_fomm': 'DF40/fomm/ff/frames',
    'FR_sadtalker': 'DF40/sadtalker/ff/frames',
    'EFS_StyleGAN2': 'DF40/StyleGAN2/ff',
    'EFS_ddim': 'DF40/ddim/ff',
    'EFS_MidJourney': 'DF40/MidJourney/fake',
}
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp')


def first_images(d, k):
    out = []
    for root, dirs, files in os.walk(d):
        dirs.sort()
        for f in sorted(files):
            if f.lower().endswith(IMG_EXT):
                out.append(os.path.join(root, f))
                if len(out) >= k:
                    return out
    return out


def official_loader():
    from dataset.frequency_dateset import DeepfakeFrequencyDataset as D
    ds = D.__new__(D)                      # official methods, no __init__ side effects
    ds.config = {'resolution': 224, 'mean': [0.5] * 3, 'std': [0.5] * 3}
    return ds


def official_tensors(ds, path):
    """Exactly __getitem__'s test path: load_rgb -> np.array -> to_tensor -> normalize."""
    img, freq, _ = ds.load_rgb(path)
    img, freq = np.array(img), np.array(freq)
    return ds.normalize(ds.to_tensor(img)), ds.normalize(ds.to_tensor(freq)), freq


def gradient_checks():
    torch.manual_seed(0)
    res = {}
    x = torch.rand(1, 3, 8, 8, dtype=torch.float64, requires_grad=True)
    res['gradcheck_smooth_spectrum'] = bool(torch.autograd.gradcheck(
        lambda t: ft.spectrum_255(t), (x,), eps=1e-6, atol=1e-4))
    x = torch.rand(2, 3, 224, 224, requires_grad=True)
    w = torch.randn(2, 3, 224, 224)
    for mode in ('ste', 'floor', 'none'):
        x.grad = None
        (ft.freq_map(x, mode) * w).sum().backward()
        g = x.grad
        res[f'grad_{mode}'] = {'finite': bool(torch.isfinite(g).all()),
                               'nonzero_frac': float((g != 0).float().mean()),
                               'abs_mean': float(g.abs().mean())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_dir', type=int, default=6)
    ap.add_argument('--out', default=os.environ.get('DLED_LOG_DIR', '.'))
    args = ap.parse_args()
    ds = official_loader()
    tmp = tempfile.mkdtemp(prefix='freq_equiv_')
    report = {'torch': torch.__version__, 'numpy': np.__version__, 'groups': {}}
    worst, n_pix, n_mis, n_img, rt_fail = 0, 0, 0, 0, 0

    for name, rel in DEFAULT_DIRS.items():
        paths = first_images(os.path.join(DATA, rel), args.per_dir)
        if not paths:
            print(f'{name:16s} no images under {rel} -- skipped')
            continue
        g = {'n': len(paths), 'max_grey_diff': 0, 'mismatch_px': 0, 'spatial_max_diff': 0.0,
             'roundtrip_ok': 0, 'example': paths[0]}
        for p in paths:
            sp_off, fr_off, fr_u8 = official_tensors(ds, p)
            x = ft.load_resized(p).unsqueeze(0)
            with torch.no_grad():
                d = ft.dled_inputs(x, 'ste')
                q = torch.floor(ft.spectrum_255(x)).clamp(0, 255)[0].permute(1, 2, 0).numpy()
            diff = np.abs(q.astype(np.int64) - fr_u8.astype(np.int64))
            g['max_grey_diff'] = max(g['max_grey_diff'], int(diff.max()))
            g['mismatch_px'] += int((diff > 0).sum())
            g['spatial_max_diff'] = max(g['spatial_max_diff'], float((d['image'][0] - sp_off).abs().max()))
            tens_diff = float((d['freq'][0] - fr_off).abs().max())
            assert tens_diff <= 2.0 / 255 + 1e-6, (p, tens_diff)
            # round trip through the official reader
            rp = os.path.join(tmp, 'rt.png')
            ft.save_png(x[0], rp)
            sp_rt, fr_rt, _ = official_tensors(ds, rp)
            ok = torch.equal(sp_rt, sp_off) and torch.equal(fr_rt, fr_off)
            g['roundtrip_ok'] += int(ok)
            rt_fail += int(not ok)
            n_pix += diff.size
            n_img += 1
        worst = max(worst, g['max_grey_diff'])
        n_mis += g['mismatch_px']
        report['groups'][name] = g
        print(f"{name:16s} n={g['n']}  max_grey_diff={g['max_grey_diff']}  "
              f"mismatch_px={g['mismatch_px']}  spatial_max_diff={g['spatial_max_diff']:.2e}  "
              f"roundtrip_ok={g['roundtrip_ok']}/{g['n']}")

    report['grad'] = gradient_checks()
    for k, v in report['grad'].items():
        print(f'{k:28s} {v}')
    rate = n_mis / max(n_pix, 1)
    report.update(images=n_img, pixels=n_pix, mismatch_rate=rate, max_grey_diff=worst,
                  roundtrip_failures=rt_fail)
    gr = report['grad']
    passed = (n_img > 0 and worst <= 1 and rate < 1e-3 and rt_fail == 0
              and all(gg['spatial_max_diff'] == 0 for gg in report['groups'].values())
              and gr['gradcheck_smooth_spectrum']
              and gr['grad_ste']['finite'] and gr['grad_ste']['nonzero_frac'] > 0.5
              and gr['grad_floor']['nonzero_frac'] == 0)
    report['passed'] = bool(passed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'freq_equivalence.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(f'\nimages={n_img} pixels={n_pix} mismatch_rate={rate:.2e} max_grey_diff={worst} '
          f'roundtrip_failures={rt_fail}')
    print('FREQ EQUIVALENCE PASSED' if passed else 'FREQ EQUIVALENCE FAILED')
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
