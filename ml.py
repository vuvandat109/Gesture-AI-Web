"""Real accelerometer-gesture preprocessing, used identically in train and infer.

The UWaveGestureLibrary multivariate archive measures accelerometer x/y/z,
315 timestamps. These features work with a variable-length CSV of three axes.
A different phone orientation or sensor device may cause distribution shift.
"""
from __future__ import annotations
import numpy as np

# The numbers refer to the original gesture vocabulary illustrated in Liu et al.
# 3 = right, 4 = left, 5 = up, 6 = down, 7 = clockwise/circular stroke.
# Class 7 is a circular hand movement, not an isolated wrist-twist measurement.
CLASS_TO_KEY = {3: 'right', 4: 'left', 5: 'up', 6: 'down', 7: 'circle'}
KEY_TO_LABEL = {
    'right': 'Vuốt phải',
    'left': 'Vuốt trái',
    'up': 'Vuốt lên',
    'down': 'Vuốt xuống',
    'circle': 'Xoay vòng tay',
}
MODEL_KEYS = ['right', 'left', 'up', 'down', 'circle']


def features(raw: np.ndarray, steps: int = 64) -> np.ndarray:
    """Feature vector from Nx3 acceleration; no artificial labels or gestures.

    Temporal interpolation to 64 frames + per-axis z-normalization + waveform
    statistics. We retain direction/orientation through signed temporal shape,
    not absolute accelerometer bias or gravity offset.
    """
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError('Cần dữ liệu dạng N × 3 tương ứng ax, ay, az.')
    if not 20 <= len(x) <= 10000 or not np.isfinite(x).all():
        raise ValueError('Cần 20–10.000 dòng gia tốc hợp lệ (không chứa NaN/Inf).')

    # Subtract per-axis DC offset. Dataset/device orientation can still differ.
    x = x - x.mean(axis=0, keepdims=True)
    src = np.linspace(0., 1., len(x))
    dest = np.linspace(0., 1., steps)
    signal = np.stack([np.interp(dest, src, x[:, i]) for i in range(3)], axis=1)
    amplitude = np.maximum(signal.std(axis=0), 1e-5)
    wave = np.clip(signal / amplitude, -8., 8.)
    derivative = np.diff(signal, axis=0)

    extras = []
    for arr in (signal, derivative):
        extras.extend(arr.std(axis=0))
        extras.extend(arr.max(axis=0))
        extras.extend(arr.min(axis=0))
        extras.extend(np.sqrt(np.mean(arr ** 2, axis=0)))
    extras.extend(np.mean(wave[:steps//2], axis=0))
    extras.extend(np.mean(wave[steps//2:], axis=0))
    extras.extend((np.argmax(np.abs(wave), axis=0) / steps).tolist())
    extras.extend([np.mean(wave[:,0] * wave[:,1]), np.mean(wave[:,1] * wave[:,2])])
    return np.concatenate([wave.ravel(), np.asarray(extras)]).astype(np.float32)
