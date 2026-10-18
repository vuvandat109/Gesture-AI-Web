"""Feature extraction and synthetic-example generator for a GESTURE DEMO.
Not a real-world trained gesture-recognition model.
"""
import numpy as np

LABELS = ['Vuốt phải', 'Vuốt trái', 'Vuốt lên', 'Vuốt xuống', 'Xoay cổ tay']

def features(raw):
    x = np.asarray(raw, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not 20 <= len(x) <= 10000 or not np.isfinite(x).all():
        raise ValueError('Dữ liệu cảm biến phải là ma trận N × 3, từ 20 đến 10000 dòng, không có NaN/Inf.')
    # Normalize offset gravity / orientation for this prototype.
    x = x - x.mean(axis=0, keepdims=True)
    d = np.diff(x, axis=0)
    feats = []
    for arr in (x, d):
        feats.extend(arr.mean(axis=0))
        feats.extend(arr.std(axis=0))
        feats.extend(arr.min(axis=0))
        feats.extend(arr.max(axis=0))
        feats.extend(np.sqrt((arr ** 2).mean(axis=0)))
    feats.extend(x[np.argmax(np.abs(x), axis=0), np.arange(3)])
    feats.extend(np.sum(x > 0, axis=0) / len(x))
    feats.extend(np.mean(x ** 3, axis=0))
    feats.extend(np.mean(d[:, 0] * d[:, 1]).reshape(1))
    feats.extend([np.mean(x[:-1, 0] * x[1:, 1] - x[1:, 0] * x[:-1, 1])])
    # temporal displacement cues from the integral of the centered acceleration
    vel = np.cumsum(x, axis=0) / len(x)
    feats.extend(vel[-1] - vel[0])
    feats.extend(np.max(vel, axis=0) - np.min(vel, axis=0))
    return np.asarray(feats, dtype=float)

def synthesize(label, rng, n=128):
    t = np.linspace(0, 1, n)
    phase = rng.uniform(-0.07, 0.07)
    center = np.clip(0.50 + phase, 0.32, 0.68)
    width = rng.uniform(0.12, 0.23)
    impulse = np.exp(-0.5 * ((t - center) / width) ** 2)
    swing = (t - center) / width * impulse
    sig = np.zeros((n, 3))
    amp = rng.uniform(0.9, 2.8)
    if label == 0:
        sig[:, 0] = amp * impulse; sig[:, 1] = 0.13 * amp * swing
    elif label == 1:
        sig[:, 0] = -amp * impulse; sig[:, 1] = -0.13 * amp * swing
    elif label == 2:
        sig[:, 1] = amp * impulse; sig[:, 0] = 0.13 * amp * swing
    elif label == 3:
        sig[:, 1] = -amp * impulse; sig[:, 0] = -0.13 * amp * swing
    else:
        angle = 2 * np.pi * rng.uniform(1.0, 1.7) * t + rng.uniform(0, 2 * np.pi)
        env = np.sin(np.pi * t) ** 2
        sig[:, 0] = amp * np.cos(angle) * env
        sig[:, 1] = amp * np.sin(angle) * env
        sig[:, 2] = 0.2 * amp * np.sin(angle * 0.5) * env
    sig += rng.normal(0, rng.uniform(0.035, 0.14), sig.shape)
    sig += rng.uniform(-0.12, 0.12, (1, 3))
    return sig
