"""Train and export REAL-gesture classifier from public UWaveGestureLibrary.

Run: python train_real_model.py
Source: THUML Time-Series-Library's UWaveGestureLibrary_TRAIN/TEST.ts,
        based on the multivariate UEA gesture archive (not generated samples).
Gesture IDs 3 right, 4 left, 5 up, 6 down, 7 circular movement.

The dataset source is openly accessible, but upstream Rice's original gesture
library has distinct redistribution terms: raw TS files and exported example
CSV files are NOT included in the public repository/package by this project.
"""
from __future__ import annotations
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml import CLASS_TO_KEY, KEY_TO_LABEL, MODEL_KEYS, features

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'data_external' / 'UWaveGestureLibrary'
MODELS = ROOT / 'models'
BASE = 'https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/UWaveGestureLibrary/'
SOURCE_PAGE = 'https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main/UWaveGestureLibrary'


def ensure_source(filename: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / filename
    if path.exists() and path.stat().st_size > 1000:
        return path
    url = BASE + filename + '?download=true'
    print('Đang tải dữ liệu thật:', url, flush=True)
    try:
        req = Request(url, headers={'User-Agent': 'GestureAI-research-educational/1.0'})
        with urlopen(req, timeout=90) as r, path.open('wb') as output:
            import shutil
            shutil.copyfileobj(r, output)
    except (URLError, HTTPError, TimeoutError, OSError) as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            'Không tải được dữ liệu tự động. Hãy mở ' + SOURCE_PAGE
            + ' để tải file ' + filename + ' thủ công, rồi chép vào thư mục '
            + str(CACHE) + '\nChi tiết: ' + str(exc)
        ) from exc
    print('Đã tải', filename, f'({path.stat().st_size:,} bytes)')
    return path


def read_uea_ts(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    """Parse equal-length multivariate UEA .ts; 3 channel series : label."""
    signals, labels = [], []
    in_data = False
    with path.open('r', encoding='utf-8-sig') as file:
        for lineno, raw_line in enumerate(file, 1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if not in_data:
                if line.lower() == '@data':
                    in_data = True
                continue
            parts = line.split(':')
            if len(parts) != 4:
                raise ValueError(f'{path.name} dòng {lineno}: yêu cầu 3 trục và 1 nhãn.')
            channels = [np.fromstring(part, sep=',', dtype=np.float64) for part in parts[:3]]
            if min(map(len, channels)) < 20 or len({len(a) for a in channels}) != 1:
                raise ValueError(f'{path.name} dòng {lineno}: độ dài 3 trục không khớp.')
            label = int(float(parts[-1]))
            if label not in CLASS_TO_KEY:
                continue  # Keep only the five specifically mapped classes.
            array = np.stack(channels, axis=1)
            if not np.isfinite(array).all():
                raise ValueError(f'{path.name} dòng {lineno}: NaN/Inf.')
            signals.append(array)
            labels.append(CLASS_TO_KEY[label])
    if not signals:
        raise ValueError(f'{path}: không tìm thấy dữ liệu của năm lớp cần thiết.')
    return signals, np.asarray(labels)


def load_split(which: str) -> tuple[list[np.ndarray], np.ndarray]:
    filename = 'UWaveGestureLibrary_' + which + '.ts'
    return read_uea_ts(ensure_source(filename))


def main() -> None:
    print('=== HUẤN LUYỆN NHẬN DIỆN CỬ CHỈ TRÊN DỮ LIỆU THẬT ===')
    train_s, train_y = load_split('TRAIN')
    test_s, test_y = load_split('TEST')
    print(f'TRAIN: {len(train_s)} mẫu; TEST độc lập: {len(test_s)} mẫu')
    print('Nhãn:', {k: int(sum(train_y == k)) for k in MODEL_KEYS})
    print('Trích xuất đặc trưng...')
    X_train = np.stack([features(a) for a in train_s])
    X_test = np.stack([features(a) for a in test_s])
    x_fit, x_val, y_fit, y_val = train_test_split(
        X_train, train_y, test_size=0.22, stratify=train_y, random_state=42)
    candidates = {
        'SVM RBF': make_pipeline(StandardScaler(), SVC(C=5, gamma='scale', probability=True, class_weight='balanced', random_state=42)),
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(C=0.2, max_iter=3000, class_weight='balanced')),
    }
    scores = {}
    for name, model in candidates.items():
        model.fit(x_fit, y_fit)
        y_pred = model.predict(x_val)
        scores[name] = float(accuracy_score(y_val, y_pred))
        print(f'Validation | {name}: {scores[name]:.3%}')

    winner_name = max(scores, key=scores.get)
    print('Chọn mô hình theo validation:', winner_name)
    winner = candidates[winner_name]
    winner.fit(X_train, train_y)  # Refit on TRAIN only. TEST stays held out.
    y_test_pred = winner.predict(X_test)
    acc = float(accuracy_score(test_y, y_test_pred))
    macro_f1 = float(f1_score(test_y, y_test_pred, average='macro'))
    print('=== ĐÁNH GIÁ TRÊN TEST CHƯA TỪNG DÙNG TRAIN ===')
    print(f'Accuracy: {acc:.3%} | Macro-F1: {macro_f1:.3%}')
    print(classification_report(test_y, y_test_pred, labels=MODEL_KEYS, zero_division=0))
    print('Confusion matrix (right, left, up, down, circle):')
    print(confusion_matrix(test_y, y_test_pred, labels=MODEL_KEYS))

    info = {
        'trained_from': 'REAL UWaveGestureLibrary multivariate accelerometer recordings',
        'source_url': SOURCE_PAGE,
        'original_classes': {str(k): v for k,v in CLASS_TO_KEY.items()},
        'model_name': winner_name,
        'validation_accuracies': scores,
        'heldout_test_accuracy': acc,
        'heldout_test_macro_f1': macro_f1,
        'train_samples': len(train_y),
        'test_samples': len(test_y),
        'note': 'Independent UEA test split; may not transfer to a different phone orientation. Circle class is NOT isolated wrist twisting.',
    }
    MODELS.mkdir(exist_ok=True)
    output = MODELS / 'gesture_real_v1.joblib'
    joblib.dump({'model': winner, 'labels': KEY_TO_LABEL, 'keys': MODEL_KEYS, 'info': info}, output, compress=3)
    (MODELS / 'training_report.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Đã xuất model THẬT:', output)
    print('Báo cáo:', MODELS / 'training_report.json')
    print('Kích thước model (bytes):', output.stat().st_size)
    print('HOÀN THÀNH! Có thể khởi động website bằng python app.py')


if __name__ == '__main__':
    main()
