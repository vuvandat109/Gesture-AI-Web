"""Locally export ONE correctly classified UEA TEST example per gesture.

For manual UI demonstration only; selected correct samples are NOT an unbiased
accuracy measurement. Keep generated samples_real/ local (ignored in Git).
"""
from pathlib import Path
import csv
import joblib
from train_real_model import load_split, ROOT
from ml import MODEL_KEYS

bundle_path = ROOT / 'models' / 'gesture_real_v1.joblib'
if not bundle_path.exists():
    raise SystemExit('Cần chạy: python train_real_model.py trước.')
model = joblib.load(bundle_path)['model']
examples, labels = load_split('TEST')
folder = ROOT / 'samples_real'
folder.mkdir(exist_ok=True)
remaining = set(MODEL_KEYS)
for signal, label in zip(examples, labels):
    if label not in remaining:
        continue
    from ml import features
    predicted = model.predict(features(signal).reshape(1, -1))[0]
    if predicted != label:
        continue
    path = folder / f'{label}_real_sample.csv'
    with path.open('w', encoding='utf-8', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['ax', 'ay', 'az'])
        writer.writerows(signal)
    remaining.remove(label)
    print(f'{label} -> {path} [dự đoán đúng]')
if remaining:
    print('Chưa tìm được mẫu dự đoán đúng của:', ', '.join(remaining))
print('CHÚ Ý: Các ví dụ được CHỌN vì model dự đoán đúng, KHÔNG dùng để báo cáo độ chính xác.')
print('Nguồn mẫu: tập TEST độc lập UWaveGestureLibrary. Không commit samples_real/.')
