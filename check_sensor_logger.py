"""Check a real Sensor Logger CSV or ZIP locally, bypassing the website.
Usage: python check_sensor_logger.py "path\\to\\up_001_processed.csv"
"""
import sys
from pathlib import Path

from sensor_input import read_accelerometer

if len(sys.argv) != 2:
    print(r'Cach chay: python check_sensor_logger.py "E:\DuLieu\up_001_processed.csv"')
    raise SystemExit(1)
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f'Khong tim thay: {path}')
values, source = read_accelerometer(path.read_bytes(), path.name)
print(f'FILE HOP LE: {path.name}')
print(f'Nguon: {source} | So dong: {len(values)} | 3 truc: x,y,z')
print('3 dong dau:')
print(values[:3])

model_file = Path(__file__).parent / 'models' / 'gesture_real_v1.joblib'
if model_file.exists():
    import joblib
    from ml import features, KEY_TO_LABEL
    bundle = joblib.load(model_file)
    model = bundle['model']
    proba = model.predict_proba(features(values)[None, :])[0]
    results = sorted(zip(model.classes_, proba), key=lambda p: -p[1])
    print('\nDU DOAN TU MODEL HIEN TAI:')
    for key, p in results:
        print(f'  {KEY_TO_LABEL.get(str(key), str(key))}: {p*100:.1f}%')
    print('Ghi chu: model cu chi duoc huan luyen tren du lieu cong khai, co the doan sai iPhone.')
else:
    print('CHUA THAY MODEL: models/gesture_real_v1.joblib (van doc duoc du lieu).')
