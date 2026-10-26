"""Six-channel Sensor Logger baseline. No TF-C or DINOv3 claim."""
import io
from pathlib import Path
import zipfile
import numpy as np
from sensor_input import _read_table, _norm, MAX_UPLOAD_BYTES

CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

def read_motion(data, filename):
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError('File trống hoặc lớn hơn 5 MB.')
    if Path(filename).suffix.lower() == '.zip':
        if not zipfile.is_zipfile(io.BytesIO(data)):
            raise ValueError('ZIP không hợp lệ.')
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            entries = z.infolist()
            if len(entries) > 100:
                raise ValueError('ZIP chứa quá nhiều tệp.')
            def read_sensor(name):
                found = [e for e in entries if Path(e.filename).name.lower() == name]
                if len(found) != 1 or found[0].file_size > MAX_UPLOAD_BYTES:
                    raise ValueError('ZIP cần đúng một Accelerometer.csv và một Gyroscope.csv, mỗi tệp dưới 5 MB.')
                f = _read_table(z.read(found[0])); f.columns = [_norm(c) for c in f.columns]
                t = next((c for c in ['secondselapsed', 'time', 'timestamp'] if c in f), None)
                if t is None:
                    raise ValueError('CSV cảm biến thiếu thời gian để đồng bộ hai cảm biến.')
                if not all(c in f for c in ['x','y','z']) or not 20 <= len(f) <= 10000:
                    raise ValueError('CSV cần 20–10.000 dòng và đủ x,y,z.')
                return f[t].to_numpy(float), f[['x','y','z']].to_numpy(float)
            ta, a = read_sensor('accelerometer.csv'); tg, g = read_sensor('gyroscope.csv')
            if not (np.isfinite(ta).all() and np.isfinite(tg).all()) or np.any(np.diff(ta)<=0) or np.any(np.diff(tg)<=0):
                raise ValueError('Thời gian cảm biến phải tăng và hợp lệ.')
            mask = (ta >= tg[0]) & (ta <= tg[-1]); ta=ta[mask]; a=a[mask]
            x = np.column_stack([a, *[np.interp(ta, tg, g[:,i]) for i in range(3)]])
    else:
        f = _read_table(data); f.columns = [_norm(c) for c in f.columns]
        cols = [_norm(c) for c in CHANNELS]
        if len(set(f.columns)) != len(f.columns) or not all(c in f for c in cols):
            raise ValueError('Cần 6 cột acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z.')
        if 'trialid' in f and f.trialid.nunique() > 1:
            raise ValueError('Chỉ tải một cử chỉ/một trial cho mỗi lần dự đoán.')
        x = f[cols].to_numpy(float)
    if not 20 <= len(x) <= 10000 or not np.isfinite(x).all():
        raise ValueError('Cần 20–10.000 dòng cảm biến hữu hạn.')
    return x, 'Sensor Logger 6 channels'

def motion_features(raw):
    x = np.asarray(raw, dtype=float)
    if x.ndim != 2 or x.shape[1] != 6 or not np.isfinite(x).all():
        raise ValueError('Cần dữ liệu N × 6 hợp lệ.')
    if len(x) > 178:
        energy = np.sum(np.diff(x, axis=0)**2, axis=1)
        start = int(np.argmax(np.convolve(energy, np.ones(177), mode='valid')))
        x = x[start:start+178]
    src=np.linspace(0,1,len(x)); dst=np.linspace(0,1,64)
    wave=np.column_stack([np.interp(dst,src,x[:,i]) for i in range(6)])
    return np.concatenate([wave.ravel(), np.log1p(np.abs(np.fft.rfft(wave, axis=0))).ravel()]).astype('float32')
