"""Read real iPhone Sensor Logger accelerometer data for Gesture AI.

Accepts:
- processed CSV: timestamp,acc_x,acc_y,acc_z,gyro_x,...,label,...
- raw Sensor Logger Accelerometer.csv: time,seconds_elapsed,z,y,x
- Sensor Logger ZIP archive containing Accelerometer.csv (and possibly Gyroscope.csv)
- model's original three-axis CSV/TXT files.

IMPORTANT: the deployed SVM was trained on three accelerometer axes only. This
reader deliberately does not pretend to use gyroscope for predictions.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_SENSOR_CSV_BYTES = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.csv', '.txt', '.zip'}


def _norm(name: object) -> str:
    """Normalize varied column styles without changing the numeric data."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower().lstrip('\ufeff'))


def _extract_zip_csv(data: bytes) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if len(archive.infolist()) > 100:
                raise ValueError('File ZIP chứa quá nhiều tệp.')
            entries = [z for z in archive.infolist() if not z.is_dir() and
                       Path(z.filename.replace('\\', '/')).name.lower() == 'accelerometer.csv']
            if len(entries) != 1:
                raise ValueError('ZIP Sensor Logger phải có đúng một file Accelerometer.csv. Hãy Export → CSV in Zip File.')
            entry = entries[0]
            if entry.file_size > MAX_SENSOR_CSV_BYTES:
                raise ValueError('Accelerometer.csv trong ZIP vượt quá 5 MB.')
            # Check archive member CRC; do not extract directories/paths.
            content = archive.read(entry)
            if len(content) > MAX_SENSOR_CSV_BYTES:
                raise ValueError('Accelerometer.csv quá lớn.')
            return content
    except (zipfile.BadZipFile, RuntimeError, EOFError) as exc:
        raise ValueError('File ZIP không hợp lệ hoặc bị hỏng.') from exc


def _read_table(data: bytes) -> pd.DataFrame:
    try:
        content = data.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('CSV phải được xuất ở định dạng UTF-8.') from exc
    if not content.strip():
        raise ValueError('File CSV trống.')

    # Sensor Logger uses comma; UWave examples may be comma, tab, or whitespace.
    first_line = next((line for line in content.splitlines() if line.strip()), '')
    if not any(d in first_line for d in (',', ';', '\t')):
        frame = pd.read_csv(io.StringIO(content), sep=r'\s+', engine='python')
    else:
        frame = pd.read_csv(io.StringIO(content), sep=None, engine='python')

    if frame.shape[1] < 3:
        raise ValueError('Không tìm thấy đủ ba cột dữ liệu gia tốc X, Y, Z.')
    # For legacy three-column files without a header, pandas has interpreted
    # numeric first row as a header. Read again to preserve that sample.
    first_three = [str(name).strip() for name in frame.columns[:3]]
    try:
        if all(np.isfinite(float(name)) for name in first_three):
            if not any(d in first_line for d in (',', ';', '\t')):
                frame = pd.read_csv(io.StringIO(content), sep=r'\s+', engine='python', header=None)
            else:
                frame = pd.read_csv(io.StringIO(content), sep=None, engine='python', header=None)
    except (TypeError, ValueError):
        pass
    return frame


def read_accelerometer(uploaded_bytes: bytes, filename: str) -> tuple[np.ndarray, str]:
    """Return finite Nx3 accelerometer data, in X, Y, Z order.

    Extra CSV columns (timestamp, gyro_*, label, trial_id) are ignored.
    Returns a data-source label for the API/UI, not for ML features.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError('Chỉ hỗ trợ CSV, TXT hoặc ZIP xuất từ Sensor Logger.')
    if not uploaded_bytes:
        raise ValueError('File tải lên trống.')
    if len(uploaded_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError('File tải lên vượt quá 5 MB.')

    data = _extract_zip_csv(uploaded_bytes) if suffix == '.zip' else uploaded_bytes
    frame = _read_table(data)
    clean_names = [_norm(name) for name in frame.columns]
    if len(clean_names) != len(set(clean_names)):
        raise ValueError('CSV chứa các tên cột bị trùng sau chuẩn hóa.')
    frame.columns = clean_names

    groups = (
        ('accx', 'accy', 'accz'),
        ('ax', 'ay', 'az'),
        ('accelerationx', 'accelerationy', 'accelerationz'),
        ('linearaccelerationx', 'linearaccelerationy', 'linearaccelerationz'),
        ('useraccelerationx', 'useraccelerationy', 'useraccelerationz'),
        ('x', 'y', 'z'),  # Sensor Logger: time,seconds_elapsed,z,y,x
    )
    cols = next((g for g in groups if all(c in frame.columns for c in g)), None)
    if cols is not None:
        xyz = frame.loc[:, list(cols)]
    elif all(str(c).isdigit() for c in frame.columns) and frame.shape[1] >= 3:
        xyz = frame.iloc[:, :3]
    else:
        raise ValueError(
            'Không thấy gia tốc X/Y/Z. Cần các cột acc_x,acc_y,acc_z '
            'hoặc Accelerometer.csv với x,y,z. '
            f'Cột hiện có: {", ".join(frame.columns[:12])}.'
        )
    if not 20 <= len(xyz) <= 10000:
        raise ValueError(f'Cần 20–10.000 dòng dữ liệu gia tốc; file hiện có {len(xyz)} dòng.')
    try:
        values = xyz.apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError('Gia tốc X/Y/Z phải là các giá trị số; kiểm tra CSV của bạn.') from exc
    if values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError('Gia tốc X/Y/Z chứa giá trị trống, NaN hoặc vô cực.')
    source = 'Sensor Logger ZIP' if suffix == '.zip' else 'CSV/TXT gia tốc'
    return values, source
