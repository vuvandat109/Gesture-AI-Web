"""Gesture AI: Flask login + CSV inference using a real-data trained gesture model."""
import os
import secrets
import sqlite3
from pathlib import Path
from functools import wraps

import joblib
from flask import Flask, jsonify, render_template, request, session, g
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.exceptions import RequestEntityTooLarge
from ml import features, MODEL_KEYS, KEY_TO_LABEL
from sensor_input import read_accelerometer, MAX_UPLOAD_BYTES, ALLOWED_EXTENSIONS

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY') or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.environ.get('COOKIE_SECURE', '0') == '1',
)
DB = Path(os.environ.get('DATABASE_PATH', str(ROOT / 'instance' / 'users.db')))
DB.parent.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(DB) as connection:
    connection.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL)')

MODEL_PATH = ROOT / 'models' / 'gesture_real_v1.joblib'
# Never silently fall back to the old synthetic model.
if MODEL_PATH.is_file():
    bundle = joblib.load(MODEL_PATH)
    MODEL = bundle['model']
    INFO = bundle['info']
else:
    MODEL = None
    INFO = {'model_name': 'Chưa huấn luyện', 'trained_from': 'NONE'}


def db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB, timeout=10)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    conn = g.pop('db', None)
    if conn is not None:
        conn.close()


def token():
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_urlsafe(32)
    return session['_csrf_token']

@app.before_request
def check_csrf():
    if request.method in {'POST', 'PUT', 'DELETE'}:
        if not secrets.compare_digest(str(request.headers.get('X-CSRF-Token', '')), str(token())):
            return jsonify(error='Phiên gửi dữ liệu không hợp lệ, vui lòng tải lại trang.'), 403

@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    return jsonify(error='Tệp lớn hơn 5 MB.'), 413

@app.get('/')
def index():
    return render_template('index.html', csrf_token=token(), username=session.get('username'), model_name=INFO['model_name'], model_ready=MODEL is not None, test_accuracy=round(INFO.get('heldout_test_accuracy', 0)*100, 1) if MODEL is not None else None)

@app.get('/api/health')
def health():
    return jsonify(status='ok', model_ready=MODEL is not None, model=INFO['model_name'], training_source=INFO['trained_from'], heldout_test_accuracy=INFO.get('heldout_test_accuracy'))

@app.post('/api/register')
def register():
    data = request.get_json(silent=True) or {}
    username = str(data.get('username', '')).strip()
    password = str(data.get('password', ''))
    if not (3 <= len(username) <= 32 and username.replace('_','').isalnum()):
        return jsonify(error='Tên tài khoản phải dài 3–32 ký tự, chỉ dùng chữ, số hoặc dấu gạch dưới.'), 400
    if not 8 <= len(password) <= 128:
        return jsonify(error='Mật khẩu phải có ít nhất 8 ký tự.'), 400
    try:
        with db():
            db().execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, generate_password_hash(password)))
    except sqlite3.IntegrityError:
        return jsonify(error='Tên tài khoản đã được sử dụng.'), 409
    session.clear()
    session.update(username=username, _csrf_token=secrets.token_urlsafe(32))
    return jsonify(username=username, csrf=session['_csrf_token'])

@app.post('/api/login')
def login():
    data = request.get_json(silent=True) or {}
    username = str(data.get('username','')).strip()
    password = str(data.get('password',''))
    record = db().execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    if not record or not check_password_hash(record['password_hash'], password):
        return jsonify(error='Sai tên đăng nhập hoặc mật khẩu.'), 401
    session.clear()
    session.update(username=username, _csrf_token=secrets.token_urlsafe(32))
    return jsonify(username=username, csrf=session['_csrf_token'])

@app.post('/api/logout')
def logout():
    session.clear()
    return jsonify(ok=True, csrf=token())

@app.post('/api/predict')
def predict():
    if not session.get('username'):
        return jsonify(error='Bạn cần đăng nhập để chạy dự đoán.'), 401
    if MODEL is None:
        return jsonify(error='Chưa có model dữ liệu thật. Hãy chạy: python train_real_model.py rồi deploy lại.'), 503
    uploaded = request.files.get('file')
    if uploaded is None or not uploaded.filename:
        return jsonify(error='Vui lòng chọn file CSV, TXT hoặc ZIP Sensor Logger.'), 400
    if Path(uploaded.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify(error='Chỉ hỗ trợ CSV, TXT hoặc ZIP Sensor Logger.'), 400
    try:
        values, data_source = read_accelerometer(uploaded.read(), uploaded.filename)
        probabilities = MODEL.predict_proba(features(values).reshape(1,-1))[0]
        predictions = sorted(({
            'key': str(c), 'label': KEY_TO_LABEL[str(c)],
            'probability': round(float(p)*100, 2)
        } for c, p in zip(MODEL.classes_, probabilities)), key=lambda x: -x['probability'])
        return jsonify(
            prediction=predictions[0]['label'], gesture_key=predictions[0]['key'],
            confidence=predictions[0]['probability'], probabilities=predictions,
            rows=len(values), model_name=INFO['model_name'], real_data=True,
            input_format=data_source, gyro_used_by_model=False,
            caveat='Lớp xoay vòng tay là cử chỉ vẽ vòng trong không gian, không phải đo riêng xoay cổ tay.'
        )
    except (ValueError, TypeError) as error:
        return jsonify(error=f'Dữ liệu không hợp lệ: {error}'), 400
    except Exception:
        app.logger.exception('Gesture inference failed')
        return jsonify(error='Không thể chạy mô hình. Kiểm tra log trên Render để biết lỗi chi tiết.'), 500

if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1', host='0.0.0.0', port=int(os.environ.get('PORT', '5000')))
