"""Gesture AI: Flask login + CSV inference using a real-data trained gesture model."""
import os
import secrets
import sqlite3
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request, session
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.exceptions import RequestEntityTooLarge
from ml import features, MODEL_KEYS, KEY_TO_LABEL
from account_store import AccountStore, DuplicateUsername
import psycopg
from sensor_input import read_accelerometer, MAX_UPLOAD_BYTES, ALLOWED_EXTENSIONS
from sensor_model import read_motion, motion_features
from tfc_model import prepare_window

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY') or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.environ.get('COOKIE_SECURE', '0') == '1',
)
accounts = AccountStore()
accounts.initialize()
if os.environ.get('RENDER') and accounts.backend == 'sqlite':
    app.logger.warning('DATABASE_URL missing: accounts are on ephemeral SQLite. Configure PostgreSQL for persistence.')
if os.environ.get('RENDER') and not os.environ.get('SECRET_KEY'):
    app.logger.warning('Set a stable SECRET_KEY for sessions across workers and restarts.')

MODEL_PATH = ROOT / 'models' / 'gesture_real_v1.joblib'
# Never silently fall back to the old synthetic model.
if MODEL_PATH.is_file():
    bundle = joblib.load(MODEL_PATH)
    MODEL = bundle['model']
    INFO = bundle['info']
else:
    MODEL = None
    INFO = {'model_name': 'Chưa huấn luyện', 'trained_from': 'NONE'}


def token():
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_urlsafe(32)
    return session['_csrf_token']

@app.before_request
def check_csrf():
    if request.method in {'POST', 'PUT', 'DELETE'}:
        if not secrets.compare_digest(str(request.headers.get('X-CSRF-Token', '')), str(token())):
            return jsonify(error='Phiên gửi dữ liệu không hợp lệ, vui lòng tải lại trang.'), 403

@app.errorhandler(psycopg.OperationalError)
@app.errorhandler(sqlite3.OperationalError)
def database_unavailable(error):
    app.logger.error('Account storage temporarily unavailable')
    return jsonify(error='Không thể kết nối nơi lưu tài khoản. Vui lòng thử lại sau.'), 503

@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    return jsonify(error='Tệp lớn hơn 5 MB.'), 413

@app.get('/')
def index():
    return render_template('index.html', csrf_token=token(), username=session.get('username'), model_name=INFO['model_name'], model_ready=MODEL is not None, account_backend=accounts.backend, test_accuracy=round(INFO.get('heldout_test_accuracy', 0)*100, 1) if MODEL is not None else None)

@app.get('/api/health')
def health():
    return jsonify(status='ok', account_storage=accounts.backend, model_ready=MODEL is not None, model=INFO['model_name'], training_source=INFO['trained_from'], heldout_test_accuracy=INFO.get('heldout_test_accuracy'))

@app.post('/api/register')
def register():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error='Dữ liệu gửi lên phải là JSON object.'), 400
    if not isinstance(data.get('username'), str) or not isinstance(data.get('password'), str):
        return jsonify(error='Tên tài khoản và mật khẩu phải là chuỗi ký tự.'), 400
    username = data['username'].strip()
    password = data['password']
    if not (3 <= len(username) <= 32 and username.replace('_','').isalnum()):
        return jsonify(error='Tên tài khoản phải dài 3–32 ký tự, chỉ dùng chữ, số hoặc dấu gạch dưới.'), 400
    if not 8 <= len(password) <= 128:
        return jsonify(error='Mật khẩu phải dài 8–128 ký tự.'), 400
    try:
        accounts.create(username, generate_password_hash(password))
    except DuplicateUsername:
        return jsonify(error='Tên tài khoản đã được sử dụng.'), 409
    session.clear()
    return jsonify(registered=True, username=username, next='login', csrf=token()), 201

@app.post('/api/login')
def login():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error='Dữ liệu gửi lên phải là JSON object.'), 400
    if not isinstance(data.get('username'), str) or not isinstance(data.get('password'), str):
        return jsonify(error='Tên tài khoản và mật khẩu phải là chuỗi ký tự.'), 400
    username = data['username'].strip()
    password = data['password']
    if len(username) > 32 or not 8 <= len(password) <= 128:
        return jsonify(error='Sai tên đăng nhập hoặc mật khẩu.'), 401
    record = accounts.find(username)
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
        tfc = INFO.get('feature_version') == 'tfc6_v1'
        sensor6 = tfc or INFO.get('feature_version') == 'sensor6_v1'
        reader = read_motion if sensor6 else read_accelerometer
        extractor = prepare_window if tfc else (motion_features if sensor6 else features)
        labels = bundle.get('labels', KEY_TO_LABEL)
        values, data_source = reader(uploaded.read(), uploaded.filename)
        probabilities = MODEL.predict_proba(extractor(values).reshape(1,-1))[0]
        predictions = sorted(({
            'key': str(c), 'label': labels[str(c)],
            'probability': round(float(p)*100, 2)
        } for c, p in zip(MODEL.classes_, probabilities)), key=lambda x: -x['probability'])
        return jsonify(
            prediction=predictions[0]['label'], gesture_key=predictions[0]['key'],
            confidence=predictions[0]['probability'], probabilities=predictions,
            rows=len(values), model_name=INFO['model_name'], real_data=True,
            input_format=data_source, gyro_used_by_model=sensor6,
            caveat=INFO.get('evaluation', 'Khác biệt thiết bị có thể ảnh hưởng dự đoán.')
        )
    except (ValueError, TypeError) as error:
        return jsonify(error=f'Dữ liệu không hợp lệ: {error}'), 400
    except Exception:
        app.logger.exception('Gesture inference failed')
        return jsonify(error='Không thể chạy mô hình. Kiểm tra log trên Render để biết lỗi chi tiết.'), 500

if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1', host='0.0.0.0', port=int(os.environ.get('PORT', '5000')))
