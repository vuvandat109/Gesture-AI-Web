"""Gesture AI demo: Flask site, local accounts, CSV inference, synthetic Logistic Regression."""
import io
import os
import secrets
import sqlite3
from pathlib import Path
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, session, g
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.exceptions import RequestEntityTooLarge
from ml import features

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY') or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=2 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.environ.get('COOKIE_SECURE', '0') == '1',
)
DB = Path(os.environ.get('DATABASE_PATH', str(ROOT / 'instance' / 'users.db')))
DB.parent.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(DB) as connection:
    connection.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL)')

bundle = joblib.load(ROOT / 'models' / 'logistic_gesture_demo.joblib')
MODEL = bundle['model']
LABELS = bundle['labels']


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
    return jsonify(error='Tệp lớn hơn 2 MB.'), 413

@app.get('/')
def index():
    return render_template('index.html', csrf_token=token(), username=session.get('username'))

@app.get('/api/health')
def health():
    return jsonify(status='ok', model='Logistic Regression (synthetic demo)')

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
    uploaded = request.files.get('file')
    if uploaded is None or not uploaded.filename:
        return jsonify(error='Vui lòng chọn file CSV hoặc TXT.'), 400
    if Path(uploaded.filename).suffix.lower() not in {'.csv','.txt'}:
        return jsonify(error='Chỉ hỗ trợ CSV hoặc TXT.'), 400
    raw_bytes = uploaded.read()
    if len(raw_bytes) > 2 * 1024 * 1024:
        return jsonify(error='File vượt quá 2 MB.'), 413
    try:
        content = raw_bytes.decode('utf-8-sig')
        # supports comma, semicolon, whitespace/tab formats
        frame = pd.read_csv(io.StringIO(content), sep=None, engine='python')
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        groups = [('ax','ay','az'),('acc_x','acc_y','acc_z'),('x','y','z')]
        found = next((list(names) for names in groups if all(c in frame.columns for c in names)), None)
        if found is None:
            # No header: try read as 3 numeric columns
            frame = pd.read_csv(io.StringIO(content), sep=r'[\s,;]+', engine='python', header=None)
            if frame.shape[1] < 3:
                raise ValueError('Cần ba cột ax, ay, az (hoặc x, y, z).')
            frame = frame.iloc[:, :3]
        else:
            frame = frame[found]
        if not 20 <= len(frame) <= 10000:
            raise ValueError('Số dòng dữ liệu cần từ 20 đến 10.000.')
        values = frame.apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Dữ liệu có NaN/Inf.')
        probabilities = MODEL.predict_proba(features(values).reshape(1,-1))[0]
        predictions = sorted(({'label': LABELS[int(c)], 'probability': round(float(p)*100, 2)} for c,p in zip(MODEL.classes_, probabilities)), key=lambda x:-x['probability'])
        return jsonify(prediction=predictions[0]['label'], confidence=predictions[0]['probability'], probabilities=predictions, rows=len(values), demo=True)
    except (ValueError, TypeError, pd.errors.ParserError, UnicodeDecodeError) as error:
        return jsonify(error=f'Dữ liệu không hợp lệ: {error}'), 400

if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1', host='0.0.0.0', port=int(os.environ.get('PORT', '5000')))
