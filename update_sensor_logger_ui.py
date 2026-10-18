"""Apply only the CSV/ZIP picker changes while preserving user UI customizations.
Run once from project root: python update_sensor_logger_ui.py
"""
from pathlib import Path

root = Path(__file__).resolve().parent
js = root / 'static' / 'app.js'
html = root / 'templates' / 'index.html'
if not (js.is_file() and html.is_file()):
    raise SystemExit('Khong thay static/app.js hoac templates/index.html. Hay copy script vao E:\\gesture_ai_web truoc.')

src = js.read_text(encoding='utf-8')
new = src.replace(r'/\.(csv|txt)$/i', r'/\.(csv|txt|zip)$/i')
new = new.replace('Chỉ hỗ trợ CSV hoặc TXT.', 'Chỉ hỗ trợ CSV, TXT hoặc ZIP Sensor Logger.')
new = new.replace('File vượt quá 2 MB.', 'File vượt quá 5 MB.')
new = new.replace('if(file.size>2*1024*1024)', 'if(file.size>5*1024*1024)')
if r'/\.(csv|txt|zip)$/i' not in new:
    raise SystemExit('Khong tim thay bo loc duoi file. Gui file static/app.js de kiem tra.')
if src != new:
    js.write_text(new, encoding='utf-8')
    print('Da cap nhat static/app.js: chap nhan CSV, TXT, ZIP.')
else:
    print('static/app.js da ho tro ZIP - khong can sua.')

src = html.read_text(encoding='utf-8')
new = src.replace('accept=".csv,.txt,text/csv,text/plain"', 'accept=".csv,.txt,.zip,text/csv,text/plain,application/zip"')
new = new.replace('CSV / TXT · ax, ay, az', 'CSV / ZIP Sensor Logger · gia tốc X/Y/Z')
if '.zip' not in new:
    raise SystemExit('Khong tim thay input file trong templates/index.html de them .zip.')
if src != new:
    html.write_text(new, encoding='utf-8')
    print('Da cap nhat templates/index.html: cho phep chon ZIP.')
else:
    print('templates/index.html da ho tro ZIP - khong can sua.')

print('HOAN THANH. Thu: python app.py')
