"""Train the demonstration Logistic Regression on synthetic signals only."""
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml import features, synthesize, LABELS

rng = np.random.default_rng(2026)
X, y = [], []
for class_id in range(len(LABELS)):
    for _ in range(260):
        length = int(rng.integers(70, 201))
        X.append(features(synthesize(class_id, rng, length)))
        y.append(class_id)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.8))
model.fit(np.asarray(X), np.asarray(y))
output = Path(__file__).parent / 'models' / 'logistic_gesture_demo.joblib'
output.parent.mkdir(exist_ok=True)
joblib.dump({'model':model, 'labels':LABELS, 'training_source':'synthetic_demo_2026'}, output)
print('Created demo model:', output)
# files used for demonstration, not real sensor collection
folder = Path(__file__).parent / 'samples'
folder.mkdir(exist_ok=True)
for class_id, filename in [(0,'vuot_phai_demo.csv'), (4,'xoay_co_tay_demo.csv')]:
    signal = synthesize(class_id, rng, 150)
    import csv
    with (folder/filename).open('w', encoding='utf-8', newline='') as f:
        writer=csv.writer(f); writer.writerow(['ax','ay','az']); writer.writerows(signal)
print('Created 2 synthetic example CSVs')
