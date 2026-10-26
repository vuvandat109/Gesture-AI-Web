"""Train on the 26,000-window archive: unlabeled PCA + supervised classifier.
Usage: python train_sensor_model.py --data PATH_TO_ZIP_OR_DATASET_DIRECTORY
"""
import argparse, io, json, re, zipfile, hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sensor_model import CHANNELS, motion_features

ROOT=Path(__file__).resolve().parent

def train(data, output):
    path=Path(data)
    archive=zipfile.ZipFile(path) if path.is_file() else None
    names=archive.namelist() if archive else [str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()]
    roots=[n[:-len('fine_tuning/index.csv')] for n in names if n.endswith('fine_tuning/index.csv')]
    if len(roots)!=1: raise ValueError('Expected exactly one fine_tuning/index.csv')
    prefix=roots[0]
    def read(n): return archive.read(prefix+n) if archive else (path/(prefix+n)).read_bytes()
    def table(n): return pd.read_csv(io.BytesIO(read(n)))
    pre=table('pretraining/index.csv'); fine=table('fine_tuning/index.csv')
    if (len(pre),len(fine))!=(20800,5200): raise ValueError('Expected 20,800 unlabeled and 5,200 labeled windows')
    if set(fine.label)!=set(['up','down','left','right','rotate']) or not (fine.label.value_counts()==1040).all():
        raise ValueError('Expected 1,040 windows per gesture')
    manifest=table('provenance_manifest.csv').set_index('sample_id')
    parents={k: (re.search(r'parent=([^|]+)',str(v)).group(1) if re.search(r'parent=([^|]+)',str(v)) else k) for k,v in manifest.data_origin.items()}
    def ancestor(k):
        seen=set()
        while parents.get(k,k)!=k:
            if k in seen: raise ValueError('Cyclic provenance')
            seen.add(k); k=parents[k]
        return k
    groups=np.array([ancestor(k) for k in fine.sample_id])
    y=fine.label.replace({'rotate':'circle'}).to_numpy()
    print('Reading 20,800 unlabeled windows...',flush=True)
    def featurize(index, folder):
        result=[]
        for row in index.itertuples():
            f=table(folder+'/'+row.file)
            if len(f)!=178: raise ValueError('Expected 178 time steps')
            result.append(motion_features(f[CHANNELS].to_numpy(float)))
        return np.stack(result)
    up=featurize(pre,'pretraining'); x=featurize(fine,'fine_tuning')
    # Do not use manifest true_label for representation learning.
    scaler=StandardScaler().fit(up)
    pca=PCA(n_components=64,random_state=42).fit(scaler.transform(up))
    z=pca.transform(scaler.transform(x))
    trainval,test=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(z,y,groups))
    fit0,val0=next(StratifiedGroupKFold(4,shuffle=True,random_state=43).split(z[trainval],y[trainval],groups[trainval]))
    fit,val=trainval[fit0],trainval[val0]
    scores={}; candidates={}
    for c in [0.1,1.,10.]:
        m=LogisticRegression(C=c,max_iter=2000).fit(z[fit],y[fit]); candidates[c]=m
        scores[c]=float(accuracy_score(y[val],m.predict(z[val])))
    best=max(scores,key=scores.get)
    clf=LogisticRegression(C=best,max_iter=2000).fit(z[trainval],y[trainval])
    pred=clf.predict(z[test])
    labels={'up':'Vuốt lên','down':'Vuốt xuống','left':'Vuốt trái','right':'Vuốt phải','circle':'Xoay cổ tay'}
    info={'model_name':'Sensor Logger PCA + Logistic Regression (6 channels)',
          'trained_from':'Sensor Logger iPhone: 26,000 processed/augmented windows',
          'feature_version':'sensor6_v1','pretraining_samples':len(pre),'labeled_samples':len(fine),
          'train_samples':len(trainval),'test_samples':len(test),'validation_accuracies':scores,
          'heldout_test_accuracy':float(accuracy_score(y[test],pred)),
          'heldout_test_macro_f1':float(f1_score(y[test],pred,average='macro')),
          'test_confusion_matrix':confusion_matrix(y[test],pred,labels=list(labels)).tolist(),
          'confusion_matrix_labels':list(labels),'selected_C':best,
          'evaluation':'Group split by augmentation ancestor; NOT independent user/session evaluation. Original recording IDs unavailable; cross-partition ancestry cannot be fully verified.',
          'method':'PCA representation fitted only on 20,800 unlabeled windows; logistic classifier trained only on labeled training groups. Not TF-C or DINOv3.',
          'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest() if archive else None}
    model=Pipeline([('scale',scaler),('pca',pca),('classifier',clf)])
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    joblib.dump({'model':model,'labels':labels,'keys':list(labels),'info':info},output/'gesture_real_v1.joblib',compress=3)
    (output/'training_report.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame({'sample_id':fine.sample_id,'ancestor':groups,'split':np.where(np.isin(np.arange(len(fine)),test),'test','train')}).to_csv(output/'evaluation_split.csv',index=False)
    print(json.dumps(info,indent=2,ensure_ascii=False))
    if archive: archive.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--output',default=str(ROOT/'models'))
    args=p.parse_args(); train(args.data,args.output)
