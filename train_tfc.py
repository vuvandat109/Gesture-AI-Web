"""Paper TF-C: unlabeled pre-training -> supervised full-network fine-tuning.
Run: python train_tfc.py --data archive.zip --pre-epochs 40 --fine-epochs 40
"""
import argparse,copy,hashlib,io,json,random,re,time,zipfile
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from tfc_model import GestureTFC,TFCInference,pretrain_objective,prepare_window,CLASSES
ROOT=Path(__file__).resolve().parent

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def weight_change(before,after):
 changed=[k for k in before if not torch.equal(before[k],after[k])]
 return {'changed_tensors':len(changed),'total_tensors':len(before),
         'l2_delta':float(sum((before[k].float()-after[k].float()).square().sum() for k in before).sqrt())}
def prepare(data,out):
 z=zipfile.ZipFile(data);root=next(n[:-len('fine_tuning/index.csv')] for n in z.namelist() if n.endswith('fine_tuning/index.csv'))
 def read(p):return z.read(root+p)
 def frame(p):return pd.read_csv(io.BytesIO(read(p)))
 pre=frame('pretraining/index.csv');fine=frame('fine_tuning/index.csv')
 assert len(pre)==20800 and len(fine)==5200
 assert fine.label.value_counts().to_dict()=={k:1040 for k in ['up','down','left','right','rotate']}
 xp=np.load(io.BytesIO(read('pretraining/X_time_zscore.npy')));xf=np.load(io.BytesIO(read('fine_tuning/X_time_zscore.npy')))
 assert xp.shape==(20800,178,6) and xf.shape==(5200,178,6)
 norm=json.loads(read('normalization.json'));channels=norm['channels'];checks=[]
 for part,idx,x in [('pretraining',pre,xp),('fine_tuning',fine,xf)]:
  for i in np.linspace(0,len(idx)-1,12,dtype=int):
   raw=frame(part+'/'+idx.file.iloc[i])[channels].to_numpy()
   error=float(np.max(np.abs((raw-norm['mean'])/norm['std']-x[i])))
   assert error<1e-4,'CSV and NPY normalization mismatch'
   checks.append(error)
 # Normalize each window identically for train/serve. Global affine cancels out.
 xp=np.stack([prepare_window(x) for x in xp]);xf=np.stack([prepare_window(x) for x in xf])
 y=np.array([CLASSES.index('circle' if v=='rotate' else v) for v in fine.label])
 manifest=frame('provenance_manifest.csv');parents={}
 for row in manifest.itertuples():
  m=re.search(r'parent=([^|]+)',str(row.data_origin));parents[row.sample_id]=m.group(1) if m else row.sample_id
 def ancestor(k):
  seen=set()
  while parents.get(k,k)!=k:
   if k in seen:raise ValueError('Provenance cycle')
   seen.add(k);k=parents[k]
  return k
 groups=np.array([ancestor(k) for k in fine.sample_id]);pg=np.array([ancestor(k) for k in pre.sample_id])
 trainval,test=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(xf,y,groups))
 a,b=next(StratifiedGroupKFold(4,shuffle=True,random_state=43).split(xf[trainval],y[trainval],groups[trainval]))
 train,val=trainval[a],trainval[b]
 assert not set(groups[train])&set(groups[val]) and not set(groups[trainval])&set(groups[test])
 # Reject exact raw-array duplicates between pre-training and held-out target.
 hashes=lambda arr:[hashlib.sha256(v.tobytes()).hexdigest() for v in arr]
 hp=hashes(xp);hf=hashes(xf)
 overlap=len(set(hp)&{hf[i] for i in np.r_[val,test]})
 assert overlap==0,'Exact pretrain/heldout duplicate; resolve source before training'
 # Known shared ancestor is forbidden, not silently ignored.
 assert not set(pg)&set(groups[np.r_[val,test]]),'Known pretraining/heldout ancestry overlap'
 split=np.full(len(fine),'train',dtype=object);split[val]='validation';split[test]='test'
 pd.DataFrame({'sample_id':fine.sample_id,'ancestor':groups,'split':split}).to_csv(out/'evaluation_split.csv',index=False)
 audit={'source_filename':Path(data).name,'data_sha256':digest(data),'pretraining_samples':len(pre),'labeled_samples':len(fine),'train_samples':len(train),'validation_samples':len(val),'test_samples':len(test),'array_shape':[178,6],'csv_npy_spotchecks':len(checks),'max_spotcheck_error':max(checks),'exact_pretrain_heldout_duplicates':overlap,'known_ancestor_groups_disjoint':True,'source_limitation':'Augmented/synthetic-derived windows. Original recording/session IDs absent; unknown shared source ancestry cannot be ruled out. Not an independent-person/session evaluation.','kaggle_url':'https://www.kaggle.com/datasets/vuvandat000099/sensor-logger-dataset','source_selection':'User approved previously uploaded Sensor logger(1).zip because Kaggle v7 reports zero bytes and download returned 404.'}
 (out/'data_audit.json').write_text(json.dumps(audit,indent=2));return xp,xf,y,train,val,test,audit

def evaluate(net,x,y,batch=128):
 net.eval();p=[]
 with torch.inference_mode():
  for v in torch.from_numpy(x).split(batch):p.append(net(v).softmax(1).numpy())
 p=np.concatenate(p);pred=p.argmax(1)
 return float(accuracy_score(y,pred)),float(f1_score(y,pred,average='macro')),p

def train(args):
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(args.threads)
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 xp,x,y,fit,val,test,audit=prepare(args.data,out)
 print('DATA_AUDIT',json.dumps(audit),flush=True)
 net=GestureTFC();initial=copy.deepcopy(net.encoder.state_dict());opt=torch.optim.Adam(net.encoder.parameters(),lr=3e-4,betas=(.9,.99),weight_decay=0)
 gen=torch.Generator().manual_seed(42)
 dl=DataLoader(torch.from_numpy(xp),batch_size=128,shuffle=True,drop_last=False,generator=gen)
 log=[];started=time.time()
 def record(r):
  log.append(r);(out/'training_history.json').write_text(json.dumps(log,indent=2));print(json.dumps(r),flush=True)
 for epoch in range(1,args.pre_epochs+1):
  net.encoder.train();s=np.zeros(4);count=0;t=time.time()
  for xb in dl:
   if len(xb)<2:continue
   opt.zero_grad(set_to_none=True);loss,parts=pretrain_objective(net.encoder,xb)
   if not torch.isfinite(loss):raise ValueError('Nonfinite pretrain loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(net.encoder.parameters(),5.);opt.step()
   s+=np.r_[loss.item(),parts.numpy()]*len(xb);count+=len(xb)
  record({'stage':'pretrain','epoch':epoch,'samples_seen':count,'total_loss':s[0]/count,'time_loss':s[1]/count,'frequency_loss':s[2]/count,'consistency_loss':s[3]/count,'seconds':time.time()-t})
  torch.save({'model_state_dict':net.encoder.state_dict(),'optimizer_state_dict':opt.state_dict(),'epoch':epoch,'seed':42},out/'tfc_pretrained.pt')
 prestate=copy.deepcopy(net.encoder.state_dict());changes={'pretraining':weight_change(initial,prestate)}
 assert changes['pretraining']['l2_delta']>0
 opt=torch.optim.Adam(net.parameters(),lr=3e-4,betas=(.9,.99))
 dl=DataLoader(TensorDataset(torch.from_numpy(x[fit]),torch.from_numpy(y[fit])),batch_size=128,shuffle=True,generator=gen)
 best=-1;bestepoch=0;beststate=None
 for epoch in range(1,args.fine_epochs+1):
  net.train();total=0;count=0;t=time.time()
  for xb,yb in dl:
   opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(net(xb),yb)
   assert torch.isfinite(loss)
   loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();total+=loss.item()*len(xb);count+=len(xb)
  acc,f1,_=evaluate(net,x[val],y[val])
  record({'stage':'fine_tune','epoch':epoch,'train_ce':total/count,'validation_accuracy':acc,'validation_macro_f1':f1,'seconds':time.time()-t})
  if f1>best:
   best=f1;bestepoch=epoch;beststate=copy.deepcopy(net.state_dict())
   torch.save({'model_state_dict':beststate,'epoch':epoch,'validation_macro_f1':f1,'classes':CLASSES},out/'tfc_finetuned.pt')
 torch.save({'model_state_dict':prestate,'epoch':args.pre_epochs,'seed':42},out/'tfc_pretrained.pt')
 net.load_state_dict(beststate);changes['finetuning_encoder']=weight_change(prestate,net.encoder.state_dict());assert changes['finetuning_encoder']['l2_delta']>0
 # Single final held-out evaluation; no test-driven epoch selection.
 acc,f1,probs=evaluate(net,x[test],y[test]);pred=probs.argmax(1)
 info={**audit,'model_name':'TF-C Time-Frequency Consistency (6-channel adaptation)','feature_version':'tfc6_v1','trained_from':'Sensor Logger: 26,000 windows; neural TF-C pretraining + full-network fine-tuning','method':'Paper Eq. 1-4 NT-Xent + consistency loss; CE supervised fine-tuning of both encoders/projectors and MLP. No PCA/LogisticRegression/DINOv3.','paper':'Zhang et al., NeurIPS 2022, arXiv:2206.08496','official_code_commit':'5096675826e9ef234a9b01cc63d484c66cb0441bc0','pretraining_epochs':args.pre_epochs,'fine_tuning_epochs':args.fine_epochs,'selected_epoch':bestepoch,'validation_macro_f1':best,'heldout_test_accuracy':acc,'heldout_test_macro_f1':f1,'test_confusion_matrix':confusion_matrix(y[test],pred,labels=range(5)).tolist(),'confusion_matrix_labels':CLASSES,'evaluation':audit['source_limitation'],'seed':42,'batch_size':128,'lambda':.2,'temperature':.2,'margin':1.,'learning_rate':.0003,'device':'cpu','torch_version':torch.__version__,'elapsed_seconds':time.time()-started,'weight_update_proof':changes,'pretrained_sha256':digest(out/'tfc_pretrained.pt'),'finetuned_sha256':digest(out/'tfc_finetuned.pt'),'adaptations':['All six channels retained; Transformer batch_first=True and flatten 6*178 projector input.','Paper Eq3/Eq4 used, unlike upstream trainer that computes loss_c but optimizes l_TF.','Fresh augmentations each batch; per-window z-score; 10% frequency perturbation, alpha=.5.','Temporal bank: jitter sigma=.2, scale SD=.1, circular shifts +/-18, four-segment permutation.','40/40 epochs from upstream default; batch128; full labeled train split, no few-shot benchmark claim.','No test checkpoint selection or upstream cross-batch attention.']}
 labels={'up':'Vuốt lên','down':'Vuốt xuống','left':'Vuốt trái','right':'Vuốt phải','circle':'Xoay cổ tay'}
 joblib.dump({'model':TFCInference(net.state_dict()),'labels':labels,'keys':CLASSES,'info':info},out/'gesture_real_v1.joblib',compress=3)
 (out/'training_report.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
 (out/'classification_report.json').write_text(json.dumps(classification_report(y[test],pred,target_names=CLASSES,output_dict=True),indent=2))
 np.savez_compressed(out/'test_predictions.npz',indices=test,labels=y[test],probabilities=probs)
 print('FINAL_RESULT',json.dumps(info,ensure_ascii=False),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',default=str(ROOT/'models'));p.add_argument('--pre-epochs',type=int,default=40);p.add_argument('--fine-epochs',type=int,default=40);p.add_argument('--threads',type=int,default=4);train(p.parse_args())
