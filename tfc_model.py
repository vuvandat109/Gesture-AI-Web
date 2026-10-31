"""TF-C (Zhang et al., NeurIPS 2022), six-channel adaptation.
Paper Eq. 1-4, explicit sample-independent Transformer batch axis.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CLASSES = ['up', 'down', 'left', 'right', 'circle']

def prepare_window(raw):
    x = np.asarray(raw, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 6 or not np.isfinite(x).all() or len(x)<20:
        raise ValueError('Expected finite N x 6 sensor data, N >= 20')
    if len(x)>178:
        energy=np.sum(np.diff(x,axis=0)**2,axis=1)
        start=int(np.argmax(np.convolve(energy,np.ones(177),mode='valid')))
        x=x[start:start+178]
    if len(x)!=178:
        src=np.linspace(0,1,len(x)); dst=np.linspace(0,1,178)
        x=np.stack([np.interp(dst,src,x[:,k]) for k in range(6)],axis=1).astype('float32')
    # Per-window statistics only: no train/test fitting or labels.
    x=(x-x.mean(axis=0,keepdims=True))/np.maximum(x.std(axis=0,keepdims=True),1e-6)
    return x.T.copy()

class TFC(nn.Module):
    def __init__(self, channels=6, length=178):
        super().__init__()
        self.channels=channels; self.length=length
        def encoder():
            layer=nn.TransformerEncoderLayer(d_model=length,nhead=2,
                dim_feedforward=2*length,dropout=.1,batch_first=True)
            return nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
        def projector():
            return nn.Sequential(nn.Linear(channels*length,256),nn.BatchNorm1d(256),
                                 nn.ReLU(),nn.Linear(256,128))
        self.time_encoder=encoder();self.frequency_encoder=encoder()
        self.time_projector=projector();self.frequency_projector=projector()
    def forward(self,t,f):
        ht=self.time_encoder(t).flatten(1); hf=self.frequency_encoder(f).flatten(1)
        return ht,self.time_projector(ht),hf,self.frequency_projector(hf)

class GestureTFC(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=TFC()
        self.classifier=nn.Sequential(nn.Linear(256,64),nn.Sigmoid(),nn.Linear(64,5))
    def forward(self,t):
        f=torch.fft.fft(t,dim=-1).abs()
        _,zt,_,zf=self.encoder(t,f)
        return self.classifier(torch.cat((zt,zf),dim=1))

def ntxent(a,b,temperature=.2):
    """Symmetric NT-Xent, self excluded, paired view is positive."""
    n=len(a);z=F.normalize(torch.cat((a,b)),dim=1)
    logits=z@z.T/temperature
    logits=logits.masked_fill(torch.eye(2*n,device=z.device,dtype=torch.bool),float('-inf'))
    target=(torch.arange(2*n,device=z.device)+n)%(2*n)
    return F.cross_entropy(logits,target)

def time_augmentation(x):
    # The four choices in paper Sec 4.1; preserve channel alignment.
    choice=int(torch.randint(4,()).item())
    if choice==0:return x+torch.randn_like(x)*.2
    if choice==1:return x*(1+torch.randn(len(x),x.shape[1],1,device=x.device)*.1)
    if choice==2:return torch.roll(x,int(torch.randint(-18,19,()).item()),dims=-1)
    parts=torch.tensor_split(torch.arange(x.shape[-1],device=x.device),4)
    order=torch.randperm(4);indices=torch.cat([parts[int(i)] for i in order])
    return x[:,:,indices]

def frequency_augmentation(f,ratio=.1,alpha=.5):
    # Paper Sec 4.2: choose adding OR removing small-budget frequency bins.
    k=max(1,int(f.shape[-1]*ratio));out=f.clone()
    if bool(torch.randint(2,()).item()):
        idx=torch.rand_like(f).argsort(dim=-1)[...,:k]
        out.scatter_(-1,idx,0.)
    else:
        amplitude=alpha*f.amax(dim=-1,keepdim=True)
        eligible=f<amplitude
        scores=torch.rand_like(f).masked_fill(~eligible,2.)
        idx=scores.argsort(dim=-1)[...,:k]
        values=torch.where(eligible.gather(-1,idx),amplitude.expand_as(f).gather(-1,idx),f.gather(-1,idx))
        out.scatter_(-1,idx,values)
    return out

def pretrain_objective(encoder,x,lam=.2,margin=1.):
    f=torch.fft.fft(x,dim=-1).abs();xt=time_augmentation(x);fa=frequency_augmentation(f)
    ht,zt,hf,zf=encoder(x,f);hat,zat,haf,zaf=encoder(xt,fa)
    lt=ntxent(ht,hat);lf=ntxent(hf,haf)
    stf=ntxent(zt,zf)
    lc=sum(stf-ntxent(a,b)+margin for a,b in [(zt,zaf),(zat,zf),(zat,zaf)])
    # Exactly the stated Eq. 3 (no unstated hinge) and Eq. 4.
    total=lam*(lt+lf)+(1-lam)*lc
    return total,torch.stack((lt,lf,lc)).detach()

class TFCInference:
    """Joblib-compatible adapter; accepts flattened normalized 6 x 178 windows."""
    def __init__(self,state):
        self.state={k:v.detach().cpu().numpy() for k,v in state.items()}
        self.classes_=np.array(CLASSES);self._net=None
    def _load(self):
        if self._net is None:
            torch.set_num_threads(1)
            net=GestureTFC();net.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in self.state.items()})
            self._net=net.eval()
        return self._net
    def predict_proba(self,x):
        x=np.asarray(x,dtype='float32').reshape(-1,6,178)
        with torch.inference_mode():return self._load()(torch.from_numpy(x)).softmax(1).numpy()
