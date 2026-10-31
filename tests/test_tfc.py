"""Regressions for sample isolation, paper objective and neural artifact."""
import sys,unittest,io
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,joblib
from unittest.mock import patch
from tfc_model import TFC,GestureTFC,TFCInference,prepare_window,ntxent,pretrain_objective

torch.set_num_threads(1)
class TFCTests(unittest.TestCase):
 def test_inference_independent_of_other_batch_samples(self):
  torch.manual_seed(7);net=GestureTFC().eval();x=torch.randn(4,6,178)
  with torch.no_grad():
   alone=net(x[:1]);batch=net(x)[:1]
  torch.testing.assert_close(alone,batch,atol=2e-6,rtol=2e-5)
 def test_raw_and_stored_affine_normalization_agree(self):
  raw=np.random.default_rng(1).normal(size=(178,6));mean=np.arange(6);sd=np.arange(6)+.5
  np.testing.assert_allclose(prepare_window(raw),prepare_window((raw-mean)/sd),atol=3e-6)
 def test_ntxent_positive_matching_beats_misalignment(self):
  x=torch.eye(16)
  self.assertLess(ntxent(x,x).item(),ntxent(x,x.roll(1,0)).item())
 def test_eq4_includes_consistency_not_only_cross_alignment(self):
  torch.manual_seed(5);model=TFC();x=torch.randn(4,6,178)
  with patch('tfc_model.time_augmentation',side_effect=lambda x:x),patch('tfc_model.frequency_augmentation',side_effect=lambda x:x):
   model.eval();loss,parts=pretrain_objective(model,x)
  torch.testing.assert_close(parts[2],torch.tensor(3.))
  torch.testing.assert_close(loss.detach(),.2*(parts[0]+parts[1])+.8*parts[2])
 def test_neural_artifact_roundtrip_probabilities(self):
  model=GestureTFC().eval();wrapper=TFCInference(model.state_dict());x=np.random.default_rng(5).normal(size=(3,6*178)).astype('float32')
  f=io.BytesIO();joblib.dump(wrapper,f);f.seek(0);restored=joblib.load(f)
  np.testing.assert_allclose(wrapper.predict_proba(x),restored.predict_proba(x),atol=1e-7)
  np.testing.assert_allclose(restored.predict_proba(x).sum(1),1,atol=1e-6)
if __name__=='__main__':unittest.main()
