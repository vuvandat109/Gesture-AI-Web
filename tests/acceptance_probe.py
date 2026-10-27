"""Report product acceptance gaps; controlled AI outputs are explicitly mocked."""
import sys,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
from test_application import FunctionalTests, web
import numpy as np
case=FunctionalTests();case.setUp();case.login();rows=[]
def record(id,condition,expected,actual,passed):rows.append(dict(id=id,condition=condition,expected=expected,actual=actual,status='PASS' if passed else 'FAIL'))
with patch.object(web.MODEL,'predict_proba',return_value=np.array([[.95,.0125,.0125,.0125,.0125]])):
 r=case.upload();record('AC01','Controlled high-confidence output (mock)','HTTP 200; prediction shown with confidence 95%',f'HTTP {r.status_code}, confidence={r.json.get("confidence")}',r.status_code==200 and r.json.get('confidence')==95)
with patch.object(web.MODEL,'predict_proba',return_value=np.array([[.4,.2,.15,.15,.1]])):
 r=case.upload();record('AC02','Controlled confidence 40% (mock); proposed threshold 80%','UNCERTAIN state and human-review warning',f'HTTP {r.status_code}; status={r.json.get("status")}; confidence={r.json.get("confidence")}',r.json.get('status')=='UNCERTAIN')
with patch.object(web.MODEL,'predict_proba',side_effect=TimeoutError('Injected timeout')):
 r=case.upload();record('AC03','Injected TimeoutError; this does not test an actual time budget','FALLBACK state and usable alternative/retry guidance',f'HTTP {r.status_code}; no FALLBACK state',r.json.get('status')=='FALLBACK')
with patch.object(web.MODEL,'predict_proba') as m:
 r=case.upload(b'not CSV','invalid.exe');record('AC04','Unsupported input file','HTTP 400 and model not called',f'HTTP {r.status_code}; model_called={m.called}',r.status_code==400 and not m.called)
routes={str(r) for r in web.app.url_map.iter_rules()}
record('AC05','Review and save prediction','User confirms/corrects prediction before persistent save','No prediction review/save feature in current UI/routes (out of original gesture-demo scope)',False)
r=case.upload();record('AC06','Traceability','Version identifier in inference response','model_name exists; immutable model_version absent','model_version' in r.json)
path=Path(__file__).resolve().parents[1]/'docs/testing/acceptance_results.json';path.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
