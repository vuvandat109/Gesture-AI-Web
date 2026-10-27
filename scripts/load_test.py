"""5-user/120-second HTTP load test. Run only against an authorized test deployment.
Generated fixtures test throughput/validation, NOT gesture classification accuracy.
"""
import argparse,io,json,time,threading,urllib.request,urllib.error,http.cookiejar,uuid,sys,platform,os,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:5055');p.add_argument('--users',type=int,default=5);p.add_argument('--duration',type=float,default=120);p.add_argument('--think-time',type=float,default=1);p.add_argument('--output',default=str(ROOT/'docs/testing/load_results.json'));args=p.parse_args()
rng=np.random.default_rng(42);fixtures=[]
for i in range(50):
    t=np.linspace(0,2*np.pi,178);x=np.column_stack([np.sin(t*(k+1)+i/10) for k in range(6)])+rng.normal(0,.3,(178,6))
    b=io.StringIO();b.write('acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n');np.savetxt(b,x,delimiter=',',fmt='%.7f');fixtures.append(b.getvalue().encode())
clients=[];run_id=uuid.uuid4().hex[:10]
for u in range(args.users):
    opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    page=opener.open(args.url+'/',timeout=30).read().decode()
    import re
    csrf=re.search(r'name="csrf-token" content="([^"]+)"',page).group(1)
    credentials={'username':f'load_{run_id}_{u}','password':uuid.uuid4().hex}
    for route in ['register','login']:
        req=urllib.request.Request(args.url+'/api/'+route,data=json.dumps(credentials).encode(),headers={'Content-Type':'application/json','X-CSRF-Token':csrf})
        with opener.open(req,timeout=30) as r:reply=json.load(r)
        csrf=reply['csrf']
    clients.append((opener,csrf))
barrier=threading.Barrier(args.users);rows=[];lock=threading.Lock()
def worker(u):
    opener,csrf=clients[u];barrier.wait();deadline=time.perf_counter()+args.duration;i=0;local=[]
    while time.perf_counter()<deadline:
        raw=fixtures[(u+i*args.users)%len(fixtures)];boundary='gestureloadboundary'
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="sample.csv"\r\nContent-Type: text/csv\r\n\r\n'.encode()+raw+f'\r\n--{boundary}--\r\n'.encode())
        req=urllib.request.Request(args.url+'/api/predict',data=body,headers={'Content-Type':f'multipart/form-data; boundary={boundary}','X-CSRF-Token':csrf})
        start=time.perf_counter();status=0;confidence=None;valid=False;fallback=False;err=None
        try:
            with opener.open(req,timeout=15) as r:status=r.status;reply=json.load(r)
            confidence=reply.get('confidence');valid=isinstance(confidence,(int,float)) and 0<=confidence<=100 and len(reply.get('probabilities',[]))==5
            fallback=reply.get('status')=='FALLBACK'
        except urllib.error.HTTPError as e:status=e.code;err='HTTP error'
        except Exception as e:err=type(e).__name__
        local.append({'user':u,'iteration':i,'latency_ms':(time.perf_counter()-start)*1000,'http_status':status,'schema_valid':valid,'confidence':confidence,'fallback':fallback,'error':err})
        i+=1
        remaining=deadline-time.perf_counter()
        if remaining>0:time.sleep(min(args.think_time,remaining))
    with lock:rows.extend(local)
start=time.perf_counter()
with ThreadPoolExecutor(max_workers=args.users) as ex:list(ex.map(worker,range(args.users)))
elapsed=time.perf_counter()-start
lat=[r['latency_ms'] for r in rows];errors=sum(r['http_status']!=200 for r in rows);success=[r for r in rows if r['schema_valid']]
result={'environment':'local HTTP, Gunicorn 2 workers x 2 threads; SQLite account storage; not Render or PostgreSQL measurements','python':platform.python_version(),'machine':platform.machine(),'logical_cpus':os.cpu_count(),'model_sha256':hashlib.sha256((ROOT/'models/gesture_real_v1.joblib').read_bytes()).hexdigest(),'workload':{'users':args.users,'duration_seconds':args.duration,'think_time_seconds':args.think_time,'fixture_count':50,'fixture_type':'deterministic synthetic 178x6 CSV; no correctness labels','fixture_size_bytes_min':min(map(len,fixtures)),'fixture_size_bytes_max':max(map(len,fixtures))},'requests':len(rows),'elapsed_seconds':elapsed,'throughput_requests_per_second':len(rows)/elapsed,'p50_ms':float(np.percentile(lat,50)),'p95_ms':float(np.percentile(lat,95)),'p99_ms':float(np.percentile(lat,99)),'error_rate':errors/len(rows),'schema_success_rate':len(success)/len(rows),'low_confidence_proxy_rate':sum(r['confidence']<80 for r in success)/len(success) if success else None,'unknown_rate':None,'fallback_rate':None,'ai_metric_note':'App has no explicit UNCERTAIN/FALLBACK states. Confidence <80% is an analysis proxy only. Not a measured classification error rate.','criteria':{'p95_ms_less_than':3000,'error_rate_less_than':.01,'schema_success_rate':1.0},'performance_pass':bool(np.percentile(lat,95)<3000 and errors/len(rows)<.01 and len(success)==len(rows))}
Path(args.output).write_text(json.dumps(result,indent=2));Path(args.output).with_name('load_requests.json').write_text(json.dumps(rows,indent=2));print(json.dumps(result,indent=2))
