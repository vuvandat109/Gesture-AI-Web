"""Isolated functional, boundary, persistence and database adapter tests."""
import os, sys, io, json, tempfile, unittest, subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = tempfile.TemporaryDirectory()
os.environ['DATABASE_URL'] = ''
os.environ['DATABASE_PATH'] = str(Path(TMP.name) / 'test.db')
os.environ['SECRET_KEY'] = 'isolated-test-only-not-for-deployment'
import app as web
from account_store import AccountStore, DuplicateUsername
import psycopg
import numpy as np
from werkzeug.security import check_password_hash

def csv_data(n=178):
    t=np.linspace(0, 2*np.pi, n)
    x=np.column_stack([np.sin(t*(i+1)) for i in range(6)])
    s=io.StringIO();s.write('acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n')
    np.savetxt(s,x,delimiter=',',fmt='%.7f');return s.getvalue().encode()

class FunctionalTests(unittest.TestCase):
    def setUp(self):
        web.app.config['TESTING']=True
        self.client=web.app.test_client();self.client.get('/')
        self.user='u'+__import__('uuid').uuid4().hex[:16]
    def post(self,path,**kwargs):
        with self.client.session_transaction() as s: csrf=s['_csrf_token']
        return self.client.post(path,headers={'X-CSRF-Token':csrf},**kwargs)
    def register(self, username=None, password='testpass123'):
        return self.post('/api/register',json={'username':username or self.user,'password':password})
    def login(self):
        self.register()
        return self.post('/api/login',json={'username':self.user,'password':'testpass123'})
    def upload(self,data=None,name='test.csv'):
        return self.post('/api/predict',data={'file':(io.BytesIO(csv_data() if data is None else data),name)},content_type='multipart/form-data')
    def test_F01_register_requires_separate_login(self):
        """Register valid account -> 201, next=login, no authenticated session."""
        r=self.register();self.assertEqual(r.status_code,201);self.assertEqual(r.json['next'],'login')
        with self.client.session_transaction() as s:self.assertNotIn('username',s)
        self.assertEqual(self.upload().status_code,401)
    def test_F02_login_success(self):
        """Register then log in with the same credentials -> authenticated session."""
        self.assertEqual(self.login().status_code,200)
        with self.client.session_transaction() as s:self.assertEqual(s['username'],self.user)
    def test_F03_duplicate_username(self):
        """Register the same username twice -> 409."""
        self.register();self.assertEqual(self.register().status_code,409)
    def test_F04_username_boundaries(self):
        """Username lengths 2/33 rejected; 3/32 accepted."""
        for n,expected in [(2,400),(33,400),(3,201),(32,201)]:
            self.assertEqual(self.register(username='a'*n).status_code,expected)
    def test_F05_password_boundaries(self):
        """Password lengths 7/129 rejected; 8/128 accepted."""
        for n,expected in [(7,400),(129,400),(8,201),(128,201)]:
            self.assertEqual(self.register(username=self.user+str(n),password='x'*n).status_code,expected)
    def test_F06_invalid_username(self):
        """Username containing spaces or SQL syntax is rejected."""
        self.assertEqual(self.register(username="a' OR 1=1--").status_code,400)
    def test_F07_wrong_password(self):
        """Wrong login password -> 401."""
        self.register();r=self.post('/api/login',json={'username':self.user,'password':'wrongpass'})
        self.assertEqual(r.status_code,401)
    def test_F08_csrf_required(self):
        """Missing CSRF header -> 403."""
        self.assertEqual(self.client.post('/api/register',json={}).status_code,403)
    def test_F09_bad_json(self):
        """JSON array/null/non-string credentials -> 400 rather than 500."""
        for body in [[],None,{'username':None,'password':'testpass123'}]:
            self.assertEqual(self.post('/api/register',data=json.dumps(body),content_type='application/json').status_code,400)
    def test_F10_logout(self):
        """Logout clears authentication; prediction returns 401."""
        self.login();self.assertEqual(self.post('/api/logout',json={}).status_code,200)
        self.assertEqual(self.upload().status_code,401)
    def test_F11_password_is_hashed(self):
        """Stored password is a verifiable hash, never plaintext."""
        self.register();h=web.accounts.find(self.user)['password_hash']
        self.assertNotEqual(h,'testpass123');self.assertTrue(check_password_hash(h,'testpass123'))
    def test_F12_persists_across_process_restart(self):
        """A fresh Python process reads the registered hash from the same SQLite file."""
        self.register()
        script='from account_store import AccountStore; from werkzeug.security import check_password_hash; import sys; assert check_password_hash(AccountStore().find(sys.argv[1])["password_hash"],"testpass123")'
        subprocess.run([sys.executable,'-c',script,self.user],cwd=ROOT,check=True,capture_output=True)
    def test_F13_valid_prediction(self):
        """Valid 178x6 CSV -> 200 with five probabilities and finite confidence."""
        self.login();r=self.upload();self.assertEqual(r.status_code,200)
        self.assertEqual(len(r.json['probabilities']),5);self.assertTrue(0<=r.json['confidence']<=100)
    def test_F14_no_file(self):
        """Authenticated upload without file -> 400."""
        self.login();self.assertEqual(self.post('/api/predict',data={}).status_code,400)
    def test_F15_bad_extension(self):
        """Executable extension -> 400 without invoking model."""
        self.login()
        with patch.object(web.MODEL,'predict_proba') as m:
            self.assertEqual(self.upload(name='bad.exe').status_code,400);m.assert_not_called()
    def test_F16_empty_file(self):
        """Empty CSV -> 400."""
        self.login();self.assertEqual(self.upload(b'').status_code,400)
    def test_F17_missing_channels(self):
        """CSV missing gyroscope columns -> 400."""
        self.login();self.assertEqual(self.upload(b'acc_x,acc_y,acc_z\n1,2,3\n').status_code,400)
    def test_F18_rows_boundaries(self):
        """19/10001 rows rejected; 20/10000 rows accepted."""
        self.login()
        for n,status in [(19,400),(20,200),(10000,200),(10001,400)]:self.assertEqual(self.upload(csv_data(n)).status_code,status)
    def test_F19_nan(self):
        """Non-finite sensor value -> 400."""
        self.login();self.assertEqual(self.upload(csv_data().replace(b'0.0000000',b'nan',1)).status_code,400)
    def test_F20_upload_over_limit(self):
        """Multipart request larger than 5 MiB -> 413."""
        self.login();self.assertEqual(self.upload(b'0'*(5*1024*1024+1)).status_code,413)
    def test_F21_invalid_zip(self):
        """Invalid ZIP bytes -> 400."""
        self.login();self.assertEqual(self.upload(b'notzip','bad.zip').status_code,400)
    def test_F22_model_unavailable(self):
        """Absent model -> clear 503 response."""
        self.login()
        with patch.object(web,'MODEL',None):self.assertEqual(self.upload().status_code,503)
    def test_F23_database_outage(self):
        """Database outage returns 503 without exposing credentials."""
        with patch.object(web.accounts,'find',side_effect=psycopg.OperationalError('SECRET_DSN')):
            r=self.post('/api/login',json={'username':self.user,'password':'testpass123'})
        self.assertEqual(r.status_code,503);self.assertNotIn('SECRET_DSN',r.text)
    def test_F24_storage_configuration(self):
        """Health reports the actual account storage backend."""
        self.assertEqual(self.client.get('/api/health').json['account_storage'],'sqlite')

class PostgresAdapterTests(unittest.TestCase):
    def test_P01_parameterized_insert(self):
        """Postgres adapter sends bound parameters and uses connection context transaction."""
        store=AccountStore(url='postgresql://example/test')
        with patch('account_store.psycopg.connect') as connect:
            conn=connect.return_value.__enter__.return_value
            store.create('user','hashed')
            self.assertEqual(conn.execute.call_args.args,('INSERT INTO users (username, password_hash) VALUES (%s, %s)',('user','hashed')))
            connect.return_value.__exit__.assert_called_once()
    def test_P02_no_silent_sqlite_fallback(self):
        """A configured PostgreSQL outage must not create a local SQLite database."""
        p=Path(TMP.name)/'must_not_exist.db';store=AccountStore(url='postgresql://example/test',path=p)
        with patch('account_store.psycopg.connect',side_effect=psycopg.OperationalError()):
            with self.assertRaises(psycopg.OperationalError):store.find('user')
        self.assertFalse(p.exists())
    def test_P03_duplicate(self):
        """Postgres unique-constraint violation maps to DuplicateUsername."""
        store=AccountStore(url='postgresql://example/test')
        with patch('account_store.psycopg.connect') as connect:
            connect.return_value.__enter__.return_value.execute.side_effect=psycopg.errors.UniqueViolation()
            with self.assertRaises(DuplicateUsername):store.create('user','hash')

class RecordedResult(unittest.TextTestResult):
    rows=[]
    def addSuccess(self,test):
        super().addSuccess(test);self.rows.append({'id':test._testMethodName,'description':test.shortDescription(),'status':'PASS'})
    def addFailure(self,test,err):
        super().addFailure(test,err);self.rows.append({'id':test._testMethodName,'description':test.shortDescription(),'status':'FAIL','detail':str(err[1])})
    def addError(self,test,err):
        super().addError(test,err);self.rows.append({'id':test._testMethodName,'description':test.shortDescription(),'status':'ERROR','detail':str(err[1])})
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2,resultclass=RecordedResult).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    (ROOT/'docs/testing/functional_results.json').write_text(json.dumps(RecordedResult.rows,indent=2,ensure_ascii=False))
    sys.exit(not result.wasSuccessful())
