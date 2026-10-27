"""Live PostgreSQL integration. Use TEST_DATABASE_URL pointing to a disposable test DB."""
import os,sys,subprocess,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from account_store import AccountStore,DuplicateUsername
from werkzeug.security import generate_password_hash,check_password_hash
url=os.environ.get('TEST_DATABASE_URL')
if not url:
    print('SKIP: no TEST_DATABASE_URL; live PostgreSQL and cloud persistence unverified.')
    sys.exit(0)
store=AccountStore(url=url);store.initialize();username='ci_'+uuid.uuid4().hex[:20];password=uuid.uuid4().hex
try:
    store.create(username,generate_password_hash(password))
    assert check_password_hash(store.find(username)['password_hash'],password)
    try:store.create(username,generate_password_hash(password));raise AssertionError('Duplicate accepted')
    except DuplicateUsername:pass
    env=dict(os.environ,TEST_USER=username,TEST_PASSWORD=password)
    subprocess.run([sys.executable,'-c',"import os; from account_store import AccountStore; from werkzeug.security import check_password_hash; s=AccountStore(url=os.environ['TEST_DATABASE_URL']); assert check_password_hash(s.find(os.environ['TEST_USER'])['password_hash'],os.environ['TEST_PASSWORD'])"],cwd=Path(__file__).resolve().parents[1],env=env,check=True)
    print('PASS: PostgreSQL create, uniqueness, hash verification and fresh-process persistence')
finally:
    with store.connection() as db:db.execute('DELETE FROM users WHERE username=%s',(username,))
