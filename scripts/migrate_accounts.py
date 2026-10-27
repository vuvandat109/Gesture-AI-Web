"""Copy existing SQLite password hashes to PostgreSQL without overwriting accounts.
Usage: DATABASE_URL=... python scripts/migrate_accounts.py /path/to/users.db
The old SQLite file must still exist. Keep it private; it contains password hashes.
"""
import os,sys,sqlite3
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from account_store import AccountStore

def migrate(source, target):
    source=Path(source).resolve()
    if not source.is_file():raise ValueError('Source SQLite database does not exist.')
    if target.backend!='postgresql':raise ValueError('Set DATABASE_URL to the destination PostgreSQL database.')
    conn=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)
    try:rows=conn.execute('SELECT username,password_hash FROM users').fetchall()
    finally:conn.close()
    target.initialize();inserted=0
    with target.connection() as db:
        for username,password_hash in rows:
            cur=db.execute('INSERT INTO users(username,password_hash) VALUES (%s,%s) ON CONFLICT(username) DO NOTHING',(username,password_hash))
            inserted+=cur.rowcount
            record=db.execute('SELECT password_hash FROM users WHERE username=%s',(username,)).fetchone()
            if record['password_hash']!=password_hash:
                raise ValueError('Username conflict with a different password hash. Entire import rolled back; no accounts overwritten.')
    return {'source_accounts':len(rows),'inserted':inserted,'already_identical':len(rows)-inserted}

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: python scripts/migrate_accounts.py /path/to/users.db')
    try:print(migrate(sys.argv[1],AccountStore()))
    except Exception as exc:
        # Database exception details may contain connection information.
        print(str(exc) if isinstance(exc,ValueError) else 'Migration failed; no account rows committed. Check database configuration.',file=sys.stderr)
        sys.exit(1)
