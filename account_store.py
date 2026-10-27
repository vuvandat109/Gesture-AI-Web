"""PostgreSQL account storage, with explicit SQLite mode for local development."""
import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager
import psycopg
from psycopg.rows import dict_row

class DuplicateUsername(Exception):
    pass

class AccountStore:
    def __init__(self, url=None, path=None):
        self.url = (url if url is not None else os.environ.get('DATABASE_URL', '')).strip()
        self.path = Path(path or os.environ.get('DATABASE_PATH', Path(__file__).parent / 'instance/users.db'))
        if self.url and not self.url.startswith(('postgres://', 'postgresql://')):
            raise ValueError('DATABASE_URL must use PostgreSQL.')
        self.backend = 'postgresql' if self.url else 'sqlite'

    @contextmanager
    def connection(self):
        if self.url:
            with psycopg.connect(self.url, row_factory=dict_row, connect_timeout=10) as conn:
                yield conn
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=10)
            conn.row_factory = sqlite3.Row
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    def initialize(self):
        with self.connection() as conn:
            if self.url:
                conn.execute('SELECT pg_advisory_xact_lock(478229013)')
            identity = 'BIGSERIAL PRIMARY KEY' if self.url else 'INTEGER PRIMARY KEY'
            conn.execute(f'CREATE TABLE IF NOT EXISTS users (id {identity}, username TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL)')

    def create(self, username, password_hash):
        marker = '%s' if self.url else '?'
        try:
            with self.connection() as conn:
                conn.execute(f'INSERT INTO users (username, password_hash) VALUES ({marker}, {marker})', (username, password_hash))
        except (sqlite3.IntegrityError, psycopg.errors.UniqueViolation) as exc:
            raise DuplicateUsername from exc

    def find(self, username):
        marker = '%s' if self.url else '?'
        with self.connection() as conn:
            row = conn.execute(f'SELECT username, password_hash FROM users WHERE username = {marker}', (username,)).fetchone()
            return dict(row) if row is not None else None
