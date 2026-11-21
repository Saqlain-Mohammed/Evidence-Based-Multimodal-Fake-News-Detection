"""Provision named local API accounts and validate bearer credentials."""
from contextvars import ContextVar
import hashlib
import os
from pathlib import Path
import secrets
import sqlite3

OWNER = ContextVar("report_owner", default="")


def account_db():
    setting = os.environ.get("VERITAS_ACCOUNTS_FILE")
    return Path(setting).resolve() if setting else None


def authenticate(path, authorization):
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token or not path.is_file():
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as db:
        row = db.execute("SELECT name FROM accounts WHERE digest=?", (digest,)).fetchone()
    return row[0] if row else None


def provision(path, name):
    if not name.strip():raise ValueError("An account name is required.")
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS accounts(name TEXT PRIMARY KEY, digest TEXT UNIQUE NOT NULL)")
        db.execute("INSERT INTO accounts VALUES (?,?)",(name,hashlib.sha256(token.encode()).hexdigest()))
    return token


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("name")
    args=parser.parse_args()
    path=account_db()
    if not path:parser.error("Set VERITAS_ACCOUNTS_FILE to a private SQLite file path first.")
    print(provision(path,args.name))
