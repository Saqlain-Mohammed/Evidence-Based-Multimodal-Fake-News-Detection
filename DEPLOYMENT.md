# Deploy Veritas on Linux

Use Python 3.12, a writable persistent directory for the SQLite databases, and an
HTTPS reverse proxy on the same host. Install the CPU dependencies using the
commands in [.github/workflows/tests.yml](.github/workflows/tests.yml), inside a
virtual environment created with `python3.12 -m venv .venv`. The shipped models
are loaded from `models/public`. Optional offline transcription requires the
separate setup in [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

Run these commands from the repository root as the service user. Choose a private
persistent directory outside the checkout.

```bash
export VERITAS_REPORTS_FILE=/var/lib/veritas/reports.sqlite3
export VERITAS_ACCOUNTS_FILE=/var/lib/veritas/accounts.sqlite3
.venv/bin/python backend/accounts.py reviewer
.venv/bin/python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
```

Provision an account once for each person and share the generated tokens
privately. Configure both environment variables in the service manager so they
survive restarts. Keep account mode enabled for shared deployments. Route HTTPS
requests to the loopback service, cap request bodies at about 32 MiB for the
30 MiB file limit plus multipart overhead, and allow a response timeout for
video and audio processing. Start with one worker. Each worker loads its own
models and has its own analysis concurrency limit.

Use `GET /api/health` for readiness. In the deployed website, unlock the workspace
with a provisioned token and run the [six sample cases](tests/samples/README.md).
Confirm a fresh session cannot access `/api/v2/history` without a token.

Online evidence is opt-in and requires outbound access to `news.google.com` and
`www.factcheck.org`. URL input also needs outbound access to the article host and
any redirects. Local analysis uses the included checkpoints. If the platform
needs to defer checkpoint loading until the first request, set
`VERITAS_WARM_MODELS=0`.
