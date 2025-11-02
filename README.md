# Dash App (Pages + Tabs, port 3000)

- Header: title left, logo right
- Left menu: Page 1–3
- Each page has Excel-like tabs; each tab is its own Python module
- Shared utils in `lib/` (e.g., CSV reads)
- `assets/` for logos/CSS/PDFs, `data/` for CSVs

## Local dev

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py   # http://192.168.1.203:3000 (binds 0.0.0.0:3000)
```

## Deploy on self-hosted Coolify (no Docker)
- Create **Application** → connect GitHub repo.
- Buildpack will detect Python from `requirements.txt`.
- Start command (from Procfile): `gunicorn app:server -b 0.0.0.0:${PORT:-3000}`
- In Coolify, set the app's **HTTP Port** to match `${PORT}` it injects (or leave default).
