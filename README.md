# Burla Agent Starter Kit

> **Minimum-intervention onboarding for [Burla](https://burla.dev).**  
> One command, one password, fully automated cluster provisioning and job execution.

This kit lets an AI coding agent — or a human — take a brand-new Burla
account from zero to **100,000 parallel remote function calls** with a
single manual step (entering your Google password, once).

Everything else — clicking "Continue with Google", clicking "Authorize"
for the CLI, discovering the cluster URL, matching the Python version
and burla client version that the cluster requires, turning the cluster
on, running test jobs — happens without any further human input.

---

## Why this exists

Every new Burla account involves five papercuts:

1. **Google OAuth** in a browser. Password can't be scripted securely.
2. **Clicking "Authorize" once for the CLI.** Trivial but manual.
3. **Matching your local Python to the cluster's containers.** A mismatch fails with `NodeConflict`.
4. **Matching your local `burla` pip version to the cluster's supported range.** A mismatch fails with `VersionMismatch`.
5. **Starting the cluster.** New clusters start in the `Off` state; `grow=True` only works on some versions, and the dashboard UI needs a click.

This kit collapses all five into one idempotent Python script.

---

## What's in the box

```
burla-agent-starter-kit/
├── onboard.py              # the single command: python onboard.py --email you@example.com
├── run_job.py              # run any script in the provisioned venv
├── scripts/bootstrap.sh    # install Playwright + run onboard + run demo
├── demos/
│   ├── square.py           # 100k squares (sanity check)
│   ├── primes.py           # primes under 1M across 500 buckets
│   └── wordlen.py          # 50k synthetic sentences → avg word length
└── burla_kit/
    ├── auth.py             # Playwright: click Continue-with-Google & Authorize
    ├── cluster.py          # grow=True → UI Start escalation ladder
    ├── probe.py            # detect Python/burla version, install correct ones
    ├── venv.py             # per-account ~/.burla/<slug>/.venv manager
    ├── config.py           # per-account ~/.burla/<slug>/{.env,user_config.json}
    ├── onboard.py          # orchestrates steps 1–4
    └── logging.py          # coloured, flush-on-write logging
```

---

## Quick start

```bash
# 1) Clone and cd into the repo
git clone https://github.com/<you>/burla-agent-starter-kit.git
cd burla-agent-starter-kit

# 2) Install deps (system Python, just the onboarder needs these)
pip install -r requirements.txt
python -m playwright install chromium

# 3) Onboard and run the 'hello world' of Burla
python onboard.py --email you@example.com --demo demos/square.py
```

**First run:** a Chromium window opens. Click in the Google sign-in and
type your password (you'll see a log line telling you exactly when).
That's the only moment of human intervention.

**Every subsequent run:** zero intervention — the saved Playwright
profile keeps you signed in, cluster state is detected, and demos run.

---

## The escalation ladder

```
┌─────────────────────────────────────────────────────┐
│  python onboard.py --email you@example.com          │
└────────────────────────────┬────────────────────────┘
                             │
           ┌─────────────────┴─────────────────┐
           │ 1. Playwright login (auth.py)     │
           │    • auto-click Continue w/ Google│
           │    • wait for Google sign-in      │  ← only manual step (once)
           │    • auto-click Authorize for CLI │
           │    • save session to chrome-profile/
           └─────────────────┬─────────────────┘
                             │
           ┌─────────────────┴─────────────────┐
           │ 2. Version probe (probe.py)       │
           │    • run trivial remote_parallel_map
           │    • on NodeConflict → rebuild    │
           │      venv with matching Python     │
           │    • on VersionMismatch → pip     │
           │      install burla==<required>    │
           └─────────────────┬─────────────────┘
                             │
           ┌─────────────────┴─────────────────┐
           │ 3. Cluster on (cluster.py)        │
           │    a) remote_parallel_map(grow=True) ── ok ──┐
           │    b) UI Start via Playwright     │          │
           │       (profile already signed in) │          │
           │    c) poll until READY            │          │
           └─────────────────┬─────────────────┘          │
                             │                             │
           ┌─────────────────┴─────────────────┐          │
           │ 4. Run demo (venv.py)             │◄─────────┘
           │    .venv/bin/python demos/square.py
           └───────────────────────────────────┘
```

---

## Per-account state

Nothing user-specific lives in this repo. Everything lands under
`~/.burla/<slug>/`:

| Path | Contents |
|------|----------|
| `~/.burla/<slug>/user_config.json` | Non-secret settings (cluster URL, project id, client versions). |
| `~/.burla/<slug>/.env` | Email + cluster URL. `chmod 600`. |
| `~/.burla/<slug>/chrome-profile/` | Playwright session (Google cookie). |
| `~/.burla/<slug>/.venv/` | Auto-provisioned venv matching cluster versions. |
| `~/Library/Application Support/burla/burla_credentials.json` | Written by `burla login` — auth token + cluster URL. |

All of these are `.gitignore`d. The `<slug>` is the email local-part
lowercased with non-alphanumerics removed (e.g. `joeyper23@gmail.com` →
`joeyper23`).

---

## Running your own jobs

```bash
# Either go through the wrapper (recommended):
python run_job.py --email you@example.com path/to/your_script.py

# Or invoke the venv directly:
~/.burla/joeyper23/.venv/bin/python path/to/your_script.py
```

Any script that `import burla; from burla import remote_parallel_map`
will just work — the venv already has the right client version.

---

## Environment variables

| Var | Default | Meaning |
|-----|---------|---------|
| `BURLA_EMAIL` | — (required) | Email to onboard |
| `BURLA_AUTH_PROVIDER` | `google` | Future: `microsoft` |
| `BURLA_URL` | auto-detected | Override the cluster URL |
| `PYTHON` | `python3` | Interpreter bootstrap uses |

---

## Re-running / repairing

Every step is **idempotent**. Safe to re-run:

```bash
python onboard.py --email you@example.com
```

- If already authed → skip auth.
- If venv already matches → skip pip install.
- If cluster already running → skip Start click.

To wipe an account's local state and start over:

```bash
rm -rf ~/.burla/<slug>
rm -f ~/Library/Application\ Support/burla/burla_credentials.json
```

---

## For agents (machine readers)

This repo is designed to be operated by an AI coding agent. Useful
contract details:

- The onboarder **never stores a Google password anywhere** (it can't —
  Google OAuth intentionally blocks that). Any agent asked to
  automate this end-to-end must accept that one password entry is
  required per saved Playwright profile lifecycle.
- The probe stage **never hard-codes** Python or burla versions — it
  parses the cluster's own error messages and installs whatever the
  cluster asks for. Safe against upstream version bumps.
- Cluster start uses `grow=True` first, the UI Start click as a
  fallback. `ensure_cluster_on(email, url, venv_python)` is the public
  entry point; agents should prefer calling it over driving the
  dashboard directly.
- Nothing in this repo should be modified per-tenant. Tenant-specific
  values go in `~/.burla/<slug>/user_config.json`.

---

## Known limitations / TODO

- Microsoft OAuth is stubbed but not implemented in `auth.py`.
- "Start" button selectors are best-effort; if the dashboard UI
  relabels its button, add the new label to
  `burla_kit/cluster.py:START_BUTTON_LABELS`.
- A pure-API cluster-start path (to skip Playwright entirely for step 3)
  is a planned optimization.

---

## License

MIT. See `LICENSE`.
