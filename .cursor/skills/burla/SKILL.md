---
name: burla
description: Onboard a Burla account end-to-end, run parallel Python jobs via remote_parallel_map, and build Burla demos using the Burla Agent Starter Kit. Invoked as /burla. Use when the user asks to set up Burla, log in, create a Burla account, run jobs on a Burla cluster, scale Python work across remote CPUs, or build/push Burla demos via the starter kit.
---

# Burla Agent Starter Kit Skill (/burla)

You are helping a user use **[Burla](https://burla.dev)** — a service that runs any Python function across many remote CPUs in parallel via `burla.remote_parallel_map` — through the tooling in **this repository**.

> Before doing anything, read [Per-account state](#per-account-state) so you know which files you may edit and which you must never touch.

---

## When to use this skill

Trigger this skill whenever the user:

- Wants to onboard a brand-new Burla account (including Google OAuth + CLI auth + cluster start)
- Wants to re-authenticate an existing account or switch between accounts
- Wants to run a `remote_parallel_map` job at scale (thousands to millions of inputs)
- Wants to build and push a Burla demo to GitHub
- Hits a `VersionMismatch`, `NodeConflict`, `NoNodes`, or `ClientConnectorError` and needs auto-remediation

---

## What the kit does (one-liner)

```bash
python onboard.py --email <user-email> --demo demos/square.py
```

First run on a new machine: one Google password entry; everything else — OAuth consent, CLI authorize, venv creation, client-version matching, cluster power-on, and a 100k-input demo — runs automatically. Every subsequent run on that machine is fully zero-touch.

---

## Architecture: the self-healing ladder

`burla_kit/cluster.py:ensure_cluster_ready` is the core loop. Failures are classified by `burla_kit/probe.py:VersionProbe` and routed to the cheapest fix:

| Detected state | Action |
|---|---|
| `OK` | Cluster reachable, versions compatible — done. |
| `NO_GROW_KWARG` | Installed client predates `grow=True` (e.g. `burla==1.4.5`). Fall through to a plain probe. |
| `VERSION_MISMATCH` | Parse the cluster's error, `pip install burla==<required>` and/or rebuild the venv with the required Python. |
| `CLUSTER_OFF` | Drive the dashboard UI via the saved Playwright profile, click *Start*, poll until a node is `READY`. |
| `UNKNOWN` | Dump last 800 chars and abort with a hypothesis. |

Never hard-code Python or `burla` versions — always read them from the cluster's own error messages.

---

## Per-account state

Everything user-specific lives under `~/.burla/<slug>/`, where `<slug>` is the email local-part lowercased with non-alphanumerics stripped (`jperry@bamboorose.com` → `jperry`). Multiple accounts coexist.

| Path | Purpose |
|---|---|
| `~/.burla/<slug>/user_config.json` | Non-secret: cluster URL, project id, pinned client versions. |
| `~/.burla/<slug>/.env` | Secrets (email, cluster URL). `chmod 600`. |
| `~/.burla/<slug>/chrome-profile/` | Playwright persistent profile (saved Google session). |
| `~/.burla/<slug>/.venv/` | Auto-provisioned Python venv matching the cluster's required versions. |
| `~/Library/Application Support/burla/burla_credentials.json` | Written by `burla login`. **Global**: only the most recently logged-in account is authoritative. |

**Rules for the agent:**

- Edit `~/.burla/<slug>/user_config.json` and `.env` freely when provisioning that account.
- Modify cluster settings via the Burla UI / API on the **user's** cluster (image, VM, authorized users) — the user has granted full admin actions on their own cluster.
- Never commit anything under `~/.burla/` to any git repo.
- Never place user-specific values (emails, cluster URLs, project ids) into this skill file or any committed file in the kit. Those values live under `~/.burla/<slug>/` only.
- Never edit the canonical sections of this skill file with a specific user's URL, email, paths, or VM specs.
- If `~/.burla/<slug>/.env` does not exist, create it and ensure `.env` is in `.gitignore`.

---

## Canonical workflows

### A. Onboarding a new user

1. Ask for the email and auth provider (Google by default). Never ask for a password.
2. Ensure deps are installed:
   ```bash
   pip install -r requirements.txt
   python -m playwright install chromium
   ```
3. Run the onboarder:
   ```bash
   python onboard.py --email <them> --demo demos/square.py
   ```
4. Tell the user exactly when to enter their Google password (the log prints a `Waiting for Google sign-in` prompt — relay that to them).
5. On success, the log ends with `demo completed` and prints 100,000 results. Save the cluster URL from `~/.burla/<slug>/user_config.json` for future reference.

### B. Running a job on an already-onboarded account

```bash
python run_job.py --email <them> path/to/script.py
```

This shells out to `~/.burla/<slug>/.venv/bin/python` so the correct `burla` client version is always used. `onboard.py --email <them>` (no `--demo`) is also safe — it's idempotent and short-circuits when state is already valid.

### C. Building a demo the user may want to push to GitHub

1. **Validate locally first.** Cold-starting a cluster takes ~90s and Burla errors are slow — don't use remote execution as your first sanity check.
2. Place the demo in `demos/` for the kit's own demos, or create a separate repo under `agents/<repo>/` if it will be pushed.
3. Define worker functions at **module top level** (must be picklable — no lambdas, no closures over outer state).
4. Run it via `run_job.py` and confirm `REMOTE_OK` appears.
5. Only **after** a successful remote run, ask the user whether to push to GitHub with a polished README.

### D. Map-Reduce via shared filesystem

`/workspace/shared` is a GCS-backed folder available on all worker nodes:

- Write to `/workspace/shared/...` → file appears in GCS bucket
- Read from `/workspace/shared/...` → reads from GCS bucket
- Visible in the Burla dashboard under **Filesystem**
- Use `Path(...).parent.mkdir(parents=True, exist_ok=True)` before writing nested paths

```python
from pathlib import Path
from burla import remote_parallel_map

def write_part(number):
    path = f"/workspace/shared/demo/parts/{number}.txt"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(f"{number}\n")
    return path

part_paths = remote_parallel_map(write_part, list(range(100)), grow=True)

def combine(paths):
    total = sum(int(Path(p).read_text().strip()) for p in paths)
    out = "/workspace/shared/demo/total.txt"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(str(total))
    return out

remote_parallel_map(combine, [part_paths], func_cpu=8, func_ram=32, grow=True)
```

Wrap the reduce input in a list-of-one (`[part_paths]`) so it runs as a single call.

---

## remote_parallel_map reference

```python
from burla import remote_parallel_map

def process(item):
    return result  # any picklable return

results = remote_parallel_map(process, items)  # returns list (unordered!)
```

Full current signature (keep this up to date with https://docs.burla.dev):

```python
remote_parallel_map(
    function_,            # picklable, <100MB pickled, module-top-level
    inputs,               # list; tuples are unpacked into *args
    func_cpu=1,
    func_ram=4,
    func_gpu=None,        # "A100"/"A100_40G", "A100_80G", "H100"/"H100_80G"
    image=None,           # container image filter (+ new-node image when grow=True)
    grow=False,           # auto-boot nodes; missing on burla<=1.4.5
    max_parallelism=None,
    detach=False,         # job survives local process exit
    generator=False,      # yield as results complete (still unordered)
    spinner=True,
)
```

Kwarg cheat-sheet:

| Kwarg | Default | Purpose |
|---|---|---|
| `func_cpu` | `1` | CPUs per worker instance. |
| `func_ram` | `4` | GB RAM per worker instance. |
| `func_gpu` | `None` | `"A100"` / `"A100_40G"`, `"A100_80G"`, `"H100"` / `"H100_80G"`. One GPU per call. |
| `image` | `None` | Restrict to nodes running this container image; with `grow=True`, new nodes boot with it. |
| `grow` | `False` | Auto-provision additional nodes. Not present on `burla<=1.4.5` (kit handles this). |
| `max_parallelism` | `None` | Cap concurrent running instances. |
| `detach` | `False` | Keep running past the local process; detached jobs run indefinitely. |
| `generator` | `False` | Stream results as they arrive (`for r in …`). |
| `spinner` | `True` | Progress spinner in terminal. |

Key runtime behaviors:

- **Results are returned in no particular order** (both `list` mode and `generator=True`). Never assume `results[i]` corresponds to `inputs[i]` — include a correlator in the return value if you need to pair them.
- Tuple items in `inputs` are unpacked into `*args` (`function_(*item)`); all other types are passed as a single argument.
- Exceptions on workers re-raise on the client with full tracebacks.
- `print()` (and any `stdout`/`stderr`) streams back in real time.
- Worker function imports are auto-detected and installed on workers.
- Non-Python system deps (ffmpeg, libGL, etc.) require either a pre-built `image=...` or updating the cluster's container image.

---

## Burla CLI reference (the kit wraps this — don't call it directly)

| Command | Purpose | Prerequisites | Kit interaction |
|---|---|---|---|
| `burla install` | Deploy a self-hosted Burla instance into the current `gcloud` project. Re-running updates the install. | `gcloud` CLI installed; `gcloud auth login` + `gcloud auth application-default login`; sufficient IAM perms (the command tells you which ones are missing). | **Never called by this kit.** Only used by a human operator standing up a self-hosted cluster. Do not invoke from agent onboarding flows. |
| `burla login` | Opens the "Authorize this Machine" page in the default browser for the cluster the user most recently logged into via the dashboard. On Authorize, writes a refreshed auth token to `burla_credentials.json` in the OS user-data dir. | The user must have logged into the cluster's dashboard at least once in a browser. | Invoked automatically by `burla_kit/auth.py` as `burla login --no_browser` so the kit can scrape the auth URL and Playwright-click Authorize without spawning another browser. |
| `--help` | CLI docs; works after any command or group. | — | Debugging only. |

`burla_credentials.json` location (macOS): `~/Library/Application Support/burla/burla_credentials.json`. It is **global** — only the most recently authorized account is active.

---

## Troubleshooting

Always run in this order before modifying the user's code:

1. **Re-run `onboard.py --email <them>`** (no `--demo`). It's idempotent and fixes most transient issues.
2. For `NoNodes` / "cluster off" / `ClientConnectorError` → the kit's UI Start fallback should catch it. If it doesn't, the user can click *Start* on the dashboard URL from `user_config.json`.
3. For `VersionMismatch` / `NodeConflict` → the probe rebuilds the venv automatically. If it keeps failing, delete `~/.burla/<slug>/.venv` and re-run onboard.
4. For auth / token errors → delete `~/Library/Application Support/burla/burla_credentials.json` and re-run onboard.
5. For worker-side errors (missing system deps) → the cluster's container image needs updating; that's a cluster-admin action via the dashboard Settings.

If the same error happens twice in a row after these steps, stop and report the exact error with a one-line hypothesis.

---

## Hard rules (do not violate)

- **Never** push to `https://github.com/Burla-Cloud/burla` (that is the upstream Burla platform repo, not this kit).
- **Never** commit `.env`, `chrome-profile/`, `.venv/`, or anything under `~/.burla/` to any git repo.
- **Never** type or store the user's Google/OAuth password. Password entry must remain with the user.
- **Never** bake user-specific emails, cluster URLs, or project ids into this skill file, the kit's source, or any README — they live under `~/.burla/<slug>/` only.
- **Never** modify another tenant's cluster. Cluster changes are allowed only on the authenticated user's own cluster via their dashboard.
- **Never silently pivot.** If the chosen data source or approach is infeasible after multiple attempts, stop and tell the user — do not substitute a different data source or approach. A demo built on the wrong data (e.g. research papers when the pitch was patents) is worse than no demo. Ask before changing anything fundamental about the idea.
- **Always** validate a demo locally before pushing it to GitHub. Burla cold starts are 3–5 min and errors are slow — don't use Burla as the first test of whether code works.
- **Always** wait for `REMOTE_OK` before creating a GitHub repo for a demo. No repo, no remote, no push until the pipeline has run end-to-end on Burla.

---

## Kit layout (for quick navigation)

```
burla-agent-starter-kit/
├── onboard.py              # entry: single-command onboarding
├── run_job.py              # entry: run any script in the provisioned venv
├── requirements.txt        # playwright + python-dotenv (onboarder deps only)
├── burla_kit/
│   ├── onboard.py          # orchestrator (auth → cluster → venv → demo)
│   ├── auth.py             # Playwright: Continue-with-Google + Authorize
│   ├── cluster.py          # grow → UI Start escalation ladder
│   ├── probe.py            # version detection, error classification
│   ├── venv.py             # per-account ~/.burla/<slug>/.venv manager
│   ├── config.py           # per-account user_config.json / .env I/O
│   └── logging.py          # coloured, flush-on-write logger
└── demos/
    ├── square.py           # 100k squares — sanity check
    ├── primes.py           # primes under 1M across 500 buckets
    └── wordlen.py          # 50k synthetic sentences → avg word length
```
