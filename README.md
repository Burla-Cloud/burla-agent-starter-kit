# Burla Agent Starter Kit

> **One command, one password, fully automated Burla onboarding + parallel job runner.**
> This README is written for an AI agent. An agent should be able to use Burla end-to-end by reading only this file.

---

## 0. Agent TL;DR (read this first)

**What Burla is.** [Burla](https://burla.dev) is a managed service that runs any Python function across many remote CPUs in parallel. The entire user-facing API is one function:

```python
from burla import remote_parallel_map

def work(item):
    return item * 2

results = remote_parallel_map(work, [1, 2, 3, 4, 5])  # -> [2, 4, 6, 8, 10]
```

That's it. You pass a function and an iterable of inputs; Burla runs the function once per input on a remote cluster and returns the list of results.

**What this kit is.** A zero-config onboarder + job runner that turns a brand-new Burla account into a working `remote_parallel_map` environment with one command and one human action (typing a Google password, once).

**The only two commands you need:**


| Goal                                            | Command                                                            |
| ----------------------------------------------- | ------------------------------------------------------------------ |
| Provision an account and run a 100k-input demo  | `python onboard.py --email you@example.com --demo demos/square.py` |
| Run any user script on Burla (after onboarding) | `python run_job.py --email you@example.com path/to/script.py`      |


**The only manual step, ever:** typing a Google password into a Chromium window on the very first run. The kit saves the session and never asks again.

**If anything goes wrong:** re-run `python onboard.py --email <them>` (no `--demo`). It is idempotent. It auto-detects and repairs version mismatches, dead clusters, stale auth, and missing venvs.

---

## 1. What the kit does, in order

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
           │      venv with matching Python    │
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

Every step is **idempotent**. Safe to re-run: already-authed → skip auth; venv already matches → skip pip; cluster already running → skip Start.

---

## 2. `remote_parallel_map` — full API reference

This is the **only** Burla function you'll ever call. If you're an agent writing Burla code, this section is your contract.

### Signature

```python
from burla import remote_parallel_map

remote_parallel_map(
    function_,
    inputs,
    func_cpu=1,
    func_ram=4,
    func_gpu=None,
    image=None,
    grow=False,
    max_parallelism=None,
    detach=False,
    generator=False,
    spinner=True,
)
```

### Parameters


| Parameter         | Type         | Default  | Meaning                                                                                                                                                                               |
| ----------------- | ------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `function_`       | `Callable`   | required | Function to run on each input. Must be picklable: **define at module top level**, no lambdas, no closures over mutable outer state. Pickled function must be `< 100 MB` (≈ 1M lines). |
| `inputs`          | `List[Any]`  | required | List of items. Each item becomes one remote call to `function_`. **Tuples are unpacked into `*args`** (`function_(*item)`); anything else is passed as the single argument.           |
| `func_cpu`        | `int`        | `1`      | CPUs per running instance of `function_`. Max depends on cluster machine type.                                                                                                        |
| `func_ram`        | `int`        | `4`      | GB RAM per running instance of `function_`. Max depends on cluster machine type.                                                                                                      |
| `func_gpu`        | `str | None` | `None`   | Allocate one GPU per call. One of `"A100"` / `"A100_40G"`, `"A100_80G"`, `"H100"` / `"H100_80G"`.                                                                                     |
| `image`           | `str | None` | `None`   | Restrict the job to nodes running this container image. With `grow=True` and no matching nodes, newly booted nodes will use this image.                                               |
| `grow`            | `bool`       | `False`  | Auto-boot additional nodes to finish the work faster. New nodes inherit existing cluster settings.                                                                                    |
| `max_parallelism` | `int | None` | `None`   | Cap on concurrent running instances of `function_`.                                                                                                                                   |
| `detach`          | `bool`       | `False`  | If `True`, the job keeps running on the cluster even if the local process stops. Detached jobs can run indefinitely in the background.                                                |
| `generator`       | `bool`       | `False`  | If `True`, return a generator yielding outputs as they are produced (instead of a list at the end).                                                                                   |
| `spinner`         | `bool`       | `True`   | Show the status indicator / progress spinner in the terminal.                                                                                                                         |


### Returns

- `List[Any]` of whatever `function_` returned, **in no particular order**.
- If `generator=True`, returns a generator yielding outputs as they are produced (also **unordered** — the order is "as they finish", not input order).

> **Important for agents:** Do **not** assume `results[i]` corresponds to `inputs[i]`. If you need to correlate an output with its input, include the identifying info in the function's return value (e.g. `return {"input": x, "result": ...}`).

### Runtime behavior you can rely on

- `print()` (and anything written to `stdout` / `stderr`) inside `function_` streams back to the local terminal in real time, as if the code were running locally.
- Exceptions inside `function_` re-raise **on the client machine** with full remote tracebacks.
- Python imports inside `function_` are auto-detected and installed on workers.
- **System dependencies** (ffmpeg, libGL, apt packages) are **not** auto-installed — they require updating the cluster's container image (a cluster-admin action, not an agent action). Use the `image` kwarg to target a pre-built image that already has what you need.
- Return values must be picklable. Prefer plain `dict` / `list` / `int` / `str` / `float`.
- If the cluster is off, pass `grow=True` to auto-boot nodes (or rely on the kit's escalation ladder; see §4).

### Canonical shapes (copy these)

```python
# 1) Map a pure function over a range
from burla import remote_parallel_map

def square(x: int) -> dict:
    return {"input": x, "square": x * x}

results = remote_parallel_map(square, list(range(100_000)))
```

```python
# 2) Bucket a big range into chunks (tuple inputs unpack to *args)
from burla import remote_parallel_map

def count_primes(lo, hi):            # note: two args, not one
    count = 0
    for n in range(max(2, lo), hi):
        is_prime = True
        i = 2
        while i * i <= n:
            if n % i == 0:
                is_prime = False
                break
            i += 1
        if is_prime:
            count += 1
    return {"range": [lo, hi], "primes": count}

buckets = [(i, i + 2000) for i in range(0, 1_000_000, 2000)]
results = remote_parallel_map(count_primes, buckets)     # tuples → *args
total = sum(r["primes"] for r in results)
```

```python
# 3) Stream results as they finish (unordered)
from burla import remote_parallel_map

for result in remote_parallel_map(process, inputs, generator=True):
    handle(result)
```

```python
# 4) Request a GPU per worker + a custom image + auto-scaling
from burla import remote_parallel_map

results = remote_parallel_map(
    infer,
    prompts,
    func_gpu="H100_80G",
    image="us-central1-docker.pkg.dev/my-proj/my-repo/vllm:latest",
    grow=True,
    max_parallelism=32,
)
```

```python
# 5) Fire-and-forget a long job from a short-lived shell
from burla import remote_parallel_map

remote_parallel_map(crawl, urls, detach=True)   # returns immediately
```

### Worker-function requirements (strict)

- Defined at **module top level** in the file you run.
- No lambdas, no inner functions, no closures over state that won't pickle.
- Total pickled size `< 100 MB`.
- Imports the worker needs should be either at module top level or inside the function body.
- Return values must pickle cleanly (no open file handles, no threading locks, no DB connections).

---

## 3. Commands in this kit — exhaustive reference

### `python onboard.py` — bring an account from zero to working

```bash
python onboard.py --email you@example.com [--demo demos/square.py] [--auth-provider google]
```


| Flag              | Required                | Default  | Purpose                                                     |
| ----------------- | ----------------------- | -------- | ----------------------------------------------------------- |
| `--email`         | yes (or `$BURLA_EMAIL`) | —        | Email of the Burla account to onboard.                      |
| `--demo`          | no                      | none     | Path to a Python script to run after onboarding succeeds.   |
| `--auth-provider` | no                      | `google` | Only `google` is implemented today; `microsoft` is stubbed. |


**What it does:**

1. Reads `~/.burla/<slug>/user_config.json`, or creates it.
2. If not already authenticated → launches Chromium, auto-clicks "Continue with Google", waits for you to type the Google password, auto-clicks "Authorize" for the CLI.
3. Writes cluster URL + project ID to `~/.burla/<slug>/user_config.json` and `~/.burla/<slug>/.env`.
4. Builds or repairs `~/.burla/<slug>/.venv/` to match the cluster's required Python + burla client version (learned from the cluster's own error messages — never hard-coded).
5. Ensures the cluster is `READY` (`grow=True` first, then a Playwright-driven dashboard Start click if needed).
6. If `--demo` was passed, runs it in the venv and prints `demo completed` on success.

**Exit codes:** `0` = success, `2` = demo not found, other = step failure (full stderr).

### `python run_job.py` — run your own script

```bash
python run_job.py --email you@example.com path/to/your_script.py [-- script-args ...]
```

Equivalent to `~/.burla/<slug>/.venv/bin/python path/to/your_script.py`. Use this so jobs always run under the pinned client version.

You can also invoke the venv directly:

```bash
~/.burla/<slug>/.venv/bin/python path/to/your_script.py
```

where `<slug>` is the email local-part lowercased with non-alphanumerics removed (`jane.doe@example.com` → `janedoe`).

### `scripts/bootstrap.sh` — one-shot bootstrap for fresh machines

```bash
export BURLA_EMAIL=you@example.com
./scripts/bootstrap.sh
```

Installs top-level deps, installs the Playwright Chromium binary, then runs `onboard.py --email $BURLA_EMAIL --demo demos/square.py`.

### The upstream `burla` CLI (the kit wraps this for you)

This kit never requires an agent to call the raw `burla` CLI directly — the onboarder runs it with the right flags automatically. Reference only:


| Command         | What it does                                                                                                                                                                                                                                | When the kit calls it                                                                                                                |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `burla install` | Deploys a self-hosted Burla cluster into the current `gcloud` project (prerequisites: `gcloud` CLI installed + logged in via `gcloud auth login` and `gcloud auth application-default login`). Re-running it updates the existing install.  | **Never from this kit** — only if a user is standing up a brand-new self-hosted cluster. Not part of agent onboarding.               |
| `burla login`   | Opens the "Authorize this Machine" page in the browser for the cluster the user was most recently logged into in the dashboard. On click, writes an auth token to `burla_credentials.json` in the OS user-data dir. Refreshed on every run. | Automatically, during step 1 of `onboard.py`, with `--no_browser` so the kit can scrape the auth URL and Playwright-click Authorize. |
| `<any> --help`  | Prints CLI docs.                                                                                                                                                                                                                            | Debugging only.                                                                                                                      |


Agents should prefer `python onboard.py --email <them>` over `burla login` directly; the kit handles the browser dance, token placement, and cluster-readiness checks as one atomic step.

---

## 4. Self-healing behavior — error → action table

You do **not** need to react to any of these errors yourself. The kit classifies and repairs them. This table exists so an agent knows what the kit will do so it doesn't "helpfully" intervene.


| Symptom / error string the cluster returns                                                           | Kit classifies as  | Kit's automatic action                                                                                                                                                                   |
| ---------------------------------------------------------------------------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Trivial probe job succeeds                                                                           | `OK`               | Done — cluster is reachable and versions match.                                                                                                                                          |
| `TypeError` mentioning `grow`                                                                        | `NO_GROW_KWARG`    | Installed client predates the `grow` kwarg (e.g. `burla==1.4.5`). Falls through to a plain probe. (Modern clients accept `grow=False` by default; this branch only fires on older pins.) |
| `VersionMismatch` / `NodeConflict` / "No compatible containers" / "cluster supports clients v… - v…" | `VERSION_MISMATCH` | Parse the required Python + burla versions from the error, rebuild `~/.burla/<slug>/.venv/` with matching Python, `pip install burla==<required>`.                                       |
| `NoNodes` / "Zero nodes are ready" / `hit "⏻ Start"` / `ClientConnectorError` to `localhost:5001`    | `CLUSTER_OFF`      | Launch the saved Playwright profile, click the dashboard **Start** button (it will already be signed in), poll until at least one node is `READY`.                                       |
| Anything else                                                                                        | `UNKNOWN`          | Print last 800 chars of the error and abort loudly.                                                                                                                                      |


The loop in `ensure_cluster_ready()` retries up to **6 iterations** combining all the above. Clusters cold-start in ~90 seconds.

---

## 5. Per-account state — what lives where

Nothing user-specific lives in this repo. Everything lands under `~/.burla/<slug>/`:


| Path                                                         | Contents                                                                                                                     | Agent may edit?                                     |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `~/.burla/<slug>/user_config.json`                           | Non-secret: cluster URL, project id, pinned client versions, email.                                                          | Yes, when provisioning that account.                |
| `~/.burla/<slug>/.env`                                       | Email + cluster URL. `chmod 600`.                                                                                            | Yes, when provisioning.                             |
| `~/.burla/<slug>/chrome-profile/`                            | Playwright session (Google cookie).                                                                                          | **Never commit.** Safe to delete to force re-login. |
| `~/.burla/<slug>/.venv/`                                     | Auto-provisioned venv matching cluster versions.                                                                             | **Never commit.** Safe to delete to force rebuild.  |
| `~/Library/Application Support/burla/burla_credentials.json` | Written by `burla login` — auth token + cluster URL. **Global** — only the most recently logged-in account is authoritative. | Safe to delete to force re-auth.                    |


All of these are `.gitignore`d. The `<slug>` is the email local-part lowercased with non-alphanumerics removed (`jane.doe@example.com` → `janedoe`). Multiple accounts coexist under `~/.burla/`.

---

## 6. Environment variables


| Var                   | Default       | Meaning                                                                               |
| --------------------- | ------------- | ------------------------------------------------------------------------------------- |
| `BURLA_EMAIL`         | — (required)  | Email of the account to onboard or run a job for.                                     |
| `BURLA_AUTH_PROVIDER` | `google`      | Future: `microsoft`. Only `google` is implemented today.                              |
| `BURLA_URL`           | auto-detected | Override the cluster dashboard URL (usually auto-read from `burla_credentials.json`). |
| `PYTHON`              | `python3`     | Interpreter used by `scripts/bootstrap.sh`.                                           |


Set them inline, in `.env.example` → `.env`, or export in your shell.

---

## 7. Quick start (humans)

```bash
git clone https://github.com/Burla-Cloud/burla-agent-starter-kit.git
cd burla-agent-starter-kit

pip install -r requirements.txt
python -m playwright install chromium

python onboard.py --email you@example.com --demo demos/square.py
```

**First run:** a Chromium window opens. Sign in to Google when prompted (you'll see a `Waiting for Google sign-in` log line). That's the only moment of human intervention.

**Every subsequent run:** zero intervention — the saved Playwright profile keeps you signed in, cluster state is auto-detected, and demos run.

### Making `/burla` available in every Cursor workspace

The kit ships a Cursor skill at `.cursor/skills/burla/SKILL.md`. When you open the kit as a Cursor workspace it registers as `/burla` automatically. To make `/burla` available from **any** Cursor workspace on this machine, symlink it into your user-scope skills directory:

```bash
mkdir -p ~/.cursor/skills-cursor
ln -sfn "$(pwd)/.cursor/skills/burla" ~/.cursor/skills-cursor/burla
```

`scripts/bootstrap.sh` does this for you when run after cloning.

---

## 8. Rules for agents (hard constraints)

- **Never** type, store, or attempt to automate a Google/OAuth password. Password entry must remain with the human — Google OAuth intentionally forbids it and so does this kit.
- **Never** commit `.env`, `chrome-profile/`, `.venv/`, `burla_credentials.json`, or anything under `~/.burla/` to any git repo.
- **Never** bake user-specific emails, cluster URLs, or project IDs into this repo's tracked files (README, source, demos). Those values belong in `~/.burla/<slug>/user_config.json` only.
- **Never** hard-code Python or `burla` client versions in new code. Let `burla_kit/probe.py` parse them from the cluster's error messages.
- **Never** push to `https://github.com/Burla-Cloud/burla` (that is the upstream Burla platform, not this kit).
- **Never** modify another tenant's cluster. Cluster changes only happen on the authenticated user's own cluster.
- **Always** validate any new demo locally (`run_job.py`) and watch for `REMOTE_OK` before pushing it to GitHub.
- **Always** prefer re-running `onboard.py` (idempotent) over manual repair when something breaks.

---

## 9. Recipes (copy-paste workflows for agents)

### A. Onboard a new user from zero

1. Ask for their email (never their password).
2. Install deps: `pip install -r requirements.txt && python -m playwright install chromium`.
3. Run `python onboard.py --email <them> --demo demos/square.py`.
4. When you see the log `Waiting for Google sign-in …`, tell the user to enter their password in the Chromium window.
5. Wait for `demo completed`. Done.

### B. Run a Burla job for an already-onboarded user

```bash
python run_job.py --email <them> path/to/script.py
```

Or, if `--email` is awkward:

```bash
export BURLA_EMAIL=<them>
python run_job.py path/to/script.py
```

### C. Author a new demo / user script

1. Put the file in `demos/` (if it belongs to this kit) or in a sibling repo.
2. Define worker functions at **module top level**. No lambdas, no closures.
3. Structure:
  ```python
   from burla import remote_parallel_map

   def work(item):
       # must be picklable; imports inside are auto-installed on workers
       return {"in": item, "out": ...}

   def main():
       results = remote_parallel_map(work, list(range(N)))
       print("REMOTE_OK")
       print(f"n={len(results)}")

   if __name__ == "__main__":
       main()
  ```
4. Run `python run_job.py --email <them> demos/your_script.py` and confirm `REMOTE_OK` prints.
5. Only then consider pushing to GitHub.

### D. Re-verify / repair a broken environment

```bash
python onboard.py --email <them>          # idempotent; fixes most things
```

If that still fails:

```bash
rm -rf ~/.burla/<slug>/.venv              # force venv rebuild
rm -f  ~/Library/Application\ Support/burla/burla_credentials.json   # force re-auth
python onboard.py --email <them>
```

Full wipe for that account:

```bash
rm -rf ~/.burla/<slug>
rm -f  ~/Library/Application\ Support/burla/burla_credentials.json
```

---

## 10. Troubleshooting decision tree

Always try in this order before editing code:

1. **Any error at all** → `python onboard.py --email <them>` (idempotent). Stop here if it succeeds.
2. `NoNodes` / `ClientConnectorError` / "cluster off" → onboarder's UI Start fallback should handle it. If not, click **Start** at the dashboard URL in `~/.burla/<slug>/user_config.json`.
3. `VersionMismatch` / `NodeConflict` → probe rebuilds the venv automatically; if it keeps failing, `rm -rf ~/.burla/<slug>/.venv` and re-run onboard.
4. Auth / token errors → `rm -f ~/Library/Application\ Support/burla/burla_credentials.json` and re-run onboard.
5. Worker-side errors about missing system libs (ffmpeg, libGL, etc.) → cluster container image needs updating via the dashboard Settings (a cluster-admin action).

If the same error repeats twice in a row after these steps, stop and report the exact error with a one-line hypothesis rather than guessing further.

---

## 11. What's in the repo

```
burla-agent-starter-kit/
├── README.md               # you are here
├── onboard.py              # CLI entry: python onboard.py --email you@example.com
├── run_job.py              # CLI entry: run any script in the provisioned venv
├── requirements.txt        # playwright + python-dotenv (onboarder deps only)
├── scripts/
│   └── bootstrap.sh        # install Playwright + run onboard + demo
├── demos/
│   ├── square.py           # 100k squares (sanity check)
│   ├── primes.py           # primes under 1M across 500 buckets
│   └── wordlen.py          # 50k synthetic sentences → avg word length
└── burla_kit/              # internals — agents should not need to read these
    ├── onboard.py          # orchestrates auth → cluster → venv → demo
    ├── auth.py             # Playwright: Continue-with-Google + Authorize
    ├── cluster.py          # grow=True → UI Start escalation ladder
    ├── probe.py            # version detection, error classification
    ├── venv.py             # per-account ~/.burla/<slug>/.venv manager
    ├── config.py           # per-account user_config.json / .env I/O
    └── logging.py          # coloured, flush-on-write logger
```

If you're an agent: **you should never need to read anything in `burla_kit/`** to use this repo. If you find yourself needing to, the README is missing something — prefer to report that gap over guessing at internals.

---

## 12. Known limitations

- Microsoft OAuth is stubbed but not implemented in `burla_kit/auth.py`.
- Dashboard "Start" button selectors are best-effort. If the dashboard relabels its button, add the new label to `START_BUTTON_LABELS` in `burla_kit/cluster.py`.
- A pure-API cluster-start path (to skip Playwright in step 3) is a planned optimization.

---

## License

MIT. See `LICENSE`.