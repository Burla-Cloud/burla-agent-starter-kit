"""Cluster power management — the self-healing escalation ladder.

Priority order (cheapest → most invasive):
  1. `remote_parallel_map(grow=True)` — pure client API, best case.
  2. Rebuild venv if we see a Python/burla version mismatch.
  3. Drive the dashboard UI via the already-signed-in Playwright profile
     and click "Start". Zero human interaction.
  4. Poll until at least one node is READY.

The loop terminates when a plain (no grow=True) probe succeeds.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Tuple

from .config import user_dir
from .logging import err, info, ok, step, warn
from .probe import ProbeResult, VersionProbe


START_BUTTON_LABELS = ["Start Cluster", "Start", "Turn On", "Power On", "Boot", "Resume"]


# ---------------------------------------------------------------------------
# Playwright UI Start
# ---------------------------------------------------------------------------


async def _ui_start_async(email: str, dashboard_url: str, wait_ready: bool = True) -> None:
    from playwright.async_api import async_playwright

    profile = user_dir(email) / "chrome-profile"
    if not profile.exists():
        raise RuntimeError(f"No saved Playwright profile at {profile}; run the auth flow first.")

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            str(profile),
            headless=False,
            viewport={"width": 1440, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"],
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        try:
            await page.bring_to_front()
        except Exception:
            pass
        step("[ui]", f"opening dashboard {dashboard_url}")
        try:
            await page.goto(dashboard_url, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            warn(f"goto warn: {e}")
        await asyncio.sleep(3)

        body = await page.inner_text("body")
        if "Running" in body or "RUNNING" in body or ("READY" in body and "Off" not in body):
            ok("cluster already running")
        else:
            clicked = False
            for label in START_BUTTON_LABELS:
                try:
                    btn = page.get_by_role("button", name=label, exact=False).first
                    await btn.wait_for(timeout=2500)
                    await btn.click()
                    ok(f"clicked Start via label={label!r}")
                    clicked = True
                    break
                except Exception:
                    continue
            if not clicked:
                try:
                    btn = page.locator("button:has-text('Start')").first
                    await btn.wait_for(timeout=2500)
                    await btn.click()
                    ok("clicked Start via text=Start")
                    clicked = True
                except Exception:
                    pass
            if not clicked:
                raise RuntimeError("Could not find a Start button on the dashboard")

        if wait_ready:
            step("[ui]", "waiting for at least one node to be READY ...")
            for i in range(120):  # ~4 minutes
                await asyncio.sleep(2)
                try:
                    body = await page.inner_text("body")
                except Exception:
                    continue
                if "READY" in body or "RUNNING" in body:
                    ok("at least one node READY")
                    break
                if i and i % 15 == 0:
                    info("still booting ...")
            else:
                warn("timed out waiting for READY; continuing anyway")

        await asyncio.sleep(1)
        await ctx.close()


def ui_start(email: str, dashboard_url: str) -> None:
    asyncio.run(_ui_start_async(email, dashboard_url))


# ---------------------------------------------------------------------------
# Top-level self-healing loop
# ---------------------------------------------------------------------------


def ensure_cluster_ready(
    email: str,
    dashboard_url: str,
    default_python: str = "3.12",
    default_burla: str = "1.4.5",
    max_iterations: int = 6,
    initial_python: Optional[str] = None,
    initial_burla: Optional[str] = None,
) -> Tuple[str, str]:
    """Idempotently bring the cluster to a state where `remote_parallel_map`
    succeeds on a trivial input. Auto-remediates version mismatches and
    auto-Starts the cluster if needed.

    `initial_python` / `initial_burla` override the hard defaults — callers
    (e.g. onboard.py) should pass the versions previously pinned in
    user_config.json so a working venv isn't needlessly rebuilt or
    downgraded on every invocation.

    Returns (python_version, burla_version) used by the venv.
    """
    probe = VersionProbe(email)
    py, bv = probe.ensure_default_venv(
        initial_python or default_python,
        initial_burla or default_burla,
    )

    ui_started_once = False
    for iteration in range(1, max_iterations + 1):
        step("[cluster]", f"iteration {iteration}: attempting grow=True via client ...")
        result, output = probe.probe(with_grow=True)

        if result == ProbeResult.OK:
            ok(f"cluster reachable via grow=True (python={py}, burla={bv})")
            return py, bv

        if result == ProbeResult.NO_GROW_KWARG:
            info("installed client doesn't support grow=True — running plain probe instead")
            result, output = probe.probe(with_grow=False)
            if result == ProbeResult.OK:
                ok(f"cluster reachable (no grow kwarg; python={py}, burla={bv})")
                return py, bv

        if result == ProbeResult.VERSION_MISMATCH:
            py, bv, changed = probe.remediate_versions(output, py, bv)
            if not changed:
                err("version-mismatch reported but no hint found; aborting")
                print(output[-600:])
                raise RuntimeError("Unparseable version mismatch")
            continue

        if result == ProbeResult.CLUSTER_OFF:
            if ui_started_once:
                warn("cluster still OFF after a UI start; will retry after a short wait")
                import time
                time.sleep(8)
            step("[cluster]", "cluster is OFF — driving dashboard UI Start ...")
            ui_start(email, dashboard_url)
            ui_started_once = True
            continue

        err("probe returned UNKNOWN — last 800 chars below:")
        print(output[-800:])
        raise RuntimeError("Cluster probe failed for an unknown reason")

    raise RuntimeError(f"Cluster not ready after {max_iterations} iterations")
