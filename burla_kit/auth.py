"""Playwright-driven Burla login.

Responsibilities:
  * Launch a Chromium with a per-user persistent profile so the Google
    session is remembered forever after a single sign-in.
  * Auto-click "Continue with Google" on login.burla.dev.
  * If the Google session is valid, we land on the cluster dashboard with
    no further interaction. Otherwise we pause and wait for the user to
    type their password (the ONLY manual step).
  * Run `burla login --no_browser` and auto-click the "Authorize" button
    on backend.burla.dev/v2/login/client/<id>.
  * Parse the CLI's saved credentials to return (cluster_url, project_id).
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from .config import read_burla_credentials, user_dir
from .logging import err, info, ok, step, warn


BURLA_LOGIN_URL = "https://login.burla.dev/"
CLI_AUTH_URL_PATTERN = re.compile(r"https://backend\.burla\.dev/v2/login/client/\S+")


def _profile_dir(email: str) -> Path:
    d = user_dir(email) / "chrome-profile"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _maybe_click(page, selectors: list[str], label: str, timeout_ms: int = 6000) -> bool:
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            await loc.wait_for(timeout=timeout_ms)
            await loc.click()
            info(f"clicked {label} via {sel}")
            return True
        except Exception:
            continue
    return False


async def _wait_for_cluster_dashboard(page, timeout_s: int = 420) -> str:
    """Block until the page URL is a tenant cluster dashboard (not login/backend/google)."""
    last = ""
    prompted = False
    for _ in range(timeout_s // 2):
        try:
            url = page.url
        except Exception:
            url = ""
        if url != last:
            info(f"browser url -> {url}")
            last = url
        on_login = "login.burla.dev" in url
        on_google = "accounts.google." in url or "myaccount.google." in url
        on_backend = "backend.burla.dev" in url
        on_about = url in ("about:blank", "")
        on_cluster = "burla.dev" in url and not (on_login or on_backend or on_about)
        if on_cluster:
            return url
        if (on_login or on_google) and not prompted:
            warn(
                "Waiting for Google sign-in in the Chromium window "
                "(this is the ONLY manual step — the session is saved afterwards)."
            )
            prompted = True
        await asyncio.sleep(2)
    raise RuntimeError("Timed out waiting for Burla cluster dashboard")


async def _click_authorize(page, auth_url: str) -> None:
    info(f"navigating to CLI auth URL: {auth_url}")
    try:
        await page.goto(auth_url, wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
        warn(f"goto warn: {e}")
    await asyncio.sleep(2)
    selectors = [
        'button:has-text("Authorize")',
        'a:has-text("Authorize")',
        '[role="button"]:has-text("Authorize")',
        'text="Authorize"',
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            await loc.wait_for(timeout=8000)
            await loc.click()
            info(f"clicked Authorize via {sel}")
            return
        except Exception:
            continue
    raise RuntimeError("Could not find Authorize button on backend page")


async def run_full_login(email: str) -> Tuple[str, str]:
    """Run the full Playwright + CLI auth flow for `email`.

    Returns (cluster_dashboard_url, project_id).
    """
    from playwright.async_api import async_playwright

    profile = _profile_dir(email)
    info(f"profile={profile}")

    async with async_playwright() as p:
        step("[auth]", "launching Chromium with persistent profile ...")
        ctx = await p.chromium.launch_persistent_context(
            str(profile),
            headless=False,
            viewport={"width": 1280, "height": 800},
            args=["--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"],
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        try:
            await page.bring_to_front()
        except Exception:
            pass

        step("[auth]", f"opening {BURLA_LOGIN_URL}")
        try:
            await page.goto(BURLA_LOGIN_URL, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            warn(f"goto warn: {e}")
        await asyncio.sleep(2)

        await _maybe_click(
            page,
            [
                'button:has-text("Continue with Google")',
                'a:has-text("Continue with Google")',
                'text="Continue with Google"',
            ],
            "Continue with Google",
            timeout_ms=5000,
        )

        cluster_url = await _wait_for_cluster_dashboard(page)
        ok(f"signed into dashboard: {cluster_url}")
        await asyncio.sleep(2)

        step("[auth]", "running `burla login --no_browser` to obtain CLI auth URL ...")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            ["burla", "login", "--no_browser"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        auth_url: Optional[str] = None
        for line in iter(proc.stdout.readline, ""):
            stripped = line.rstrip()
            if stripped:
                info(f"[burla] {stripped}")
            m = CLI_AUTH_URL_PATTERN.search(line)
            if m:
                auth_url = m.group(0).rstrip("?.,)")
                break
        if not auth_url:
            proc.kill()
            raise RuntimeError("Could not find CLI auth URL in `burla login` output")

        try:
            await _click_authorize(page, auth_url)
        finally:
            step("[auth]", "waiting for CLI to save credentials ...")
            try:
                proc.wait(timeout=90)
                info(f"burla login exit code={proc.returncode}")
            except subprocess.TimeoutExpired:
                err("`burla login` timed out; killing.")
                proc.kill()

        await asyncio.sleep(1)
        await ctx.close()

    creds = read_burla_credentials()
    if not creds:
        raise RuntimeError("burla_credentials.json missing after login")
    cluster_url = creds.get("cluster_dashboard_url") or ""
    project_id = creds.get("project_id") or ""
    if not cluster_url.endswith("/"):
        cluster_url = cluster_url + "/"
    ok(f"CLI authenticated as {creds.get('email')} on project={project_id}")
    return cluster_url, project_id
