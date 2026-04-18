"""Minimal console logger — one place for styled, flush-on-write output."""

import sys
import time

_GREEN = "\033[32m"
_BLUE = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _now() -> str:
    return time.strftime("%H:%M:%S")


def info(msg: str) -> None:
    print(f"{_DIM}[{_now()}]{_RESET} {msg}", flush=True)


def step(tag: str, msg: str) -> None:
    print(f"{_DIM}[{_now()}]{_RESET} {_BOLD}{_BLUE}{tag}{_RESET} {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"{_DIM}[{_now()}]{_RESET} {_GREEN}✓{_RESET} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"{_DIM}[{_now()}]{_RESET} {_YELLOW}!{_RESET} {msg}", flush=True)


def err(msg: str) -> None:
    print(f"{_DIM}[{_now()}]{_RESET} {_RED}✗{_RESET} {msg}", file=sys.stderr, flush=True)


def banner(msg: str) -> None:
    bar = "─" * (len(msg) + 4)
    print(f"\n{_BOLD}{bar}\n  {msg}  \n{bar}{_RESET}\n", flush=True)
