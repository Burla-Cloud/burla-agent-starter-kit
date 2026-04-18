"""Per-user config & secrets helpers.

Everything user-specific lives under ~/.burla/<slug>/ where <slug> is a
filesystem-safe version of the email local-part. The top-level
`burla-agent-starter-kit` repo never stores anything about a specific
tenant; it only reads/writes ~/.burla/<slug>/.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

BURLA_CREDENTIALS_PATH = Path(
    os.path.expanduser("~/Library/Application Support/burla/burla_credentials.json")
)


def email_slug(email: str) -> str:
    local = email.split("@", 1)[0]
    slug = re.sub(r"[^a-z0-9]+", "", local.lower())
    return slug or "user"


def user_dir(email: str) -> Path:
    return Path.home() / ".burla" / email_slug(email)


@dataclass
class UserConfig:
    email: str
    auth_provider: str = "google"
    burla_url: Optional[str] = None
    project_id: Optional[str] = None
    venv_python: Optional[str] = None
    client_python_version: Optional[str] = None
    client_burla_version: Optional[str] = None
    cluster_python_version: Optional[str] = None
    default_first_run_inputs: int = 100_000
    notes: list[str] = field(default_factory=list)


def load_user_config(email: str) -> UserConfig:
    path = user_dir(email) / "user_config.json"
    if not path.exists():
        return UserConfig(email=email)
    data = json.loads(path.read_text())
    data.setdefault("email", email)
    known_fields = {f for f in UserConfig.__annotations__}
    filtered = {k: v for k, v in data.items() if k in known_fields}
    return UserConfig(**filtered)


def save_user_config(cfg: UserConfig) -> Path:
    d = user_dir(cfg.email)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "user_config.json"
    path.write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    return path


def write_env_file(email: str, burla_url: Optional[str], auth_provider: str = "google") -> Path:
    d = user_dir(email)
    d.mkdir(parents=True, exist_ok=True)
    env_path = d / ".env"
    lines = [
        f"BURLA_EMAIL={email}",
        f"BURLA_AUTH_PROVIDER={auth_provider}",
        f"BURLA_URL={burla_url or ''}",
    ]
    env_path.write_text("\n".join(lines) + "\n")
    try:
        env_path.chmod(0o600)
    except Exception:
        pass
    return env_path


def read_burla_credentials() -> dict:
    """Read the burla CLI's saved credentials (written by `burla login`)."""
    if not BURLA_CREDENTIALS_PATH.exists():
        return {}
    return json.loads(BURLA_CREDENTIALS_PATH.read_text())
