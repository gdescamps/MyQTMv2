"""Fixtures partagees pour les tests bourso-cli (dry-run)."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SUBMODULE = ROOT / "external" / "bourso-api"


def find_bourso_cli():
    """Localise le binaire bourso-cli (PATH puis ~/.local/bin)."""
    path = shutil.which("bourso-cli")
    if path:
        return path
    fallback = Path.home() / ".local" / "bin" / "bourso-cli"
    if fallback.exists():
        return str(fallback)
    return None


def pinned_version():
    """Version epinglee dans le Cargo.toml du submodule, ou None."""
    cargo = SUBMODULE / "Cargo.toml"
    if not cargo.exists():
        return None
    for line in cargo.read_text().splitlines():
        line = line.strip()
        if line.startswith("version"):
            # version = "0.5.3"
            parts = line.split('"')
            if len(parts) >= 2:
                return parts[1]
    return None


@pytest.fixture(scope="session")
def bourso_cli():
    """Chemin du binaire bourso-cli; skip tout le module s'il est absent."""
    path = find_bourso_cli()
    if not path:
        pytest.skip("bourso-cli introuvable (lancer ./bourso_cli_update.sh)")
    return path


def run_cli(binary, *args, timeout=20):
    """Lance bourso-cli avec des args, retourne (returncode, stdout, stderr)."""
    env = dict(os.environ)
    proc = subprocess.run(
        [binary, *args],
        capture_output=True, text=True, timeout=timeout, env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: test qui contacte Boursorama (peut etre instable)"
    )
