#!/usr/bin/env bash
#
# Installe ou met a jour bourso-cli a partir du submodule external/bourso-api.
#
# Le submodule est epingle sur un commit precis du depot amont (le DERNIER commit
# de la branche `main`, qui peut etre posterieur au dernier tag — la version reste
# affichee v0.5.3 tant que le Cargo.toml n'est pas bumpe en amont).
# Ce script compile ce commit (Rust/cargo) et copie le binaire dans ~/.local/bin/.
#
# Usage:
#   ./2_bourso_cli_update.sh            # build le commit epingle du submodule + installe
#   ./2_bourso_cli_update.sh --pull     # avance le submodule sur le DERNIER commit de main, puis build
#
# Apres --pull, pense a committer le nouveau pointeur du submodule:
#   git add external/bourso-api && git commit -m "chore: bump bourso-cli (<short-sha>)"
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMODULE="$ROOT/external/bourso-api"
DEST="$HOME/.local/bin"

cd "$ROOT"

# 1. S'assurer que le submodule est present (clone initial inclus)
if [ ! -f "$SUBMODULE/Cargo.toml" ]; then
    echo "==> Initialisation du submodule external/bourso-api"
    git submodule update --init external/bourso-api
fi

# 2. Optionnel: avancer sur le dernier commit de main
if [ "${1:-}" = "--pull" ]; then
    echo "==> Recuperation du dernier commit upstream (main)"
    git -C "$SUBMODULE" fetch --tags --quiet origin
    git -C "$SUBMODULE" checkout --quiet origin/main
    echo "==> Submodule sur: $(git -C "$SUBMODULE" rev-parse --short HEAD) — $(git -C "$SUBMODULE" log -1 --format=%s)"
fi

PIN_VER="$(grep -m1 '^version' "$SUBMODULE/Cargo.toml" | sed -E 's/.*"([^"]+)".*/\1/')"
PIN_COMMIT="$(git -C "$SUBMODULE" rev-parse --short HEAD)"
echo "==> Compilation bourso-cli v$PIN_VER (commit $PIN_COMMIT)"

# 3. Build release
command -v cargo >/dev/null 2>&1 || { echo "ERREUR: cargo introuvable (installer Rust: https://rustup.rs)"; exit 1; }
cargo build --release --manifest-path "$SUBMODULE/Cargo.toml"

# 4. Installation
mkdir -p "$DEST"
cp "$SUBMODULE/target/release/bourso-cli" "$DEST/bourso-cli"
echo "==> Installe: $DEST/bourso-cli"

# 5. Verification
INSTALLED_VER="$("$DEST/bourso-cli" --version 2>/dev/null | awk '{print $2}')"
echo "==> Version installee: $INSTALLED_VER (attendu: $PIN_VER)"
if [ "$INSTALLED_VER" != "$PIN_VER" ]; then
    echo "ATTENTION: la version installee ne correspond pas au pin du submodule." >&2
    exit 1
fi
echo "==> OK"
