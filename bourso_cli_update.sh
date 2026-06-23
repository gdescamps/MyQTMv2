#!/usr/bin/env bash
#
# Installe ou met a jour bourso-cli a partir du submodule external/bourso-api.
#
# Le submodule est epingle a une version precise (voir `git -C external/bourso-api describe --tags`).
# Ce script compile cette version-la (Rust/cargo) et copie le binaire dans ~/.local/bin/.
#
# Usage:
#   ./bourso_cli_update.sh            # build le commit epingle du submodule + installe
#   ./bourso_cli_update.sh --pull     # avance le submodule sur le dernier tag upstream, puis build
#
# Apres --pull, pense a committer le nouveau pointeur du submodule:
#   git add external/bourso-api && git commit -m "chore: bump bourso-cli a vX.Y.Z"
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

# 2. Optionnel: avancer sur le dernier tag upstream
if [ "${1:-}" = "--pull" ]; then
    echo "==> Recuperation des tags upstream"
    git -C "$SUBMODULE" fetch --tags --quiet origin
    LATEST_TAG="$(git -C "$SUBMODULE" tag -l 'v*' --sort=-v:refname | head -1)"
    echo "==> Dernier tag upstream: $LATEST_TAG — checkout"
    git -C "$SUBMODULE" checkout --quiet "$LATEST_TAG"
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
