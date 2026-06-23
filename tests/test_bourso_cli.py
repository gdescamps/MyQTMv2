"""Tests dry-run du binaire bourso-cli.

Ces tests verifient que bourso-cli est installe et fonctionne, SANS jamais
passer d'ordre reel ni s'authentifier : on s'appuie sur l'introspection du
binaire (--version, --help, aide des sous-commandes) et sur la commande
`quote` (non authentifiee). Aucun ordre d'achat/vente n'est emis.
"""

import re

import pytest

from tests.conftest import pinned_version, run_cli

# Sous-commandes attendues dans l'aide de bourso-cli v0.5.x
EXPECTED_COMMANDS = ["accounts", "config", "trade", "quote", "export", "transfer"]


def test_binary_present(bourso_cli):
    """Le binaire existe et repond a --version."""
    rc, out, err = run_cli(bourso_cli, "--version")
    assert rc == 0, f"--version a echoue: {err}"
    assert "bourso" in (out + err).lower()


def test_version_parses(bourso_cli):
    """La version suit le format 'bourso X.Y.Z'."""
    rc, out, err = run_cli(bourso_cli, "--version")
    m = re.search(r"(\d+\.\d+\.\d+)", out + err)
    assert m, f"version introuvable dans: {out + err!r}"


def test_version_matches_submodule_pin(bourso_cli):
    """Le binaire installe correspond a la version epinglee dans le submodule.

    Garde-fou: detecte un binaire perime par rapport au pin du submodule
    (relancer ./bourso_cli_update.sh pour resynchroniser).
    """
    pin = pinned_version()
    if pin is None:
        pytest.skip("submodule external/bourso-api absent (git submodule update --init)")
    rc, out, err = run_cli(bourso_cli, "--version")
    installed = re.search(r"(\d+\.\d+\.\d+)", out + err).group(1)
    assert installed == pin, (
        f"binaire v{installed} != pin submodule v{pin} "
        f"— relancer ./bourso_cli_update.sh"
    )


def test_help_runs(bourso_cli):
    """--help sort en code 0."""
    rc, out, err = run_cli(bourso_cli, "--help")
    assert rc == 0


def test_help_lists_expected_commands(bourso_cli):
    """L'aide liste bien toutes les sous-commandes attendues."""
    rc, out, err = run_cli(bourso_cli, "--help")
    text = out + err
    missing = [c for c in EXPECTED_COMMANDS if c not in text]
    assert not missing, f"sous-commandes manquantes dans l'aide: {missing}"


@pytest.mark.parametrize("subcommand", ["quote", "trade", "accounts", "export"])
def test_subcommand_help(bourso_cli, subcommand):
    """L'aide de chaque sous-commande fonctionne (dry-run, aucune action)."""
    rc, out, err = run_cli(bourso_cli, subcommand, "--help")
    assert rc == 0, f"'{subcommand} --help' a echoue: {err}"


@pytest.mark.network
@pytest.mark.xfail(
    reason="endpoint quote Boursorama renvoie 410 Gone en v0.5.3 (corrige en amont)",
    strict=False,
)
def test_quote_live(bourso_cli):
    """Cotation reelle non authentifiee — bourso-cli doit retourner un prix.

    Marque network + xfail: en v0.5.3 l'API Boursorama renvoie 410 Gone.
    Si un build plus recent corrige le scraping, ce test passera (xpass) et
    signalera que la cotation refonctionne.
    """
    try:
        rc, out, err = run_cli(bourso_cli, "quote", "--symbol", "1rTPUST",
                               "--length", "30", "last", timeout=30)
    except Exception as e:  # pragma: no cover - reseau indisponible
        pytest.skip(f"reseau indisponible: {e}")
    assert rc == 0, f"quote a echoue (rc={rc}): {err}"
    assert re.search(r"\d+[.,]\d+", out + err), "aucun prix dans la sortie"
