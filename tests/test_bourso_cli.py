"""Tests dry-run du binaire bourso-cli.

But: identifier IMMEDIATEMENT qu'un build de bourso-cli est casse, SANS jamais
passer d'ordre reel ni s'authentifier. On s'appuie sur l'introspection du
binaire (--version, --help, aide de chaque sous-commande), l'absence de panic
Rust, et la commande `quote` (non authentifiee). Aucun ordre n'est emis.

Le submodule `external/bourso-api` est epingle sur le DERNIER commit de `main`
(et non sur le tag v0.5.3, anterieur a la commande `export`). `export` fait
donc partie des commandes attendues : si le binaire installe ne l'expose pas,
c'est que le build est perime/casse -> le test echoue.
"""

import re

import pytest

from tests.conftest import pinned_version, run_cli

# Sous-commandes attendues du dernier commit main (inclut `export`, ajoute apres v0.5.3)
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
    (relancer ./2_bourso_cli_update.sh pour resynchroniser).
    """
    pin = pinned_version()
    if pin is None:
        pytest.skip("submodule external/bourso-api absent (git submodule update --init)")
    rc, out, err = run_cli(bourso_cli, "--version")
    installed = re.search(r"(\d+\.\d+\.\d+)", out + err).group(1)
    assert installed == pin, (
        f"binaire v{installed} != pin submodule v{pin} "
        f"— relancer ./2_bourso_cli_update.sh"
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


@pytest.mark.parametrize("subcommand", EXPECTED_COMMANDS)
def test_subcommand_help(bourso_cli, subcommand):
    """L'aide de CHAQUE sous-commande fonctionne (dry-run, aucune action).

    Un build casse fait souvent planter une sous-commande precise : ce test
    parametre sur toutes les commandes attendues les couvre une a une.
    """
    rc, out, err = run_cli(bourso_cli, subcommand, "--help")
    assert rc == 0, f"'{subcommand} --help' a echoue: {err}"


def test_no_rust_panic(bourso_cli):
    """Le binaire ne doit jamais paniquer (build sain) face a une commande invalide.

    Une commande inconnue doit sortir proprement (usage clap, rc=2) et NE PAS
    afficher un panic Rust ('panicked at') ni un backtrace — symptome d'un
    binaire corrompu/casse.
    """
    rc, out, err = run_cli(bourso_cli, "commande-bidon-inexistante")
    combined = (out + err).lower()
    assert "panicked" not in combined, f"panic Rust detecte: {out + err}"
    assert "backtrace" not in combined, f"backtrace detecte: {out + err}"
    assert rc != 0, "une commande invalide devrait sortir en erreur"


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
