"""Tests dry-run du binaire bourso-cli.

But: identifier IMMEDIATEMENT qu'un build de bourso-cli est casse, SANS jamais
passer d'ordre reel ni s'authentifier. On s'appuie sur l'introspection du
binaire (--version, --help, aide de chaque sous-commande), l'absence de panic
Rust, et la commande `quote` (non authentifiee). Aucun ordre n'est emis.

Le submodule `external/bourso-api` pointe sur notre fork (branche `myqtm` =
tag v0.5.4 + patchs maison : `trade summary`, options d'ordre). Les commandes
attendues ci-dessous correspondent a ce build ; on verifie aussi que la
sous-commande maison `trade summary` (indispensable a la lecture du PEA) est
bien presente. Si le binaire installe ne les expose pas toutes, c'est que le
build est perime/casse ou non patche -> le test echoue.
"""

import os
import re

import pytest

from tests.conftest import pinned_version, run_cli

# Sous-commandes attendues du tag v0.5.4 (`export` ajoute en amont dans ce tag)
EXPECTED_COMMANDS = ["accounts", "config", "trade", "quote", "transfer", "export"]


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


def test_trade_summary_patch_present(bourso_cli):
    """La sous-commande maison `trade summary` doit etre presente (patch myqtm).

    C'est elle qui permet a `prepare.py`/`get_pea_state` de lire le PEA. Un binaire
    non patche (azerpas vanilla) ne l'expose pas -> ce test echoue immediatement.
    """
    rc, out, err = run_cli(bourso_cli, "trade", "summary", "--help")
    assert rc == 0, f"'trade summary --help' a echoue (binaire non patche ?): {err}"


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
    reason="endpoint quote Boursorama renvoie 410 Gone (toujours KO en v0.5.4)",
    strict=False,
)
def test_quote_live(bourso_cli):
    """Cotation reelle non authentifiee — bourso-cli doit retourner un prix.

    Marque network + xfail: l'API Boursorama renvoie 410 Gone, y compris en
    v0.5.4 (verifie le 2026-07-12, non corrige en amont) — d'ou le fallback
    scrape HTTP de `src/bourso/quote.py`. Si un build plus recent corrige le
    scraping, ce test passera (xpass) et signalera que la cotation refonctionne.
    """
    try:
        rc, out, err = run_cli(bourso_cli, "quote", "--symbol", "1rTPUST",
                               "--length", "30", "last", timeout=30)
    except Exception as e:  # pragma: no cover - reseau indisponible
        pytest.skip(f"reseau indisponible: {e}")
    assert rc == 0, f"quote a echoue (rc={rc}): {err}"
    assert re.search(r"\d+[.,]\d+", out + err), "aucun prix dans la sortie"


@pytest.mark.live
def test_prepare_live():
    """Connexion au compte REEL — equivalent de src/bourso/prepare.py.

    Lance un vrai `bourso-cli trade prepare` (via prepare_order) sur le PEA/PUST :
    authentifie aupres de BoursoBank, lit cours + cash + position. Aucun ordre.
    C'est ce qui valide a 20h que la chaine d'execution du matin marchera.

    DESACTIVE par defaut : ne s'execute que si BOURSO_LIVE_TESTS=1 (defini par le
    cron de 20h), pour ne pas authentifier le compte a chaque `pytest` de dev.
    """
    if not os.environ.get("BOURSO_LIVE_TESTS"):
        pytest.skip("BOURSO_LIVE_TESTS non defini — check compte reel desactive")

    from src.bourso.prepare import prepare_order, PEA_ACCOUNT_ID, SYMBOLS
    data = prepare_order(PEA_ACCOUNT_ID, SYMBOLS["PUST"])

    price = data["symbol"]["last_price"]
    cash = data["account"]["cash"]
    assert price and price > 0, f"cours PUST invalide: {price!r}"
    assert cash is not None, "cash du compte illisible (None)"
