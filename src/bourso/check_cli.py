"""Verification quotidienne de bourso-cli (lance par le cron de 20h).

Fait trois choses puis envoie un mail de rapport :
  1. Passe les tests unitaires dry-run de bourso-cli (tests/).
  2. Verifie si de nouveaux commits/tags ont ete pousses sur le depot
     amont (azerpas/bourso-api) au-dela de la version epinglee.
  3. Envoie un email de rapport (statut OK / ALERTE).

Usage:
  python -m src.bourso.check_cli            # tests + check upstream + email
  python -m src.bourso.check_cli --no-email # affiche le rapport sans envoyer
"""

import shutil
import subprocess
import sys
from pathlib import Path

from src.bourso.notify import send_email

ROOT = Path(__file__).resolve().parent.parent.parent
SUBMODULE = ROOT / "external" / "bourso-api"
# remote du depot d'origine (azerpas) — `origin` est notre fork gdescamps
UPSTREAM_REMOTE = "upstream"


def _git(*args, timeout=60):
    """Lance une commande git dans le submodule, retourne stdout (strip)."""
    proc = subprocess.run(
        ["git", "-C", str(SUBMODULE), *args],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def run_tests():
    """Lance les tests dry-run de bourso-cli. Retourne (ok, resume)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300,
    )
    lines = proc.stdout.strip().splitlines()
    summary = lines[-1] if lines else "(pas de sortie)"
    ok = proc.returncode == 0
    if not ok:
        # joindre les dernieres lignes utiles pour le diagnostic
        summary += "\n" + "\n".join(lines[-15:])
    return ok, summary


def check_parse():
    """Rejoue les tests serde Rust du submodule (deserialisation order/prepare).

    Garde-fou contre l'incident du 2026-06-30 : une fois une moins-value realisee
    presente, `order/prepare` renvoie `accountFiscality.realGL` en FLOTTANT
    (ex -20.84). Le champ etait type `i64` -> `bourso-cli order` plantait et
    l'achat du matin ne partait pas (sans alerte, car ce chemin n'est atteint
    qu'a l'execution reelle). Le test Rust `order_prepare_deserializes_with_float_realgl`
    rejoue une vraie reponse order/prepare ; on le lance via `cargo test`.
    Aucun reseau, aucune authentification, aucun ordre.

    Retourne {ok, detail, error, skipped}.
    """
    info = {"ok": False, "detail": None, "error": None, "skipped": False}
    manifest = SUBMODULE / "src" / "bourso_api" / "Cargo.toml"
    if not manifest.exists():
        info.update(ok=True, skipped=True, detail="submodule absent — test ignore")
        return info
    if shutil.which("cargo") is None:
        info.update(ok=True, skipped=True, detail="cargo introuvable — test ignore")
        return info
    try:
        proc = subprocess.run(
            ["cargo", "test", "--manifest-path", str(manifest), "--quiet", "order_prepare"],
            capture_output=True, text=True, timeout=600,
        )
        info["ok"] = proc.returncode == 0
        if info["ok"]:
            info["detail"] = "deserialisation order/prepare OK (realGL flottant)"
        else:
            tail = (proc.stdout + proc.stderr).strip().splitlines()
            info["error"] = "\n".join(tail[-15:]) or "cargo test a echoue"
    except Exception as e:
        info["error"] = str(e)
    return info


def check_account():
    """Verifie la connexion au compte reel — equivalent de src/bourso/prepare.py.

    Lance un vrai `bourso-cli trade summary` (via prepare_order) sur le PEA/PUST :
    cela authentifie aupres de BoursoBank et lit cours + cash + position. C'est
    le meme chemin que le recap du soir et l'execution du matin. Aucun ordre emis.

    Retourne {ok, detail, error}.
    """
    info = {"ok": False, "detail": None, "error": None}
    try:
        from src.bourso.prepare import prepare_order, PEA_ACCOUNT_ID, SYMBOLS
        data = prepare_order(PEA_ACCOUNT_ID, SYMBOLS["PUST"])
        price = data["symbol"]["last_price"]
        cash = data["account"]["cash"]
        qty = data["quantity_held"]
        if not price or price <= 0:
            raise RuntimeError(f"cours PUST invalide: {price!r}")
        if cash is None:
            raise RuntimeError("cash du compte illisible (None)")
        info["ok"] = True
        info["detail"] = (
            f"PUST {price:.4f} EUR — PEA: {cash:.2f} EUR especes, {qty} part(s)"
        )
    except Exception as e:
        info["error"] = str(e)
    return info


def check_upstream():
    """Detecte un nouveau tag amont (azerpas) au-dela du tag de base de notre patch.

    Notre branche `myqtm` (fork) = un tag upstream + le patch `trade summary`.
    On rebase sur les nouveaux tags ; donc l'alerte pertinente = un tag azerpas
    plus recent que le tag de base. Retourne un dict de statut.
    """
    info = {
        "pinned_commit": None, "pinned_version": None, "base_tag": None,
        "latest_tag": None, "new_tag": False, "log": "", "error": None,
    }
    try:
        info["pinned_commit"] = _git("rev-parse", "--short", "HEAD")
        cargo = SUBMODULE / "Cargo.toml"
        for line in cargo.read_text().splitlines():
            if line.strip().startswith("version"):
                info["pinned_version"] = line.split('"')[1]
                break

        # Tag upstream sur lequel notre patch est rebase
        info["base_tag"] = _git("describe", "--tags", "--abbrev=0", "HEAD")

        # Recuperer les tags du depot d'origine azerpas (necessite le reseau)
        _git("fetch", "--tags", "--quiet", UPSTREAM_REMOTE)
        info["latest_tag"] = _git(
            "tag", "-l", "v*", "--sort=-v:refname"
        ).splitlines()[0]

        if info["latest_tag"] != info["base_tag"]:
            info["new_tag"] = True
            # changelog du nouveau tag (messages complets), pour decider du rebase
            info["log"] = _git(
                "log", "--no-decorate", "--date=short",
                "--format=- %h (%an, %ad): %s%n%w(0,4,4)%b",
                f"{info['base_tag']}..{info['latest_tag']}", "-n", "30",
            )
    except Exception as e:
        info["error"] = str(e)
    return info


def build_report(tests_ok, tests_summary, parse, acct, up):
    """Assemble (sujet, corps, alerte) du mail.

    Causes d'alerte, par ordre de gravite (refletees dans le sujet) :
      - build casse        : les tests dry-run echouent sur le binaire installe ;
      - parse CLI casse    : les tests serde Rust (order/prepare) echouent -> un
                             ordre planterait silencieusement a l'execution ;
      - connexion compte KO: le `trade summary` reel sur le PEA echoue (auth/reseau/CLI) ;
      - nouveau tag amont  : azerpas a publie un tag plus recent que notre base.
    """
    build_broken = not tests_ok
    parse_broken = not parse.get("ok")
    account_ko = not acct.get("ok")
    has_new = bool(up.get("new_tag"))
    alert = build_broken or parse_broken or account_ko or has_new or bool(up.get("error"))

    # Sujet priorisant le probleme le plus grave
    if build_broken:
        flag = "BUILD CASSE"
    elif parse_broken:
        flag = "PARSE CLI CASSE"
    elif account_ko:
        flag = "CONNEXION COMPTE KO"
    elif has_new:
        flag = f"nouveau tag {up['latest_tag']}"
    elif up.get("error"):
        flag = "check amont impossible"
    else:
        flag = "OK"

    tests_line = "REUSSIS" if tests_ok else "ECHEC — build casse, NE PAS deployer"
    if parse.get("skipped"):
        parse_line = f"IGNORE — {parse.get('detail')}"
    elif parse.get("ok"):
        parse_line = f"OK — {parse.get('detail')}"
    else:
        parse_line = f"ECHEC — order/prepare ne deserialise plus :\n   {parse.get('error')}"
    if acct.get("ok"):
        acct_line = f"OK — {acct['detail']}"
    else:
        acct_line = f"ECHEC — {acct.get('error')}"

    if up.get("error"):
        up_block = f"Verification amont IMPOSSIBLE: {up['error']}"
    elif has_new:
        up_block = (
            f"Nouveau tag upstream {up['latest_tag']} (notre patch est base sur "
            f"{up['base_tag']}, pin {up['pinned_commit']}).\n\n"
            f"Changelog {up['base_tag']} -> {up['latest_tag']}:\n{up['log']}\n"
            f"Pour rebaser le patch sur ce tag et reinstaller:\n"
            f"  ./2_bourso_cli_update.sh --pull        # rebase myqtm sur le tag + build + install\n"
            f"  cd external/bourso-api && git push origin myqtm --force-with-lease && cd -\n"
            f"  git add external/bourso-api && git commit -m 'chore: bump bourso-cli {up['latest_tag']}'"
        )
    else:
        up_block = (
            f"A jour: patch base sur {up['base_tag']} (pin {up['pinned_commit']}, "
            f"v{up['pinned_version']}) = dernier tag azerpas."
        )

    subject = f"[MyQTM] bourso-cli — {flag}"
    body = (
        f"Verification quotidienne de bourso-cli\n"
        f"{'=' * 40}\n\n"
        f"1. Build installe (tests dry-run): {tests_line}\n"
        f"   {tests_summary}\n\n"
        f"2. Deserialisation CLI (cargo test order/prepare): {parse_line}\n\n"
        f"3. Connexion compte reel (trade summary PEA/PUST): {acct_line}\n\n"
        f"4. Depot amont (azerpas/bourso-api):\n"
        f"   {up_block}\n"
    )
    return subject, body, alert


def main():
    tests_ok, tests_summary = run_tests()
    parse = check_parse()
    acct = check_account()
    up = check_upstream()
    subject, body, alert = build_report(tests_ok, tests_summary, parse, acct, up)

    print(subject)
    print(body)

    if "--no-email" not in sys.argv:
        try:
            send_email(subject, body)
        except Exception as e:
            print(f"[ERREUR] envoi email: {e}")
            return 1
    return 1 if alert else 0


if __name__ == "__main__":
    sys.exit(main())
