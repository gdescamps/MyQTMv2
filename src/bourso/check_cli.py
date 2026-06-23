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

import subprocess
import sys
from pathlib import Path

from src.bourso.notify import send_email

ROOT = Path(__file__).resolve().parent.parent.parent
SUBMODULE = ROOT / "external" / "bourso-api"
UPSTREAM_BRANCH = "origin/main"


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


def check_upstream():
    """Compare la version epinglee a l'amont. Retourne un dict de statut."""
    info = {
        "pinned_commit": None, "pinned_version": None,
        "latest_tag": None, "new_commits": 0, "log": "", "error": None,
    }
    try:
        info["pinned_commit"] = _git("rev-parse", "--short", "HEAD")
        cargo = SUBMODULE / "Cargo.toml"
        for line in cargo.read_text().splitlines():
            if line.strip().startswith("version"):
                info["pinned_version"] = line.split('"')[1]
                break

        # Recuperer commits + tags amont (necessite le reseau)
        _git("fetch", "--tags", "--quiet", "origin")
        info["latest_tag"] = _git(
            "tag", "-l", "v*", "--sort=-v:refname"
        ).splitlines()[0]

        count = _git("rev-list", "--count", f"HEAD..{UPSTREAM_BRANCH}")
        info["new_commits"] = int(count)
        if info["new_commits"] > 0:
            # Inclure le message complet (sujet + corps) de chaque nouveau commit
            info["log"] = _git(
                "log", "--no-decorate", "--date=short",
                "--format=- %h (%an, %ad): %s%n%w(0,4,4)%b",
                f"HEAD..{UPSTREAM_BRANCH}", "-n", "20",
            )
    except Exception as e:
        info["error"] = str(e)
    return info


def build_report(tests_ok, tests_summary, up):
    """Assemble (sujet, corps, alerte) du mail.

    Deux causes d'alerte, reflechies dans le sujet :
      - build casse  : les tests dry-run echouent sur le binaire installe ;
      - nouveau commit amont : un commit est apparu au-dela du pin courant.
    """
    build_broken = not tests_ok
    has_new = up.get("new_commits", 0) > 0
    alert = build_broken or has_new or bool(up.get("error"))

    # Sujet priorisant le probleme le plus grave
    if build_broken:
        flag = "BUILD CASSE"
    elif has_new:
        n = up["new_commits"]
        flag = f"{n} nouveau commit" + ("s" if n > 1 else "")
    elif up.get("error"):
        flag = "check amont impossible"
    else:
        flag = "OK"

    tests_line = "REUSSIS" if tests_ok else "ECHEC — build casse, NE PAS deployer"

    if up.get("error"):
        up_block = f"Verification amont IMPOSSIBLE: {up['error']}"
    elif has_new:
        up_block = (
            f"{up['new_commits']} nouveau(x) commit(s) amont au-dela du pin "
            f"(commit {up['pinned_commit']}, v{up['pinned_version']}).\n\n"
            f"Messages des commits:\n{up['log']}\n"
            f"Pour mettre a jour manuellement:\n"
            f"  ./2_bourso_cli_update.sh --pull        # checkout dernier commit main + build + install\n"
            f"  git add external/bourso-api && git commit -m 'chore: bump bourso-cli'"
        )
    else:
        up_block = (
            f"A jour: le pin (commit {up['pinned_commit']}, v{up['pinned_version']}) "
            f"= dernier commit de main. Dernier tag publie: {up['latest_tag']}."
        )

    subject = f"[MyQTM] bourso-cli — {flag}"
    body = (
        f"Verification quotidienne de bourso-cli\n"
        f"{'=' * 40}\n\n"
        f"1. Build installe (tests dry-run): {tests_line}\n"
        f"   {tests_summary}\n\n"
        f"2. Depot amont (azerpas/bourso-api):\n"
        f"   {up_block}\n"
    )
    return subject, body, alert


def main():
    tests_ok, tests_summary = run_tests()
    up = check_upstream()
    subject, body, alert = build_report(tests_ok, tests_summary, up)

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
