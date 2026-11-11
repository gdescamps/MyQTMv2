"""Send email notifications via Gmail SMTP.

Requires GMAIL_APP_PASSWORD in .env (16-char app password from Google).
Generate at: https://myaccount.google.com/apppasswords

Usage:
  python -m src.bourso.notify "Sujet" "Corps du message"
  python -m src.bourso.notify --test
  python -m src.bourso.notify --recap   # recap backtest du soir
"""

import json
import os
import smtplib
import sys
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

SENDER = "descamps.gregory@gmail.com"
MAILING_LIST = [
    "descamps.gregory@gmail.com",
]
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

ROOT = Path(__file__).resolve().parent.parent.parent
SIGNAL_PATH = ROOT / "outputs" / "qqq_strategy" / "signal.json"
BACKTEST_1M = ROOT / "outputs" / "qqq_strategy" / "backtest_1m.png"


def _get_app_password():
    """Load GMAIL_APP_PASSWORD from environment or .env."""
    pwd = os.environ.get("GMAIL_APP_PASSWORD")
    if pwd:
        return pwd
    env_path = ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GMAIL_APP_PASSWORD="):
                    return line.split("=", 1)[1].strip('"').strip("'")
    raise RuntimeError(
        "GMAIL_APP_PASSWORD manquant. Ajouter dans .env ou en variable d'env.\n"
        "Generer sur: https://myaccount.google.com/apppasswords"
    )


def default_recipients():
    """Destinataire UNIQUE de tous les mails (execution du matin de chaque compte, recap
    du soir, alertes check_cli, erreurs) : BOURSO_MAIL_1 (env ou .env), repli MAILING_LIST
    s'il est vide. Les BOURSO_MAIL_n des autres slots ne recoivent plus rien."""
    from src.bourso.accounts import _env, _read_dotenv
    mail = _env("BOURSO_MAIL_1", _read_dotenv())
    return [mail] if mail else list(MAILING_LIST)


def send_email(subject, body, to=None, images=None):
    """Send an email via Gmail SMTP, with optional inline images.

    to: single address or list. Defaults to default_recipients() (BOURSO_MAIL_1).
    images: list of file paths to embed inline in the email body.
    """
    if to is None:
        to = default_recipients()
    if isinstance(to, str):
        to = [to]

    pwd = _get_app_password()

    if images:
        msg = MIMEMultipart("related")
        html = "<html><body>"
        html += body.replace("\n", "<br>")
        for i, path in enumerate(images):
            cid = f"img{i}"
            html += f'<br><img src="cid:{cid}" style="max-width:100%;"><br>'
        html += "</body></html>"
        msg.attach(MIMEText(html, "html", "utf-8"))
        for i, path in enumerate(images):
            p = Path(path)
            if p.exists():
                with open(p, "rb") as f:
                    img = MIMEImage(f.read())
                img.add_header("Content-ID", f"<img{i}>")
                img.add_header("Content-Disposition", "inline", filename=p.name)
                msg.attach(img)
    else:
        msg = MIMEText(body, "plain", "utf-8")

    msg["Subject"] = subject
    msg["From"] = SENDER
    msg["To"] = ", ".join(to)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SENDER, pwd)
        server.send_message(msg)

    print(f"Email envoye a {', '.join(to)}: {subject}")


EMERGENCY_LINE = "  ARRET D'URGENCE ACTIF — aucun achat tant qu'il n'est pas desactive"


def _account_preview(account, alloc, emergency=False):
    """Apercu de l'action du matin pour UN compte, en REUTILISANT le vrai chemin
    d'execution (real_bourso) : meme plafond d'exposition, meme bande asymetrique,
    meme force cash, meme apport detecte, meme emergency-off (cible forcee a 0).
    Source unique -> le recap ne peut pas diverger de ce que le cron du matin fera
    reellement (hors apport detecte le matin meme). Renvoie (texte, state)."""
    from src.real_bourso import (
        plan_for_account, exposure_after, describe_order, composition_line,
        target_exposure, E_MAX, STRATEGY_LABEL,
    )
    target_e = 0.0 if emergency else target_exposure(alloc, E_MAX)
    state, plan = plan_for_account(account, target_e)
    if plan["orders"]:
        for o in plan["orders"]:
            o["status"] = "dry-run"
        preview = "Prevu demain matin : " + " ; ".join(describe_order(o) for o in plan["orders"])
    else:
        reason = "arret d'urgence actif, exposition maintenue a 0%" if emergency else plan["reason"]
        preview = f"Pas de changement prevu demain matin ({reason})"
    e_after = exposure_after(plan, state)
    preview += (
        f"\n  Compte:     {state['account_name']} (compte {account.slot}) — connexion OK"
        f"\n  Strategie:  {STRATEGY_LABEL} (PUST x1 + LQQ x2), plafond d'expo x{E_MAX}"
        f"\n  PEA actuel: {composition_line(state)}, {state['cash']:.0f} EUR especes"
        f"\n  Exposition: {plan['e_eff']*100:.0f}% -> {e_after*100:.0f}% "
        + ("(cible forcee a 0% : arret d'urgence)" if emergency else
           f"(cible {target_e*100:.0f}%, levier inclus)")
    )
    if plan.get("deposit", 0) > 0:
        preview += (f"\n  Apport detecte : +{plan['deposit']:.0f} EUR, investi demain matin "
                    f"a la composition cible")
    return preview, state


def send_recap():
    """Recap du soir : UN seul mail (a BOURSO_MAIL_1, cf. default_recipients) = signal +
    graphe + synthese de TOUS les PEA geres (apercu de l'action du matin compte par
    compte + total). Sans compte gere, le recap part sans apercu."""
    if not SIGNAL_PATH.exists():
        print("[ERREUR] signal.json introuvable")
        return False

    with open(SIGNAL_PATH) as f:
        signal = json.load(f)

    status = signal.get("status", "?")
    ticker = signal.get("ticker", "QQQ")
    date = signal.get("date", "?")
    alloc = signal.get("allocation", 0)

    if status != "ok":
        subject = f"[MyQTM] ERREUR backtest {ticker} {date}"
        body = f"Le backtest a echoue (status={status}).\nVerifier logs/cron_backtest.log."
        send_email(subject, body)
        return False

    from src.risk_off_strategy.strategy import E_MAX, target_exposure
    from src.real_bourso import is_emergency_off, EXPO_NOTE, BAND_NOTE, target_change
    emergency = is_emergency_off()
    expo = signal.get("exposure", target_exposure(alloc, E_MAX))
    signal.setdefault("exposure", expo)
    t_subj, t_line = target_change(signal)
    header = (
        (f"{EMERGENCY_LINE}\n  (supprimer logs/emergency_off.json pour reprendre la strategie)\n\n"
         if emergency else "") +
        f"Backtest {ticker} termine avec succes.\n\n"
        f"  Date:        {date}\n"
        f"  Changement de cible a executer demain matin :\n"
        f"  {t_line}\n"
        f"  ({EXPO_NOTE})\n"
        f"  {BAND_NOTE}\n"
        f"  Graphe joint: dernier mois — strategie deployee PUST + LQQ x{E_MAX} (net de frais) vs B&H {ticker}\n\n"
    )
    images = [BACKTEST_1M] if BACKTEST_1M.exists() else []

    from src.bourso.accounts import load_accounts
    accounts = load_accounts()
    if not accounts:
        send_email(f"[MyQTM] {ticker} {date} — {t_subj} demain matin",
                   header + "(aucun compte Bourso gere)\n", images=images)
        return True

    ok = True
    sections, tot_eq, tot_cash, n_ko = [], 0.0, 0.0, 0
    for account in accounts:
        try:
            preview, state = _account_preview(account, alloc, emergency)
            tot_eq += state["equity"]
            tot_cash += state["cash"]
        except Exception as e:  # noqa: BLE001
            preview = (f"  Compte:     {account.label} — CONNEXION KO ({str(e)[:200]})"
                       f"\n  Impossible de verifier le PEA ; l'execution du matin reessaiera.")
            ok = False
            n_ko += 1
        sections.append(f"[compte {account.slot} — {account.name or account.label}]\n{preview}\n")
    eur = lambda x: f"{x:,.0f}".replace(",", " ")  # noqa: E731
    synth = (f"SYNTHESE DES {len(accounts)} PEA — total {eur(tot_eq)} EUR (especes {eur(tot_cash)} EUR)"
             + (f" — {n_ko} compte(s) illisible(s)" if n_ko else "") + "\n\n")
    subject = (f"[MyQTM] {'ARRET D URGENCE — ' if emergency else ''}{ticker} {date} — {t_subj} demain matin "
               f"— {len(accounts)} PEA" + (f" — {n_ko} KO" if n_ko else ""))
    try:
        send_email(subject, header + synth + "\n".join(sections), images=images)
    except Exception as e:  # noqa: BLE001
        print(f"[ERREUR] envoi recap: {e}")
        ok = False
    return ok


if __name__ == "__main__":
    if "--test" in sys.argv:
        send_email("Test MyQTM", "Ceci est un test d'envoi depuis MyQTMv2.")
    elif "--recap" in sys.argv:
        send_recap()
    elif len(sys.argv) >= 3:
        send_email(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python -m src.bourso.notify \"Sujet\" \"Message\"")
        print("       python -m src.bourso.notify --test")
        print("       python -m src.bourso.notify --recap")
        sys.exit(1)
