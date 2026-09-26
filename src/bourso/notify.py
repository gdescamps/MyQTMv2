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


def send_email(subject, body, to=None, images=None):
    """Send an email via Gmail SMTP, with optional inline images.

    to: single address or list. Defaults to MAILING_LIST.
    images: list of file paths to embed inline in the email body.
    """
    if to is None:
        to = MAILING_LIST
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


def _account_preview(account, alloc, alloc_pct):
    """Apercu de l'action du matin pour UN compte, en REUTILISANT le vrai chemin
    d'execution (real_bourso) : meme plafond d'exposition, meme bande asymetrique,
    meme force cash, meme reserve DCA. Source unique -> le recap ne peut pas diverger
    de ce que le cron du matin fera reellement (hors apport detecte le matin meme)."""
    from src.real_bourso import (
        plan_for_account, exposure_after, describe_order, composition_line,
        target_exposure, E_MAX, STRATEGY_LABEL,
    )
    target_e = target_exposure(alloc, E_MAX)
    state, plan = plan_for_account(account, target_e)
    if plan["orders"]:
        for o in plan["orders"]:
            o["status"] = "dry-run"
        preview = "Prevu demain matin : " + " ; ".join(describe_order(o) for o in plan["orders"])
    else:
        preview = f"Pas de changement prevu demain matin ({plan['reason']})"
    e_after = exposure_after(plan, state)
    preview += (
        f"\n  Compte:     {state['account_name']} (compte {account.slot}) — connexion OK"
        f"\n  Strategie:  {STRATEGY_LABEL} (PUST x1 + LQQ x2), plafond d'expo x{E_MAX}"
        f"\n  PEA actuel: {composition_line(state, plan['reserved'])}, {state['cash']:.0f} EUR especes"
        f"\n  Exposition: {plan['e_eff']*100:.0f}% -> {e_after*100:.0f}% "
        f"(cible {target_e*100:.0f}% = min(2 x {alloc_pct}, {E_MAX*100:.0f}%))"
    )
    if plan["reserved"] > 0:
        preview += f"\n  Apport en cours de deploiement : {plan['reserved']:.0f} EUR reserves (DCA/RSI)"
    return preview


def send_recap():
    """Recap du soir : signal + graphe, puis UN mail par compte Bourso gere (a l'adresse
    du compte) avec l'apercu de l'action du matin sur CE compte. Sans compte gere,
    le recap part a la MAILING_LIST sans apercu."""
    if not SIGNAL_PATH.exists():
        print("[ERREUR] signal.json introuvable")
        return False

    with open(SIGNAL_PATH) as f:
        signal = json.load(f)

    status = signal.get("status", "?")
    ticker = signal.get("ticker", "QQQ")
    date = signal.get("date", "?")
    prob = signal.get("probability", 0)
    alloc = signal.get("allocation", 0)

    if status != "ok":
        subject = f"[MyQTM] ERREUR backtest {ticker} {date}"
        body = f"Le backtest a echoue (status={status}).\nVerifier logs/cron_backtest.log."
        send_email(subject, body)
        return False

    alloc_pct = f"{alloc*100:.0f}%"
    header = (
        f"Backtest {ticker} termine avec succes.\n\n"
        f"  Date:        {date}\n"
        f"  Probabilite: {prob:.4f}\n"
        f"  Allocation:  {alloc_pct}\n\n"
    )
    images = [BACKTEST_1M] if BACKTEST_1M.exists() else []

    from src.bourso.accounts import load_accounts
    accounts = load_accounts()
    if not accounts:
        send_email(f"[MyQTM] {ticker} {date} — alloc {alloc_pct} (prob {prob:.3f})",
                   header + "(aucun compte Bourso gere)\n", images=images)
        return True

    ok = True
    for account in accounts:
        try:
            preview = _account_preview(account, alloc, alloc_pct)
        except Exception as e:  # noqa: BLE001
            preview = (f"  Compte:     {account.label} — CONNEXION KO ({str(e)[:200]})"
                       f"\n  Impossible de verifier le PEA ; l'execution du matin reessaiera.")
            ok = False
        subject = f"[MyQTM] {ticker} {date} — alloc {alloc_pct} (prob {prob:.3f}) — {account.name or account.label}"
        try:
            send_email(subject, header + preview + "\n", to=account.recipients, images=images)
        except Exception as e:  # noqa: BLE001
            print(f"[ERREUR] envoi recap {account.label}: {e}")
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
