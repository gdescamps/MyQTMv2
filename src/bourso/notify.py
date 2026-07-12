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


def send_recap():
    """Send evening recap email with signal info and backtest chart."""
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

    # Predit l'action du matin en REUTILISANT le vrai chemin d'execution
    # (real_bourso) : meme instrument actif (TRADE_INSTRUMENT: PUST x1 / LQQ x2),
    # meme bande asymetrique et meme force cash. Source unique -> le recap ne peut
    # plus diverger de ce que le cron du matin fera reellement (avant : PUST en dur
    # + ancienne logique de seuil -> prevoyait des parts PUST meme en LQQ).
    action_preview = ""
    try:
        from src.real_bourso import (
            get_pea_state, compute_orders, INSTRUMENT, INSTRUMENTS, LEVERAGE,
        )
        state = get_pea_state()
        price = state["etf_price"]
        shares = state["etf_shares"]
        equity = state["equity"]
        current_alloc = (shares * price) / equity if equity > 0 else 0
        side, qty, reason = compute_orders(alloc, state)

        label = INSTRUMENTS[INSTRUMENT]["label"]
        if side == "buy":
            action_preview = f"ACHAT {qty} parts {INSTRUMENT} prevu demain matin"
        elif side == "sell":
            action_preview = f"VENTE {qty} parts {INSTRUMENT} prevue demain matin"
        else:
            action_preview = f"Pas de changement prevu demain matin ({reason})"

        # exposition visee vs realisee (utile en LQQ : 1 part = 2x le poids)
        realized_w = ((shares + (qty if side == "buy" else -qty if side == "sell" else 0))
                      * price) / equity if equity > 0 else 0
        action_preview += (
            f"\n  Instrument: {INSTRUMENT} ({label}, levier x{LEVERAGE:.0f})"
            f"\n  PEA actuel: {shares} parts, {state['cash']:.0f} EUR especes, "
            f"alloc poids {current_alloc*100:.0f}% -> {alloc_pct}"
            f"\n  Exposition: {current_alloc*LEVERAGE*100:.0f}% -> {realized_w*LEVERAGE*100:.0f}% "
            f"(cible {alloc*LEVERAGE*100:.0f}%)"
        )
    except Exception as e:
        action_preview = f"(impossible de verifier le PEA: {e})"

    subject = f"[MyQTM] {ticker} {date} — alloc {alloc_pct} (prob {prob:.3f})"
    body = (
        f"Backtest {ticker} termine avec succes.\n\n"
        f"  Date:        {date}\n"
        f"  Probabilite: {prob:.4f}\n"
        f"  Allocation:  {alloc_pct}\n\n"
        f"{action_preview}\n"
    )

    images = []
    if BACKTEST_1M.exists():
        images.append(BACKTEST_1M)

    send_email(subject, body, images=images)
    return True


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
