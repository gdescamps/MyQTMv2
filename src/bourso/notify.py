"""Send email notifications via Gmail SMTP.

Requires GMAIL_APP_PASSWORD in .env (16-char app password from Google).
Generate at: https://myaccount.google.com/apppasswords

Usage:
  python -m src.bourso.notify "Sujet" "Corps du message"
  python -m src.bourso.notify --test
"""

import os
import smtplib
import sys
from email.mime.text import MIMEText
from pathlib import Path

EMAIL = "descamps.gregory@gmail.com"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def _get_app_password():
    """Load GMAIL_APP_PASSWORD from environment or .env."""
    pwd = os.environ.get("GMAIL_APP_PASSWORD")
    if pwd:
        return pwd
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
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


def send_email(subject, body, to=EMAIL):
    """Send an email via Gmail SMTP."""
    pwd = _get_app_password()
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = EMAIL
    msg["To"] = to

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL, pwd)
        server.send_message(msg)

    print(f"Email envoye a {to}: {subject}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        send_email("Test MyQTM", "Ceci est un test d'envoi depuis MyQTMv2.")
    elif len(sys.argv) >= 3:
        send_email(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python -m src.bourso.notify \"Sujet\" \"Message\"")
        print("       python -m src.bourso.notify --test")
        sys.exit(1)
