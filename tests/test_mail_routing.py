"""Mails : destinataire unique BOURSO_MAIL_1 + mail de synthese des PEA (sans envoi)."""
import src.bourso.accounts as A
import src.bourso.notify as N
import src.real_bourso as rb


def _no_dotenv(monkeypatch, **env):
    monkeypatch.setattr(A, "_read_dotenv", lambda: {})
    for k in ("BOURSO_MAIL_1", "BOURSO_MAIL_2"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def test_all_mails_go_to_mail_1(monkeypatch):
    _no_dotenv(monkeypatch, BOURSO_MAIL_1="one@x.fr", BOURSO_MAIL_2="two@x.fr")
    assert N.default_recipients() == ["one@x.fr"]
    acc2 = A.BoursoAccount(slot=2, bourso_id="i", bourso_code="c", mail="two@x.fr")
    assert acc2.recipients == ["one@x.fr"]


def test_fallback_mailing_list_when_mail_1_empty(monkeypatch):
    _no_dotenv(monkeypatch)
    assert N.default_recipients() == list(N.MAILING_LIST)


def test_digest_mail_summarises_all_accounts():
    accs = [A.BoursoAccount(slot=n, bourso_id="i", bourso_code="c", name="PEA X") for n in (1, 2, 3)]
    entries = [
        {"account": accs[0], "subject": "s1", "body": "DETAIL 1", "short": "ACHAT 507x PUST",
         "stocks": 0.0, "cash": 54686.0, "stocks_after": 52000.0, "cash_after": 2400.0,
         "e_before": 0.0, "e_after": 1.02, "n_done": 2},
        {"account": accs[1], "subject": "s2", "body": "DETAIL 2", "short": "aucun changement",
         "stocks": 53227.0, "cash": 100.0, "e_before": 1.0, "e_after": 1.0},
        {"account": accs[2], "subject": "s3", "body": "DETAIL 3", "short": "CONNEXION KO — aucun ordre",
         "stocks": None, "cash": None, "e_before": None, "e_after": None},
    ]
    signal = {"date": "2026-11-10", "allocation": 0.51, "rsi14": 40.1}
    subject, body = rb.digest_mail(entries, "LIVE", 1.017, signal)
    assert subject.startswith("[MyQTM] LIVE 3 PEA — expo cible 102%")
    assert subject == ("[MyQTM] LIVE 3 PEA — expo cible 102% — expo reelle c1 102% | c2 100% | c3 KO"
                       " — EXECUTION c1")
    assert "TOTAL 2 PEA" in body
    assert "Avant execution : titres 53 227 EUR | especes 54 786 EUR | total 108 013 EUR" in body
    assert "Apres execution : titres 105 227 EUR | especes 2 500 EUR | total 107 727 EUR" in body
    # compte sans execution : une seule ligne
    assert "      titres 53 227 EUR | especes 100 EUR | total 53 327 EUR" in body
    assert "expo 0% -> 102%" in body
    for d in ("DETAIL 1", "DETAIL 2", "DETAIL 3"):
        assert d in body


def test_flush_sends_one_mail(monkeypatch):
    sent = []
    monkeypatch.setattr(N, "send_email", lambda subject, body, **kw: sent.append(subject))
    acc = A.BoursoAccount(slot=1, bourso_id="i", bourso_code="c", name="PEA X")
    for _ in range(3):
        rb._send_account_email(acc, "s", "b", short="aucun changement", stocks=1.0, cash=0.0,
                               e_before=1.0, e_after=1.0)
    rb.flush_mail_digest("DRY-RUN", 1.0, {"date": "d", "allocation": 0.5})
    assert len(sent) == 1 and sent[0].endswith("expo reelle c1 100% | c1 100% | c1 100% — aucune execution")
    assert rb._MAIL_DIGEST == []


# ── changement de cible avant -> apres ───────────────────
def _sig(hist, exposure, date="2026-11-12"):
    return {"date": date, "exposure": exposure,
            "exposure_history": [{"date": d, "exposure": e} for d, e in hist]}


def test_target_change_big_move_in_subject_and_body():
    from src.real_bourso import target_change
    sig = _sig([("2026-11-10", 1.017), ("2026-11-11", 1.02), ("2026-11-12", 0.40)], 0.40)
    subj, line = target_change(sig)
    assert subj == "expo cible 102% -> 40%"
    assert "avant 102% (signal du 2026-11-11) -> apres 40% (signal du 2026-11-12), -62 pts" in line
    assert "102% / 102% / 40%" in line


def test_target_change_small_move_and_unchanged():
    from src.real_bourso import target_change
    subj, line = target_change(_sig([("2026-11-11", 1.00), ("2026-11-12", 1.02)], 1.02))
    assert subj == "expo cible 102%"            # < 5 pts : objet sans fleche
    assert "-> apres 102%" in line and "+2 pts" in line
    subj, line = target_change(_sig([("2026-11-11", 1.02), ("2026-11-12", 1.02)], 1.02))
    assert "inchangee" in line


def test_target_change_forced_target_and_legacy_signal():
    from src.real_bourso import target_change
    subj, _ = target_change(_sig([("2026-11-11", 1.02), ("2026-11-12", 1.02)], 1.02), target_e=0.0)
    assert subj == "expo cible 102% -> 0%"      # arret d'urgence / garde-fou
    subj, line = target_change({"date": "2026-11-12", "exposure": 0.4})
    assert subj == "expo cible 40%" and line == "Exposition cible : 40%"


def test_band_rule_in_morning_mail():
    from src.real_bourso import digest_mail, BAND_NOTE
    acc = A.BoursoAccount(slot=1, bourso_id="i", bourso_code="c", name="PEA X")
    e = [dict(account=acc, body="d", short="aucun changement", stocks=100.0, cash=0.0,
              e_before=1.02, e_after=1.02, n_done=0, n_error=0)]
    _, body = digest_mail(e, "LIVE", 0.40, {"date": "2026-11-12", "exposure": 0.40})
    assert BAND_NOTE in body and "au moins 100 pts" in BAND_NOTE and "au moins 50 pts" in BAND_NOTE
