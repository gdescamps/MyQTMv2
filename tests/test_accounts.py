"""Tests du multicompte Bourso (src/bourso/accounts.py) — aucun binaire, aucun reseau.

  - lecture des slots BOURSO_ID_n / CODE_n / MAIL_n : un slot n'est gere que si id ET
    code sont renseignes ; compat de l'ancienne paire BOURSO_ID/BOURSO_CODE = slot 1
  - parsing de la sortie Debug de `bourso-cli accounts --trading` + choix du PEA
  - real_bourso : la cle de prix de reference (split) est par compte, repli legacy slot 1
"""

import json

import pytest

from src.bourso import accounts as A


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Pas de .env, pas de cache, pas d'env BOURSO_* herite."""
    monkeypatch.setattr(A, "ENV_PATH", tmp_path / "nope.env")
    monkeypatch.setattr(A, "ACCOUNTS_CACHE", tmp_path / "accounts.json")
    for k in list(__import__("os").environ):
        if k.startswith("BOURSO_"):
            monkeypatch.delenv(k)


def test_slots_with_id_and_code_only():
    env = {
        "BOURSO_ID_1": "111", "BOURSO_CODE_1": "aaa", "BOURSO_MAIL_1": "one@x.fr",
        "BOURSO_ID_2": "", "BOURSO_CODE_2": "", "BOURSO_MAIL_2": "",
        "BOURSO_ID_3": "333", "BOURSO_CODE_3": "", "BOURSO_MAIL_3": "three@x.fr",   # code vide -> ignore
        "BOURSO_ID_4": "444", "BOURSO_CODE_4": "ddd", "BOURSO_MAIL_4": "",
    }
    accs = A.load_accounts(env)
    assert [a.slot for a in accs] == [1, 4]
    assert accs[0].mail == "one@x.fr" and accs[0].creds == ("111", "aaa")
    assert accs[1].mail == "" and accs[1].recipients   # repli MAILING_LIST


def test_legacy_pair_is_slot_1():
    accs = A.load_accounts({"BOURSO_ID": "999", "BOURSO_CODE": "zzz"})
    assert len(accs) == 1 and accs[0].slot == 1 and accs[0].creds == ("999", "zzz")


def test_slot_1_prefers_suffixed_vars():
    accs = A.load_accounts({"BOURSO_ID": "old", "BOURSO_CODE": "old",
                            "BOURSO_ID_1": "new", "BOURSO_CODE_1": "new"})
    assert accs[0].creds == ("new", "new")


def test_no_account_when_empty():
    assert A.load_accounts({}) == []


def test_env_override_and_cache(tmp_path):
    A.ACCOUNTS_CACHE.write_text(json.dumps({"2": {"pea_account_id": "cafe" * 8, "name": "PEA X"}}))
    accs = A.load_accounts({"BOURSO_ID_2": "2", "BOURSO_CODE_2": "2",
                            "BOURSO_ID_3": "3", "BOURSO_CODE_3": "3", "BOURSO_PEA_ID_3": "beef" * 8})
    assert accs[0].pea_account_id == "cafe" * 8 and accs[0].name == "PEA X"
    assert accs[1].pea_account_id == "beef" * 8 and accs[1].name == ""
    assert A.load_cached_accounts() == {2: {"pea_account_id": "cafe" * 8, "name": "PEA X"}}


DEBUG_OUTPUT = '''
 INFO bourso_cli: Login successful ✅
 INFO bourso_cli: Found 3 accounts
[
    Account {
        id: "e0aeafb04e60bdbe140479e499fd79d2",
        name: "ORDINAIRE DESCAMPS",
        balance: 12345,
        bank_name: "BoursoBank",
        kind: Trading,
    },
    Account {
        id: "1111111111111111111111111111aaaa",
        name: "PEA-PME DESCAMPS",
        balance: 0,
        bank_name: "BoursoBank",
        kind: Trading,
    },
    Account {
        id: "faab190372918f26c5d2d518fd307d05",
        name: "PEA DESCAMPS",
        balance: 5439364,
        bank_name: "BoursoBank",
        kind: Trading,
    },
]
'''


def test_parse_accounts_debug_output():
    rows = A.parse_accounts_output(DEBUG_OUTPUT)
    assert [r["name"] for r in rows] == ["ORDINAIRE DESCAMPS", "PEA-PME DESCAMPS", "PEA DESCAMPS"]
    assert rows[2]["id"] == "faab190372918f26c5d2d518fd307d05"
    assert rows[2]["balance"] == 54393.64 and rows[2]["kind"] == "Trading"


def test_pick_pea_skips_pme_and_cto():
    pea = A.pick_pea(A.parse_accounts_output(DEBUG_OUTPUT))
    assert pea["name"] == "PEA DESCAMPS"
    assert A.pick_pea([{"id": "x", "name": "ORDINAIRE"}]) is None


def test_resolve_pea_uses_cli_once_then_cache(monkeypatch):
    calls = []

    def fake_cli(*args, creds=None):
        calls.append((args, creds))
        return DEBUG_OUTPUT, "", 0
    monkeypatch.setattr("src.bourso.prepare._run_cli_raw", fake_cli)
    acc = A.BoursoAccount(slot=2, bourso_id="id2", bourso_code="c2", mail="m@x.fr")
    A.resolve_pea(acc)
    assert acc.pea_account_id == "faab190372918f26c5d2d518fd307d05" and acc.name == "PEA DESCAMPS"
    assert calls == [(("accounts", "--trading", "1"), ("id2", "c2"))]
    # cache ecrit -> un nouveau load ne rappelle pas le CLI
    accs = A.load_accounts({"BOURSO_ID_2": "id2", "BOURSO_CODE_2": "c2"})
    A.resolve_pea(accs[0])
    assert len(calls) == 1 and accs[0].name == "PEA DESCAMPS"


def test_resolve_pea_raises_without_pea(monkeypatch):
    monkeypatch.setattr("src.bourso.prepare._run_cli_raw",
                        lambda *a, creds=None: ('[Account { id: "abc", name: "ORDINAIRE", balance: 1, '
                                                'bank_name: "B", kind: Trading, }]', "", 0))
    with pytest.raises(RuntimeError, match="aucun PEA"):
        A.resolve_pea(A.BoursoAccount(slot=3, bourso_id="i", bourso_code="c"))


# ── real_bourso : prix de reference (split) par compte ────
def test_last_price_key_per_account(monkeypatch, tmp_path):
    import src.real_bourso as rb
    monkeypatch.setattr(rb, "LAST_PRICE_FILE", tmp_path / "last_price.json")
    # ancien fichier mono-compte : cle = instrument -> lu comme slot 1 uniquement
    (tmp_path / "last_price.json").write_text(json.dumps({"LQQ": 9.8}))
    assert rb.load_last_price("LQQ", 1) == 9.8
    assert rb.load_last_price("LQQ", 2) is None
    rb.save_last_price("LQQ", 10.1, slot=2)
    rb.save_last_price("LQQ", 10.2, slot=1)
    data = json.loads((tmp_path / "last_price.json").read_text())
    assert data == {"LQQ": 10.2, "LQQ#2": 10.1, "LQQ#1": 10.2}
    assert rb.load_last_price("LQQ", 2) == 10.1
