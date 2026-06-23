"""Tests unitaires des wrappers Python autour de bourso-cli (dry-run pur).

Aucun appel binaire ni reseau : on verifie la logique d'extraction JSON
qui parse la sortie (souvent bruitee) de bourso-cli.
"""

from src.bourso.prepare import _extract_json


def test_extract_json_from_response_error():
    """La CLI Rust echoue parfois a deserialiser et renvoie le JSON brut
    dans 'Response: {...}' — on doit savoir le recuperer."""
    text = (
        " INFO bourso_cli: Welcome\n"
        'Error: deserialization failed. Response: {"resourceId": "abc", '
        '"position": {"cash": 1000.0, "quantity": 3}}'
    )
    data = _extract_json(text)
    assert data is not None
    assert data["resourceId"] == "abc"
    assert data["position"]["quantity"] == 3


def test_extract_json_plain_object():
    """Un JSON nu precede de lignes de log doit etre extrait."""
    text = ' INFO log line\n INFO autre\n{"symbol": {"lastPrice": 42.5}}'
    data = _extract_json(text)
    assert data["symbol"]["lastPrice"] == 42.5


def test_extract_json_array():
    """Le premier tableau JSON est extrait si pas d'objet."""
    text = "INFO header\n[1, 2, 3]"
    data = _extract_json(text)
    assert data == [1, 2, 3]


def test_extract_json_none_when_absent():
    """Sans JSON, retourne None plutot que de lever."""
    assert _extract_json("aucun json ici") is None
