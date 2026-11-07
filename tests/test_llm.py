"""Tests des LLM Anthropic (Claude) via Amazon Bedrock.

Client : ``anthropic.AnthropicBedrock`` (``bedrock-runtime`` InvokeModel,
``https://bedrock-runtime.<region>.amazonaws.com/model/<id>/invoke``), signé
SigV4 avec les credentials du ``.env`` :

    BEDROCK_ACCESS_KEY_ID      clé d'accès IAM
    BEDROCK_SECRET_ACCESS_KEY  secret IAM
    BEDROCK_REGION             région AWS (ex. eu-west-3, us-east-1)

Deux familles de tests :
- **hors ligne** (toujours exécutés) : construction du client + requête
  interceptée par un ``httpx2.MockTransport`` -> vérifie l'URL, la signature
  SigV4, le corps envoyé et le parsing de la réponse, sans réseau ni coût ;
- **live** (marqueur ``llm``) : vrais appels Bedrock, ignorés si une des trois
  variables manque. Modèle par défaut = profil d'inférence géographique
  ``<geo>.anthropic.claude-opus-5-5`` (``eu.`` pour une région ``eu-*``),
  surchargeable via ``BEDROCK_MODEL``.

Pourquoi pas ``AnthropicBedrockMantle`` (le client recommandé par le SDK) : en
eu-west-1 l'endpoint ``bedrock-mantle`` répond 404 « model does not exist » pour
tous les Claude 5.x (testé le 2026-11-06), alors que ``bedrock-runtime`` les sert
via les profils d'inférence ``eu.anthropic.*`` (les modèles ne sont exposés qu'en
``INFERENCE_PROFILE``, pas en appel direct ``anthropic.*``).

    pytest tests/test_llm.py                 # tout (live ignorés sans credentials)
    pytest tests/test_llm.py -m "not llm"    # hors ligne seulement
"""

import base64
import io
import json
import os

import anthropic
import httpx2
import pytest
from dotenv import load_dotenv

load_dotenv()

ENV_KEYS = ("BEDROCK_ACCESS_KEY_ID", "BEDROCK_SECRET_ACCESS_KEY", "BEDROCK_REGION")
BASE_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")


def geo_prefix(region):
    """Préfixe du profil d'inférence cross-region pour une région AWS."""
    region = (region or "").lower()
    if region.startswith("eu-"):
        return "eu"
    if region.startswith(("us-", "ca-")):
        return "us"
    if region.startswith("ap-"):
        return "apac"
    return "global"


def model_id(base, region=None):
    """``claude-opus-5-5`` -> ``eu.anthropic.claude-opus-5-5`` (selon la région)."""
    region = region if region is not None else os.environ.get("BEDROCK_REGION")
    return f"{geo_prefix(region)}.anthropic.{base}"


DEFAULT_MODEL = model_id("claude-opus-5-5", "eu-west-1")


def make_client(http_client=None, **overrides):
    """Client Bedrock (Mantle) construit à partir des variables BEDROCK_*."""
    kwargs = dict(
        aws_access_key=os.environ.get("BEDROCK_ACCESS_KEY_ID"),
        aws_secret_key=os.environ.get("BEDROCK_SECRET_ACCESS_KEY"),
        aws_region=os.environ.get("BEDROCK_REGION"),
    )
    kwargs.update(overrides)
    if http_client is not None:
        kwargs["http_client"] = http_client
    return anthropic.AnthropicBedrock(**kwargs)


def text_of(message):
    """Concatène les blocs texte d'une réponse (ignore les blocs thinking)."""
    return "".join(b.text for b in message.content if b.type == "text")


# --------------------------------------------------------------------------
# Hors ligne : transport simulé
# --------------------------------------------------------------------------

FAKE_ENV = {
    "BEDROCK_ACCESS_KEY_ID": "AKIATESTFAKEKEY",
    "BEDROCK_SECRET_ACCESS_KEY": "fake-secret",
    "BEDROCK_REGION": "eu-west-1",
}


def fake_message(text="OK", model=DEFAULT_MODEL, stop_reason="end_turn"):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }


@pytest.fixture
def fake_env(monkeypatch):
    for k, v in FAKE_ENV.items():
        monkeypatch.setenv(k, v)
    # Évite qu'un profil / des credentials AWS ambiants interfèrent.
    for k in ("AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
              "AWS_SESSION_TOKEN", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def mock_bedrock(fake_env):
    """Client branché sur un MockTransport ; renvoie (client, requêtes vues, réponses)."""
    seen = []
    responses = []

    def handler(request):
        seen.append(request)
        status, payload = responses.pop(0) if responses else (200, fake_message())
        return httpx2.Response(status, json=payload)

    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    client = make_client(http_client=http, max_retries=0)
    return client, seen, responses


@pytest.mark.parametrize("region,expected", [
    ("eu-west-1", "eu.anthropic.claude-opus-5-5"),
    ("eu-central-1", "eu.anthropic.claude-opus-5-5"),
    ("us-east-1", "us.anthropic.claude-opus-5-5"),
    ("ap-northeast-1", "apac.anthropic.claude-opus-5-5"),
    ("sa-east-1", "global.anthropic.claude-opus-5-5"),
])
def test_model_id_geo_prefix(region, expected):
    assert model_id("claude-opus-5-5", region) == expected


def test_client_uses_bedrock_env(fake_env):
    client = make_client()
    assert isinstance(client, anthropic.AnthropicBedrock)
    assert str(client.base_url).rstrip("/") == "https://bedrock-runtime.eu-west-1.amazonaws.com"


def test_request_is_sigv4_signed_to_runtime_endpoint(mock_bedrock):
    client, seen, _ = mock_bedrock
    client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=16,
        messages=[{"role": "user", "content": "ping"}],
    )
    assert len(seen) == 1
    req = seen[0]
    assert str(req.url) == (
        "https://bedrock-runtime.eu-west-1.amazonaws.com/model/eu.anthropic.claude-opus-5-5/invoke"
    )
    auth = req.headers["authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=AKIATESTFAKEKEY/")
    assert "/eu-west-1/bedrock/aws4_request" in auth
    assert "x-amz-date" in req.headers
    assert "x-api-key" not in req.headers  # pas de clé API Anthropic sur Bedrock


def test_request_body(mock_bedrock):
    client, seen, _ = mock_bedrock
    client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=32,
        system="Tu es concis.",
        messages=[{"role": "user", "content": "Bonjour"}],
    )
    body = json.loads(seen[0].content)
    assert "model" not in body  # sur bedrock-runtime le modèle est dans l'URL
    assert body["anthropic_version"]  # injecté par le client Bedrock
    assert body["max_tokens"] == 32
    assert body["system"] == "Tu es concis."
    assert body["messages"] == [{"role": "user", "content": "Bonjour"}]


def test_response_parsing(mock_bedrock):
    client, _, responses = mock_bedrock
    responses.append((200, fake_message(text="Paris")))
    msg = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=16,
        messages=[{"role": "user", "content": "Capitale de la France ?"}],
    )
    assert text_of(msg) == "Paris"
    assert msg.stop_reason == "end_turn"
    assert msg.usage.input_tokens == 5


@pytest.mark.parametrize("status,exc", [
    (400, anthropic.BadRequestError),
    (403, anthropic.PermissionDeniedError),
    (404, anthropic.NotFoundError),
    (429, anthropic.RateLimitError),
])
def test_http_errors_map_to_sdk_exceptions(mock_bedrock, status, exc):
    client, _, responses = mock_bedrock
    responses.append((status, {"type": "error", "error": {"type": "x", "message": "boom"}}))
    with pytest.raises(exc):
        client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=16,
            messages=[{"role": "user", "content": "ping"}],
        )


# --------------------------------------------------------------------------
# Live : vrais appels Bedrock (coût réel, faible : quelques dizaines de tokens)
# --------------------------------------------------------------------------

_missing = [k for k in ENV_KEYS if not os.environ.get(k)]
live = pytest.mark.skipif(bool(_missing), reason=f"variables .env manquantes : {_missing}")


@pytest.fixture(scope="module")
def live_client():
    return make_client(timeout=120.0)


@pytest.fixture(scope="module")
def live_model():
    return os.environ.get("BEDROCK_MODEL") or model_id("claude-opus-5-5")


def _check_not_refused(msg):
    assert msg.stop_reason != "refusal", f"refus : {msg.stop_details}"


@pytest.mark.llm
@live
@pytest.mark.parametrize("base", BASE_MODELS)
def test_live_basic_message(live_client, base):
    msg = live_client.messages.create(
        model=model_id(base),
        max_tokens=1024,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": "Réponds uniquement par le mot : OK"}],
    )
    _check_not_refused(msg)
    assert msg.stop_reason == "end_turn"
    assert "OK" in text_of(msg).upper()
    assert msg.usage.input_tokens > 0 and msg.usage.output_tokens > 0


@pytest.mark.llm
@live
def test_live_system_and_multiturn(live_client, live_model):
    history = [
        {"role": "user", "content": "Retiens ce code : 4271. Réponds juste 'noté'."},
    ]
    first = live_client.messages.create(
        model=live_model, max_tokens=1024, output_config={"effort": "low"},
        system="Tu réponds en français, très brièvement.", messages=history,
    )
    _check_not_refused(first)
    history.append({"role": "assistant", "content": first.content})
    history.append({"role": "user", "content": "Quel était le code ? Chiffres seulement."})
    second = live_client.messages.create(
        model=live_model, max_tokens=1024, output_config={"effort": "low"},
        system="Tu réponds en français, très brièvement.", messages=history,
    )
    _check_not_refused(second)
    assert "4271" in text_of(second)


@pytest.mark.llm
@live
def test_live_streaming(live_client, live_model):
    chunks = []
    with live_client.messages.stream(
        model=live_model, max_tokens=1024, output_config={"effort": "low"},
        messages=[{"role": "user", "content": "Compte de 1 à 5, séparés par des espaces."}],
    ) as stream:
        for text in stream.text_stream:
            chunks.append(text)
        final = stream.get_final_message()
    _check_not_refused(final)
    out = "".join(chunks)
    assert out == text_of(final)
    for n in "12345":
        assert n in out


@pytest.mark.llm
@live
def test_live_structured_output(live_client, live_model):
    schema = {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "allocation": {"type": "number"},
        },
        "required": ["ticker", "allocation"],
        "additionalProperties": False,
    }
    msg = live_client.messages.create(
        model=live_model, max_tokens=1024, output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": schema},
        },
        messages=[{"role": "user", "content": "Ticker QQQ, allocation 0.85. Renvoie le JSON."}],
    )
    _check_not_refused(msg)
    data = json.loads(text_of(msg))
    assert data["ticker"] == "QQQ"
    assert data["allocation"] == pytest.approx(0.85)


@pytest.mark.llm
@live
def test_live_bad_credentials_rejected(live_model):
    client = make_client(aws_access_key="AKIAINVALIDKEY0000", aws_secret_key="invalid", max_retries=0)
    with pytest.raises((anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        client.messages.create(
            model=live_model, max_tokens=16,
            messages=[{"role": "user", "content": "ping"}],
        )


def make_test_png(text="73", color="red"):
    """PNG en mémoire : fond uni ``color`` + ``text`` en grand, en blanc."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(3, 3), dpi=100, facecolor=color)
    fig.text(0.5, 0.5, text, color="white", fontsize=96, fontweight="bold",
             ha="center", va="center")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=color)
    plt.close(fig)
    return buf.getvalue()


def test_make_test_png_is_valid():
    png = make_test_png()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.llm
@live
@pytest.mark.parametrize("base", BASE_MODELS)
def test_live_image_input(live_client, base):
    data = base64.standard_b64encode(make_test_png("73", "red")).decode()
    msg = live_client.messages.create(
        model=model_id(base),
        max_tokens=1024,
        output_config={"effort": "low"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                {"type": "text", "text": (
                    "Quel nombre est écrit sur cette image et quelle est la couleur du fond ? "
                    "Réponds au format : <nombre> <couleur en anglais>, rien d'autre."
                )},
            ],
        }],
    )
    _check_not_refused(msg)
    out = text_of(msg).lower()
    print(f"\n[{base}] image -> {out!r}")
    assert "73" in out
    assert "red" in out


# --------------------------------------------------------------------------
# Embeddings : Amazon Titan Text Embeddings V2 (pas d'embeddings chez Anthropic)
# Appel via boto3 bedrock-runtime.invoke_model, mêmes credentials BEDROCK_*.
# --------------------------------------------------------------------------

EMBED_MODEL = "amazon.titan-embed-text-v2:0"


def make_runtime_client():
    import boto3
    return boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("BEDROCK_REGION"),
        aws_access_key_id=os.environ.get("BEDROCK_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("BEDROCK_SECRET_ACCESS_KEY"),
    )


def embed(runtime, text, dimensions=512):
    """Embedding Titan V2 normalisé (norme L2 = 1) d'un texte."""
    body = {"inputText": text, "dimensions": dimensions, "normalize": True}
    resp = runtime.invoke_model(
        modelId=EMBED_MODEL, body=json.dumps(body),
        contentType="application/json", accept="application/json",
    )
    return json.loads(resp["body"].read())["embedding"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


def test_cosine_basics():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([1, 2], [-1, -2]) == pytest.approx(-1.0)


def test_embed_request_body():
    """Hors ligne : corps envoyé à invoke_model + lecture de la réponse."""
    calls = []

    class FakeRuntime:
        def invoke_model(self, **kw):
            calls.append(kw)
            return {"body": io.BytesIO(json.dumps({"embedding": [0.6, 0.8]}).encode())}

    vec = embed(FakeRuntime(), "bonjour", dimensions=256)
    assert vec == [0.6, 0.8]
    assert calls[0]["modelId"] == EMBED_MODEL
    assert json.loads(calls[0]["body"]) == {"inputText": "bonjour", "dimensions": 256, "normalize": True}


@pytest.fixture(scope="module")
def runtime():
    return make_runtime_client()


@pytest.mark.llm
@live
@pytest.mark.parametrize("dimensions", [256, 512, 1024])
def test_live_embedding_shape(runtime, dimensions):
    vec = embed(runtime, "Le Nasdaq-100 a fortement progressé.", dimensions)
    assert len(vec) == dimensions
    assert sum(x * x for x in vec) ** 0.5 == pytest.approx(1.0, abs=1e-3)


# (texte A, texte B similaire, texte C différent)
SIMILARITY_CASES = [
    ("Le marché actions américain a chuté fortement aujourd'hui.",
     "Wall Street a enregistré une forte baisse en séance.",
     "La recette de la tarte aux pommes demande du beurre et de la cannelle."),
    ("La Réserve fédérale a relevé ses taux directeurs pour lutter contre l'inflation.",
     "The Federal Reserve raised interest rates to fight inflation.",  # même sens, autre langue
     "Le chat dort au soleil sur le rebord de la fenêtre."),
    ("Vendre les ETF à effet de levier quand la volatilité explose.",
     "Réduire l'exposition aux fonds levier lors des pics de volatilité.",
     "Le match de football s'est terminé sur un score nul."),
]


@pytest.mark.llm
@live
@pytest.mark.parametrize("a,similar,different", SIMILARITY_CASES)
def test_live_embedding_similar_vs_different(runtime, a, similar, different):
    ea, es, ed = (embed(runtime, t) for t in (a, similar, different))
    sim_close = cosine(ea, es)
    sim_far = cosine(ea, ed)
    print(f"\n  similaire={sim_close:.3f}  different={sim_far:.3f}  | {a[:50]}")
    # Titan V2 (512 d) : paraphrases ~0.50-0.57, sujets sans rapport ~0.04-0.09
    # (mesuré le 2026-11-06) -> marges larges pour rester stable.
    assert sim_close > sim_far + 0.25
    assert sim_close > 0.4
    assert sim_far < 0.25


@pytest.mark.llm
@live
def test_live_embedding_identical_text(runtime):
    t = "Allocation cible 85 % sur QQQ."
    assert cosine(embed(runtime, t), embed(runtime, t)) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.llm
@live
def test_live_embedding_ranking(runtime):
    """Recherche sémantique : la requête doit retrouver le bon document en tête."""
    docs = [
        "Le stop suiveur coupe l'exposition sous la moyenne mobile 250 jours.",
        "Les dividendes sont réinvestis automatiquement dans le PEA.",
        "Le chauffage au bois est économique en hiver.",
        "L'indice NFCI mesure les conditions financières aux États-Unis.",
    ]
    query = "Quel indicateur suit le stress financier américain ?"
    eq = embed(runtime, query)
    scores = [cosine(eq, embed(runtime, d)) for d in docs]
    print("\n  " + "\n  ".join(f"{s:.3f}  {d}" for s, d in sorted(zip(scores, docs), reverse=True)))
    assert max(range(len(docs)), key=scores.__getitem__) == 3
