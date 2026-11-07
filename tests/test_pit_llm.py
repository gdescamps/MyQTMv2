"""Tests d'inférence d'un LLM « point-in-time » (PIT) : Diamegs/PIT-4B-FT-201912.

Contexte : RESEARCH.md §8 (fuite de connaissance du LLM). Pour rejouer
l'année N sans biais d'anticipation, on utilise un modèle dont la date de coupure
est au 31/12/N-1. Les modèles de Kelly, Malamud, Schwab & Xu (2026, arXiv
2607.11889) sont publiés une fois par an sur https://huggingface.co/Diamegs.

Modèle testé : ``PIT-4B-FT-201912``. C'est un décodeur 4B (tokenizer GPT-2) affiné
par LoRA pour suivre des instructions, dont la connaissance s'arrête en décembre 2019.
Il ne connaît donc ni le nom « COVID-19 » (février 2020) ni la pandémie.

Format de prompt (fiche du modèle) : ``<|user|>\\n{instruction}\\n<|assistant|>\\n``,
réponse terminée par ``<|end|>`` (ce n'est pas un token spécial : c'est une suite de
tokens BPE, d'où un critère d'arrêt maison).

Code distant : le modèle exige ``trust_remote_code=True`` (``modeling_pit.py`` /
``configuration_pit.py``, relus le 2026-11-06 : imports torch/transformers
uniquement, ni réseau ni écriture de fichier). La révision est figée sur
``PIT_REVISION`` pour qu'un push ultérieur sur le dépôt HF ne puisse pas changer
le code exécuté.

Deux familles de tests :
- **hors ligne** (toujours exécutés, sans torch) : prompt, extraction de la
  réponse, critère d'arrêt, choix du millésime selon l'année rejouée ;
- **inférence réelle** (marqueur ``pit``), ignorée sauf si ``PIT_LLM_TESTS=1`` et
  que torch + transformers sont installés (``pip install -r requirements_pit.txt``).
  Le premier lancement télécharge environ 17,8 Go (poids en float32), chargés en
  bfloat16 (environ 8 Go), sur GPU si disponible.

    pytest tests/test_pit_llm.py                       # hors ligne
    PIT_LLM_TESTS=1 pytest tests/test_pit_llm.py -m pit   # inférence réelle
"""

import os

import pytest

PIT_REPO = "Diamegs/PIT-4B-FT-201912"
PIT_REVISION = "dd482b6a647f86b963626355a8e78d5588f63803"  # commit HF du 2026-10-16
END_MARKER = "<|end|>"
CUTOFF = "2019-12"

# Millésimes dont la date de coupure n'est pas décembre (vu sur la page Diamegs).
VINTAGE_OVERRIDES = {2015: "201511"}
FIRST_VINTAGE, LAST_VINTAGE = 2013, 2024

# Termes postérieurs à décembre 2019 : s'ils apparaissent, le modèle « sait »
# des choses qu'il ne devrait pas savoir.
POST_CUTOFF_TERMS = (
    "sars-cov-2", "omicron", "delta variant", "lockdown", "pfizer", "moderna",
    "biontech", "march 2020", "wuhan lab leak",
)


# --------------------------------------------------------------------------- #
# Helpers (purs, sans torch)
# --------------------------------------------------------------------------- #
def format_prompt(instruction):
    """Gabarit de la fiche du modèle."""
    return f"<|user|>\n{instruction}\n<|assistant|>\n"


def extract_response(text):
    """Texte généré -> réponse : coupe au premier ``<|end|>``, retire les espaces autour."""
    return text.split(END_MARKER)[0].strip()


def ends_with(ids, end_ids):
    """Vrai si la séquence ``ids`` se termine par ``end_ids`` (critère d'arrêt)."""
    n = len(end_ids)
    return n > 0 and len(ids) >= n and list(ids[-n:]) == list(end_ids)


def pit_repo_for_year(year, size="4B", instruct=True):
    """Millésime à utiliser pour rejouer l'année ``year`` : coupure au 31/12/``year``-1."""
    vintage_year = year - 1
    if not FIRST_VINTAGE <= vintage_year <= LAST_VINTAGE:
        raise ValueError(f"pas de millésime PIT pour rejouer {year} "
                         f"(millésimes {FIRST_VINTAGE}-{LAST_VINTAGE})")
    vintage = VINTAGE_OVERRIDES.get(vintage_year, f"{vintage_year}12")
    ft = "-FT" if instruct else ""
    return f"Diamegs/PIT-{size}{ft}-{vintage}"


def leaked_terms(text, terms=POST_CUTOFF_TERMS):
    """Termes postérieurs à la coupure présents dans ``text`` (insensible à la casse)."""
    low = text.lower()
    return [t for t in terms if t in low]


# --------------------------------------------------------------------------- #
# Tests hors ligne
# --------------------------------------------------------------------------- #
def test_format_prompt():
    assert format_prompt("Hi?") == "<|user|>\nHi?\n<|assistant|>\n"


def test_extract_response_cuts_at_end_marker():
    assert extract_response("  Yes, it is.<|end|>garbage <|end|>") == "Yes, it is."
    assert extract_response("no marker ") == "no marker"


def test_ends_with():
    end = [27, 91, 437, 91, 29]          # "<|end|>" en BPE GPT-2 (config.json)
    assert ends_with([1, 2] + end, end)
    assert not ends_with([1, 2] + end + [3], end)
    assert not ends_with(end[1:], end)
    assert not ends_with([1, 2], [])


def test_pit_repo_for_year():
    assert pit_repo_for_year(2020) == PIT_REPO            # 2020 rejoué avec la coupure 2019-12
    assert pit_repo_for_year(2025) == "Diamegs/PIT-4B-FT-202412"
    assert pit_repo_for_year(2016) == "Diamegs/PIT-4B-FT-201511"
    assert pit_repo_for_year(2020, size="1B", instruct=False) == "Diamegs/PIT-1B-201912"
    with pytest.raises(ValueError):
        pit_repo_for_year(2026)
    with pytest.raises(ValueError):
        pit_repo_for_year(2013)


def test_leaked_terms():
    assert leaked_terms("A Lockdown was declared in March 2020") == ["lockdown", "march 2020"]
    assert leaked_terms("A new coronavirus, similar to SARS (2003).") == []


# --------------------------------------------------------------------------- #
# Inférence réelle
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pit():
    """Charge tokenizer + modèle une seule fois pour le module."""
    if os.environ.get("PIT_LLM_TESTS") != "1":
        pytest.skip("PIT_LLM_TESTS=1 requis (téléchargement ~17,8 Go)")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    tok = transformers.AutoTokenizer.from_pretrained(PIT_REPO, revision=PIT_REVISION, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        PIT_REPO, revision=PIT_REVISION, trust_remote_code=True, dtype=torch.bfloat16,
    ).to(device).eval()
    end_ids = tok.encode(END_MARKER, add_special_tokens=False)

    class StopOnEnd(transformers.StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return ends_with(input_ids[0].tolist(), end_ids)

    def generate(instruction, max_new_tokens=200):
        """Décodage glouton (déterministe) -> (réponse, nombre de tokens générés)."""
        enc = tok(format_prompt(instruction), return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id,
                stopping_criteria=transformers.StoppingCriteriaList([StopOnEnd()]),
            )
        new = out[0, enc["input_ids"].shape[1]:]
        return extract_response(tok.decode(new, skip_special_tokens=True)), len(new)

    def p_yes(instruction):
        """P(Yes) / (P(Yes) + P(No)) sur le premier token de la réponse.

        Score de risque continu et déterministe, sans dépendre de la capacité d'un
        modèle 4B à respecter un format de sortie.
        """
        enc = tok(format_prompt(instruction + " Answer Yes or No."), return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)

        def mass(words):
            ids = {tok.encode(w, add_special_tokens=False)[0] for w in words}
            return float(sum(probs[i] for i in ids))

        y, n = mass(["Yes", " Yes", "yes"]), mass(["No", " No", "no"])
        return y / (y + n), y + n

    return {"generate": generate, "p_yes": p_yes, "model": model}


@pytest.mark.pit
def test_pit_model_config(pit):
    cfg = pit["model"].config
    assert cfg.model_type == "pit"
    assert cfg.n_layer == 20 and cfg.n_embd == 4096


@pytest.mark.pit
def test_covid_pandemic_question(pit):
    """La question telle quelle : le modèle de 2019 ne connaît pas le nom COVID-19."""
    q = "Is there a risk of a global pandemic with COVID-19?"
    answer, n_tokens = pit["generate"](q)
    print(f"\n[PIT {CUTOFF}] Q: {q}\nA: {answer}\n({n_tokens} tokens)")
    assert answer, "réponse vide"
    assert not leaked_terms(answer), f"connaissance postérieure à {CUTOFF} : {leaked_terms(answer)}"


@pytest.mark.pit
def test_wuhan_pneumonia_question_point_in_time(pit):
    """La même question formulée avec l'information disponible le 31/12/2019
    (alerte de Wuhan à l'OMS) : c'est l'usage réaliste en rejeu."""
    q = ("On December 31, 2019, Chinese authorities reported a cluster of pneumonia cases "
         "of unknown cause in Wuhan, possibly a new coronavirus. Is there a risk that this "
         "becomes a global pandemic, and what would it mean for stock markets?")
    answer, n_tokens = pit["generate"](q, max_new_tokens=300)
    print(f"\n[PIT {CUTOFF}] Q: {q}\nA: {answer}\n({n_tokens} tokens)")
    assert answer, "réponse vide"
    leaks = leaked_terms(answer, POST_CUTOFF_TERMS + ("covid",))  # le prompt ne dit pas « COVID »
    assert not leaks, f"connaissance postérieure à {CUTOFF} : {leaks}"


@pytest.mark.pit
def test_pandemic_risk_score(pit):
    """Score oui/non continu : on vérifie qu'il est bien défini et on l'affiche.

    Aucune assertion sur sa valeur : c'est une mesure, pas une vérité attendue.
    La masse Yes+No doit toutefois être significative, sinon le modèle ne
    comprend pas la consigne et le score n'a pas de sens.
    """
    for label, q in {
        "covid-19": "Is there a risk of a global pandemic with COVID-19?",
        "wuhan": ("A cluster of pneumonia cases of unknown cause, possibly a new coronavirus, "
                  "was reported in Wuhan, China. Is there a risk of a global pandemic?"),
        "control": "Is there a risk of a global pandemic caused by the common cold this week?",
    }.items():
        p, m = pit["p_yes"](q)
        print(f"\n[PIT {CUTOFF}] P(yes)={p:.3f} (masse Yes+No={m:.2f}) — {label}")
        assert 0.0 <= p <= 1.0
        assert m > 0.05, f"le modèle ne répond pas oui/non ({label}, masse {m:.3f})"
