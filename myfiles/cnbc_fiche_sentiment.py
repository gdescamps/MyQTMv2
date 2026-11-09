"""Note de sentiment « fondamental » de chaque fiche de contexte CNBC (Haiku 5.5).

Seuls les faits économiques TANGIBLES comptent (données macro publiées, résultats/guidances
chiffrés, décisions de politique effectivement prises, événements de crédit, accords signés).
L'humeur, les craintes, les anticipations, les opinions d'analystes et les mouvements de cours
eux-mêmes sont notés neutres. Le champ « humeur » de la fiche n'est pas transmis au modèle.

Entrée : data/text/cnbc/context/days/*.json (cf. cnbc_market_context.py)
Sortie : data/text/cnbc/sentiment/days/YYYY-MM-DD.json + data/text/cnbc/sentiment/sentiment.parquet
Non chaîné → parallèle. Reprise : les jours déjà notés sont sautés.
    python myfiles/cnbc_fiche_sentiment.py --limit 5     # test
    python myfiles/cnbc_fiche_sentiment.py               # tout
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from test_llm import make_client, model_id, text_of  # noqa: E402

IN_DIR = "data/text/cnbc/context/days"
OUT_DIR = "data/text/cnbc/sentiment/days"
OUT_PQ = "data/text/cnbc/sentiment/sentiment.parquet"
PRICE_IN, PRICE_OUT = 0.10, 0.50  # $/M tokens, Haiku 5.5

SCHEMA = {
    "type": "object",
    "properties": {
        "faits": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "fait": {"type": "string"},
                "tangible": {"type": "boolean"},
                "sens": {"type": "integer", "enum": [-2, -1, 0, 1, 2]},
            },
            "required": ["fait", "tangible", "sens"], "additionalProperties": False}},
        "score": {"type": "integer", "enum": list(range(-5, 6))},
        "justification": {"type": "string"},
    },
    "required": ["faits", "score", "justification"],
    "additionalProperties": False,
}

SYSTEM = """Tu es économiste. On te donne la fiche de contexte de marché d'une séance (Nasdaq / \
tech + macro US). Note l'état des FONDAMENTAUX ÉCONOMIQUES décrit par la fiche, du point de vue \
d'un investisseur en actions Nasdaq, en ne retenant que les éléments TANGIBLES.

Tangible (à noter) = progrès ou dégradation économique concrète et constatée :
- données macro publiées (emploi, inflation, PIB, ISM/PMI, ventes au détail, chômage…) vs attentes \
ou vs tendance ;
- résultats et guidances d'entreprises chiffrés (bénéfices, chiffre d'affaires, capex, commandes) ;
- décisions de politique effectivement prises (hausse/baisse de taux, QT, tarifs douaniers \
imposés ou levés, lois votées, sanctions) ;
- communications OFFICIELLES de la banque centrale, assimilées à des décisions : communiqué et \
compte rendu (minutes) du FOMC, projections (dot plot), discours ou témoignage du président de \
la Fed. Un signal de resserrement (hausses plus rapides/nombreuses, réduction du bilan) est une \
dégradation, un signal d'assouplissement un progrès. Note-les de la même façon chaque jour où la \
fiche les cite comme moteur ;
- événements de crédit ou de liquidité avérés (faillites, défauts, sauvetages bancaires), \
accords signés, chocs d'offre réels (prix du pétrole, ruptures d'approvisionnement).

Non tangible (sens = 0, tangible = false) : humeur ou psychologie du marché, craintes, espoirs, \
spéculations, rumeurs, anticipations sur des événements non encore survenus (y compris les paris \
du marché sur la Fed, hors communication officielle), opinions et notes d'analystes, rendements \
obligataires et autres prix de marché, valorisations jugées chères, rotations sectorielles, et les mouvements de cours \
eux-mêmes (une baisse des indices n'est pas une dégradation économique).

Consignes :
- faits : liste les 3 à 8 éléments principaux de la fiche ; pour chacun, tangible oui/non et sens \
de -2 (dégradation nette) à +2 (progrès net), 0 si non tangible ou ambigu.
- score : bilan net des seuls faits tangibles, de -5 (dégradation forte et généralisée) à +5 \
(progrès fort et généralisé) ; 0 si rien de tangible ou si équilibré.
- justification : 1 phrase."""


def out_path(d):
    return os.path.join(OUT_DIR, f"{d}.json")


def build_prompt(rec):
    f = dict(rec["fiche"])
    f.pop("humeur", None)
    return f"Séance : {rec['date']}\n\n## Fiche\n{json.dumps(f, ensure_ascii=False, indent=1)}"


def call(client, model, prompt):
    for attempt in range(5):
        try:
            msg = client.messages.create(
                model=model, max_tokens=4000,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                system=SYSTEM, messages=[{"role": "user", "content": prompt}],
            )
            if msg.stop_reason != "end_turn":
                raise RuntimeError(f"stop_reason={msg.stop_reason}")
            return json.loads(text_of(msg)), msg.usage
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            print(f"    retry {attempt + 1}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))


def one(client, model, path):
    rec = json.load(open(path))
    res, u = call(client, model, build_prompt(rec))
    out = {"date": rec["date"], "sentiment": res, "usage": {"in": u.input_tokens, "out": u.output_tokens}}
    with open(out_path(rec["date"]), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return u.input_tokens, u.output_tokens


def aggregate():
    rows = []
    for f in sorted(os.listdir(OUT_DIR)):
        d = json.load(open(os.path.join(OUT_DIR, f)))
        s = d["sentiment"]
        tang = [x for x in s["faits"] if x["tangible"]]
        rows.append({"date": pd.Timestamp(d["date"]), "score": s["score"],
                     "n_faits": len(s["faits"]), "n_tangibles": len(tang),
                     "somme_tangibles": sum(x["sens"] for x in tang),
                     "justification": s["justification"],
                     "faits": json.dumps(s["faits"], ensure_ascii=False)})
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PQ, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    todo = [os.path.join(IN_DIR, f) for f in sorted(os.listdir(IN_DIR))
            if f[:10] >= a.start and not os.path.exists(out_path(f[:10]))]
    if a.limit:
        todo = todo[:a.limit]
    client, model = make_client(timeout=300.0, max_retries=3), model_id("claude-haiku-5-5")
    tin = tout = n = 0
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(one, client, model, p) for p in todo]):
            i, o = fut.result()
            tin, tout, n = tin + i, tout + o, n + 1
            if n % 50 == 0 or n == len(todo):
                cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
                print(f"  {n}/{len(todo)}  in={tin} out={tout}  ≈ {cost:.3f} $  {time.time() - t0:.0f} s", flush=True)
    df = aggregate()
    print(f"{len(df)} fiches notées → {OUT_PQ}")


if __name__ == "__main__":
    main()
