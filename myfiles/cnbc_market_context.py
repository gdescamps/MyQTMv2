"""Fiche de contexte de marché quotidienne, chaînée, à partir des clôtures CNBC.

Pour chaque séance J (data/text/cnbc/cnbc_close.parquet), Claude Haiku 5.5 (Bedrock, client de
tests/test_llm.py) reçoit :
- la fiche de J−1 (mémoire longue : thèmes suivis depuis des semaines) ;
- l'article de clôture complet de J + les résumés des WINDOW−1 séances précédentes ;
et produit une fiche JSON : régime, thèmes moteurs, attentes (catalyseurs à venir), tensions,
résumé. Séquentiel (chaînage). Une fiche par jour : data/text/cnbc/context/days/YYYY-MM-DD.json,
+ agrégat data/text/cnbc/context/context.parquet.

ATTENTION look-ahead : les pages CNBC sont modifiées après la clôture → la fiche J n'est
utilisable qu'à partir de J+1 dans un backtest.

Reprise : les jours déjà présents sont relus (servent de J−1) et sautés.
    python myfiles/cnbc_market_context.py --start 2026-10-23 --end 2026-11-06   # test
    python myfiles/cnbc_market_context.py --start 2023-01-03                    # 2023 → fin
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from test_llm import make_client, model_id, text_of  # noqa: E402

CNBC = "data/text/cnbc/cnbc_close.parquet"
OUT_DIR = "data/text/cnbc/context/days"
OUT_PQ = "data/text/cnbc/context/context.parquet"
WINDOW = 10
PRICE_IN, PRICE_OUT = 0.10, 0.50  # $/M tokens, Haiku 5.5 (API Anthropic)

SCHEMA = {
    "type": "object",
    "properties": {
        "humeur": {"type": "string", "enum": ["risk-on", "risk-off", "rotation", "attentisme", "panique", "euphorie"]},
        "tendance": {"type": "string"},
        "themes": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "theme": {"type": "string"},
                "intensite": {"type": "integer"},
                "evolution": {"type": "string", "enum": ["nouveau", "monte", "stable", "baisse", "eteint"]},
                "detail": {"type": "string"},
            },
            "required": ["theme", "intensite", "evolution", "detail"], "additionalProperties": False}},
        "attentes": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "evenement": {"type": "string"},
                "echeance": {"type": "string"},
                "attente": {"type": "string"},
            },
            "required": ["evenement", "echeance", "attente"], "additionalProperties": False}},
        "tensions": {"type": "array", "items": {
            "type": "object",
            "properties": {"sujet": {"type": "string"}, "justification": {"type": "string"}},
            "required": ["sujet", "justification"], "additionalProperties": False}},
        "resume": {"type": "string"},
    },
    "required": ["humeur", "tendance", "themes", "attentes", "tensions", "resume"],
    "additionalProperties": False,
}

SYSTEM = """Tu es stratégiste actions US. Tu tiens à jour, séance après séance, une fiche de \
CONTEXTE DE MARCHÉ (focus Nasdaq / tech, mais macro incluse). On te donne la fiche de la séance \
précédente (ta mémoire) et les comptes rendus de clôture CNBC des dernières séances, le dernier \
étant la séance du jour (texte complet).

Mets la fiche à jour pour la clôture du jour, en français :
- humeur : régime dominant du marché.
- tendance : 1 phrase sur la tendance récente des indices (niveaux, % depuis les plus hauts, série).
- themes : 3 à 6 sujets qui font bouger le marché en ce moment (ex. « rentabilité du capex IA », \
« Fed / taux », « tarifs », « résultats big tech »). intensite 1-5 ; evolution vs la fiche \
précédente (nouveau / monte / stable / baisse / eteint — garde un thème « eteint » une seule fois \
puis retire-le) ; detail = 1 phrase factuelle.
- attentes : catalyseurs à venir que le marché attend (résultats, FOMC, CPI, NFP, échéances \
politiques) avec l'échéance (date ou « cette semaine ») et ce qui est anticipé / pricé. Retire les \
événements passés.
- tensions : 1 à 4 points sur lesquels le marché est fébrile (ex. « investissements IA jugés trop \
lourds, le marché exige des revenus ») avec la justification tirée des textes.
- resume : 3-4 phrases de contexte de marché à la clôture du jour.
Conserve la continuité avec la fiche précédente mais n'invente rien : appuie-toi sur les textes."""


def day_path(d):
    return os.path.join(OUT_DIR, f"{d:%Y-%m-%d}.json")


def build_prompt(c, i, prev):
    win = c.iloc[max(0, i - WINDOW + 1):i + 1]
    parts = []
    for j, (d, row) in enumerate(win.iterrows()):
        last = j == len(win) - 1
        txt = row.body if last else row.summary
        tag = "SÉANCE DU JOUR (texte complet)" if last else "résumé"
        parts.append(f"### {d:%Y-%m-%d} ({d:%a}) — {tag} — {row.title}\n{txt}")
    prev_txt = json.dumps(prev["fiche"], ensure_ascii=False) if prev else "(aucune : première fiche)"
    d = c.index[i]
    return (f"Séance du jour : {d:%Y-%m-%d} ({d:%A}).\n\n## Fiche précédente"
            f"{' (' + prev['date'] + ')' if prev else ''}\n{prev_txt}\n\n## Clôtures CNBC\n" + "\n\n".join(parts))


def call(client, model, prompt):
    for attempt in range(5):
        try:
            msg = client.messages.create(
                model=model, max_tokens=8000,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                system=SYSTEM, messages=[{"role": "user", "content": prompt}],
            )
            if msg.stop_reason != "end_turn":
                raise RuntimeError(f"stop_reason={msg.stop_reason}")
            return json.loads(text_of(msg)), msg.usage
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            print(f"    retry {attempt + 1}: {e}")
            time.sleep(10 * (attempt + 1))


def aggregate():
    rows = []
    for f in sorted(os.listdir(OUT_DIR)):
        d = json.load(open(os.path.join(OUT_DIR, f)))
        fi = d["fiche"]
        rows.append({
            "date": pd.Timestamp(d["date"]), "cnbc_url": d["cnbc_url"], "humeur": fi["humeur"],
            "tendance": fi["tendance"], "resume": fi["resume"],
            "themes": json.dumps(fi["themes"], ensure_ascii=False),
            "attentes": json.dumps(fi["attentes"], ensure_ascii=False),
            "tensions": json.dumps(fi["tensions"], ensure_ascii=False),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PQ, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-03")
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    c = pd.read_parquet(CNBC).set_index("date").sort_index()
    idx = [i for i, d in enumerate(c.index) if d >= pd.Timestamp(a.start) and (a.end is None or d <= pd.Timestamp(a.end))]
    client, model = make_client(timeout=300.0, max_retries=3), model_id("claude-haiku-5-5")
    prev, tin, tout, n = None, 0, 0, 0
    # fiche J−1 si elle existe déjà (reprise / prolongement)
    if idx and idx[0] > 0 and os.path.exists(day_path(c.index[idx[0] - 1])):
        prev = json.load(open(day_path(c.index[idx[0] - 1])))
    t0 = time.time()
    for i in idx:
        d = c.index[i]
        if os.path.exists(day_path(d)):
            prev = json.load(open(day_path(d)))
            continue
        fiche, u = call(client, model, build_prompt(c, i, prev))
        rec = {"date": f"{d:%Y-%m-%d}", "cnbc_url": c.iloc[i].url, "cnbc_title": c.iloc[i].title,
               "window": [f"{x:%Y-%m-%d}" for x in c.index[max(0, i - WINDOW + 1):i + 1]],
               "fiche": fiche, "usage": {"in": u.input_tokens, "out": u.output_tokens}}
        with open(day_path(d), "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        prev = rec
        tin, tout, n = tin + u.input_tokens, tout + u.output_tokens, n + 1
        if n % 10 == 0 or i == idx[-1]:
            cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
            print(f"  {d:%Y-%m-%d}  {n} fiches  in={tin} out={tout}  ≈ {cost:.3f} $  "
                  f"{(time.time() - t0) / n:.1f} s/fiche", flush=True)
    cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
    print(f"Run : {n} fiches, in={tin} out={tout} ≈ {cost:.3f} $ (tarif API)")
    df = aggregate()
    print(f"{len(df)} fiches → {OUT_DIR}/ + {OUT_PQ}")


if __name__ == "__main__":
    main()
