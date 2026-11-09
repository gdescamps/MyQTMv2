"""Classement de chaque tweet DeltaOne par Claude Haiku 5.5, par lot journalier, avec contexte.

Pour chaque jour UTC D, Haiku (Bedrock, client de tests/test_llm.py) reçoit :
- la dernière fiche de contexte de marché STRICTEMENT antérieure à D
  (data/text/cnbc/context/days/, cf. myfiles/cnbc_market_context.py) → pas de look-ahead ;
- tous les tweets DeltaOne du jour (hors retweets), numérotés [k] avec heure UTC.
Pour chaque tweet il renvoie : catégorie, sens attendu pour le Nasdaq (-1/0/+1), `mover`
(news susceptible de faire bouger significativement le Nasdaq — cible ≈ 3 % des tweets) et
`confiance` 0-100 (probabilité que la news fasse réellement bouger le marché dans ce sens).

Sortie : un fichier par jour data/text/x/deltaone/classified/YYYY-MM-DD.json (tous les tweets,
classement conservé) + agrégat data/text/x/deltaone/classified.parquet.

BIAIS : coupure des connaissances de Haiku 5.5 = juin 2026 → avant juillet 2026 le modèle peut
« savoir » quelles news ont fait bouger le marché. Seul 2026-07 → est hors échantillon.

    python myfiles/deltaone_classify.py --start 2026-10-20 --end 2026-11-07   # test
    python myfiles/deltaone_classify.py                                      # tout (reprise)
    python myfiles/deltaone_classify.py --model claude-sonnet-5-5 --start 2026-07-01   # → classified_sonnet/
"""
import argparse
import glob
import html
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from test_llm import make_client, model_id, text_of  # noqa: E402

TWEETS = "data/text/x/deltaone/tweets.parquet"
CTX_DIR = "data/text/cnbc/context/days"
OUT_DIR = "data/text/x/deltaone/classified"
OUT_PQ = "data/text/x/deltaone/classified.parquet"
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0)}  # $/M tokens (API Anthropic)
PRICE_IN, PRICE_OUT = PRICES["claude-haiku-5-5"]

CATS = ["fed_taux", "inflation", "emploi", "croissance_macro", "tarifs_commerce", "geopolitique",
        "resultats_entreprises", "ia_semis", "energie_petrole", "politique_us", "credit_banques",
        "crypto", "reaction_marche", "autre", "hors_marche"]

SCHEMA = {
    "type": "object",
    "properties": {"tweets": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "k": {"type": "integer"},
            "categorie": {"type": "string", "enum": CATS},
            "sens": {"type": "integer", "enum": [-1, 0, 1]},
            "mover": {"type": "boolean"},
            "confiance": {"type": "integer"},
        },
        "required": ["k", "categorie", "sens", "mover", "confiance"],
        "additionalProperties": False}}},
    "required": ["tweets"],
    "additionalProperties": False,
}

SYSTEM = """Tu es trader actions US (Nasdaq-100). On te donne la fiche de contexte de marché de la \
dernière clôture connue (humeur, thèmes, attentes, tensions) puis TOUS les tweets de news de \
l'agrégateur DeltaOne d'une journée, numérotés [k] avec l'heure UTC.

Classe CHAQUE tweet (un objet par [k], dans l'ordre, aucun oublié) :
- categorie : thème de la news ; « reaction_marche » = simple constat de prix/flux (« NASDAQ \
FALLS 2% », « VIX JUMPS »), « hors_marche » = sport, faits divers, politique sans enjeu économique.
- sens : effet attendu sur le Nasdaq-100 au moment de la publication (+1 hausse, -1 baisse, 0 neutre).
- mover : true seulement si la news est NOUVELLE et assez importante pour faire bouger le \
Nasdaq-100 de façon significative (≈ 1 % ou plus) dans les heures qui suivent. Tiens compte du \
contexte : une news qui touche une tension ou une attente de la fiche pèse plus ; une news déjà \
anticipée, une redite d'un tweet précédent, un commentaire ou une simple réaction de prix \
(reaction_marche) n'est PAS un mover. Ordre de grandeur attendu : ≈ 3 % des tweets, soit 0 à 3 \
par jour en temps normal, davantage seulement lors de journées exceptionnelles.
- confiance : 0-100, probabilité que cette news fasse réellement bouger le marché dans le sens \
indiqué (élevée seulement pour les movers évidents ; faible pour le bruit).
Juge uniquement d'après le contexte et la news, comme en temps réel : ne t'appuie pas sur ce que \
tu pourrais savoir de la suite des événements."""


def load_ctx():
    ctx = {}
    for f in sorted(glob.glob(os.path.join(CTX_DIR, "*.json"))):
        d = json.load(open(f))
        ctx[pd.Timestamp(d["date"])] = d
    return pd.Series(ctx).sort_index()


def ctx_before(ctx, day):
    prev = ctx[ctx.index < day]
    return prev.iloc[-1] if len(prev) else None


def fiche_txt(c):
    f = c["fiche"]
    th = "\n".join(f"- {t['theme']} ({t['intensite']}/5, {t['evolution']}) : {t['detail']}" for t in f["themes"])
    at = "\n".join(f"- {a['evenement']} [{a['echeance']}] : {a['attente']}" for a in f["attentes"])
    te = "\n".join(f"- {t['sujet']} : {t['justification']}" for t in f["tensions"])
    return (f"Clôture du {c['date']} — humeur {f['humeur']}\nTendance : {f['tendance']}\n"
            f"Thèmes :\n{th}\nAttentes :\n{at or '-'}\nTensions :\n{te or '-'}\nRésumé : {f['resume']}")


CHUNK = 25  # taille des paquets de repli quand un lot complet revient tronqué


def _call(client, model, day, ctx_txt, lines, ks, n_total, usage):
    """Un appel Haiku sur les tweets d'indices ks (numérotation [k] du jour conservée)."""
    head = (f"## Contexte (dernière clôture connue)\n{ctx_txt}\n\n"
            f"## Tweets DeltaOne du {day:%Y-%m-%d} ({day:%A}) — {n_total} tweets"
            + (f" (paquet de {len(ks)} : classe exactement ces {len(ks)} indices)" if len(ks) < n_total else ""))
    prompt = head + "\n" + "\n".join(lines[k] for k in ks)
    msg = client.messages.create(
        model=model, max_tokens=32000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM, messages=[{"role": "user", "content": prompt}],
    )
    usage[0] += msg.usage.input_tokens
    usage[1] += msg.usage.output_tokens
    if msg.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason={msg.stop_reason}")
    return {o["k"]: o for o in json.loads(text_of(msg))["tweets"] if o["k"] in set(ks)}


def classify_day(client, model, day, tw, c):
    lines = [f"[{k}] {ts:%H:%M} {html.unescape(x).replace(chr(10), ' ')}"
             for k, (ts, x) in enumerate(zip(tw.created_at, tw.text))]
    ctx_txt = fiche_txt(c) if c is not None else "(aucun)"
    n, usage, allk = len(tw), [0, 0], list(range(len(tw)))
    out = {}
    for attempt in range(2):  # lot complet, une relance
        out = _call(client, model, day, ctx_txt, lines, allk, n, usage)
        if n - len(out) <= 0.05 * n:
            break
        print(f"    {day:%Y-%m-%d}: lot incomplet ({len(out)}/{n}), relance {attempt + 1}", flush=True)
    else:  # repli : paquets de CHUNK tweets sur les indices manquants
        miss = [k for k in allk if k not in out]
        for i in range(0, len(miss), CHUNK):
            ks = miss[i:i + CHUNK]
            for _ in range(2):
                got = _call(client, model, day, ctx_txt, lines, ks, n, usage)
                out.update(got)
                if len(ks) - len(got) <= 0.05 * len(ks):
                    break
        if n - len(out) > 0.05 * n:
            raise RuntimeError(f"lot incomplet après repli par paquets ({len(out)}/{n})")
        print(f"    {day:%Y-%m-%d}: complété par paquets ({len(out)}/{n})", flush=True)
    rows = []
    for k, x in enumerate(tw.itertuples()):
        o = out.get(k)
        rows.append({
            "id": int(x.id), "created_at": x.created_at.isoformat(), "url": x.url,
            "text": html.unescape(x.text),
            "categorie": o["categorie"] if o else None, "sens": o["sens"] if o else None,
            "mover": o["mover"] if o else None, "confiance": o["confiance"] if o else None,
        })
    return {"date": f"{day:%Y-%m-%d}", "context_date": c["date"] if c is not None else None,
            "n_tweets": len(tw), "n_missing": sum(r["categorie"] is None for r in rows),
            "tweets": rows, "usage": {"in": usage[0], "out": usage[1]}}


def aggregate():
    rows = []
    if not glob.glob(os.path.join(OUT_DIR, "*.json")):
        return pd.DataFrame(columns=["mover"])
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "*.json"))):
        d = json.load(open(f))
        for r in d["tweets"]:
            rows.append({**r, "day": d["date"], "context_date": d["context_date"]})
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df.to_parquet(OUT_PQ, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="claude-haiku-5-5", choices=sorted(PRICES))
    a = ap.parse_args()
    global OUT_DIR, OUT_PQ, PRICE_IN, PRICE_OUT
    PRICE_IN, PRICE_OUT = PRICES[a.model]
    if a.model != "claude-haiku-5-5":  # sorties séparées, le classement Haiku n'est jamais écrasé
        tag = a.model.split("-")[1]
        OUT_DIR, OUT_PQ = f"{OUT_DIR}_{tag}", f"{OUT_DIR}_{tag}.parquet"
    os.makedirs(OUT_DIR, exist_ok=True)
    t = pd.read_parquet(TWEETS)
    t = t[~t["is_retweet"]].sort_values("created_at")
    t["day"] = t["created_at"].dt.tz_convert(None).dt.normalize()
    t = t[t.day >= pd.Timestamp(a.start)]
    if a.end:
        t = t[t.day <= pd.Timestamp(a.end)]
    ctx = load_ctx()
    days = [d for d in t.day.unique() if not os.path.exists(os.path.join(OUT_DIR, f"{d:%Y-%m-%d}.json"))]
    print(f"{len(days)} jours à classer ({t[t.day.isin(days)].shape[0]} tweets)")
    client, model = make_client(timeout=600.0, max_retries=5), model_id(a.model)
    lock, tin, tout, err = threading.Lock(), 0, 0, 0
    groups = {d: g for d, g in t.groupby("day")}
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(classify_day, client, model, d, groups[d], ctx_before(ctx, d)): d for d in days}
        for n, fu in enumerate(as_completed(futs), 1):
            d = futs[fu]
            try:
                res = fu.result()
            except Exception as e:  # noqa: BLE001
                err += 1
                print(f"  ERREUR {d:%Y-%m-%d}: {e}", flush=True)
                continue
            with open(os.path.join(OUT_DIR, f"{d:%Y-%m-%d}.json"), "w") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
            with lock:
                tin += res["usage"]["in"]
                tout += res["usage"]["out"]
            if n % 25 == 0 or n == len(futs):
                cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
                print(f"  {n}/{len(futs)}  in={tin} out={tout}  ≈ {cost:.3f} $  erreurs={err}", flush=True)
    cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
    print(f"Run : in={tin} out={tout} ≈ {cost:.3f} $ (tarif API), erreurs={err}")
    df = aggregate()
    print(f"{len(df)} tweets classés → {OUT_DIR}/ + {OUT_PQ} ; movers {df.mover.mean() * 100:.1f} %")


if __name__ == "__main__":
    main()
