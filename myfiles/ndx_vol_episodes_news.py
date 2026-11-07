"""Explique chaque épisode de volatilité NDX (myfiles/ndx_vol_episodes.csv) par les news.

Pour chaque épisode, un appel Claude Haiku 5.5 (Bedrock, client de tests/test_llm.py) reçoit :
- CNBC (data/text/cnbc/cnbc_close.parquet) : article complet de la séance de début + résumé
  de chaque séance suivante jusqu'à la fin de l'épisode (séance = date US/Eastern) ;
- DeltaOne (data/text/x/deltaone/tweets.parquet, hors retweets) : tweets de [début − 12 h, fin],
  numérotés [k] avec horodatage UTC.
Il renvoie (JSON contraint) le résumé CNBC de la séance, le catalyseur, les indices des tweets
déclencheurs et des tweets de prolongation ; les indices sont remappés sur texte/heure/URL.

Reprise : les épisodes déjà présents dans le .jsonl sont sautés.
    python myfiles/ndx_vol_episodes_news.py --limit 5     # test
    python myfiles/ndx_vol_episodes_news.py               # tout
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from test_llm import make_client, model_id, text_of  # noqa: E402

EP = "myfiles/ndx_vol_episodes.csv"
CNBC = "data/text/cnbc/cnbc_close.parquet"
TWEETS = "data/text/x/deltaone/tweets.parquet"
OUT_JSONL = "myfiles/ndx_vol_episodes_news.jsonl"
OUT_CSV = "myfiles/ndx_vol_episodes_news.csv"
PRE_H = 12
PRICE_IN, PRICE_OUT = 0.10, 0.50  # $/M tokens, Haiku 5.5 (API Anthropic)

SCHEMA = {
    "type": "object",
    "properties": {
        "resume_cnbc": {"type": "string"},
        "catalyseur": {"type": "string"},
        "explication": {"type": "string"},
        "declencheurs": {"type": "array", "items": {"type": "integer"}},
        "prolongations": {"type": "array", "items": {"type": "integer"}},
        "confiance": {"type": "string", "enum": ["haute", "moyenne", "basse"]},
    },
    "required": ["resume_cnbc", "catalyseur", "explication", "declencheurs", "prolongations", "confiance"],
    "additionalProperties": False,
}

SYSTEM = """Tu es analyste marchés actions US. On te donne un épisode de forte volatilité du \
Nasdaq-100 (CFD horaire, heures UTC), les comptes rendus de clôture CNBC des séances concernées \
et les tweets de l'agrégateur de news DeltaOne, numérotés [k] avec leur heure UTC.

Réponds en français, en JSON :
- resume_cnbc : 2-4 phrases, d'après CNBC, expliquant pourquoi le marché a monté/baissé pendant \
la ou les séances de l'épisode (catalyseurs précis, chiffres clés).
- catalyseur : thème court (ex. « tarifs douaniers », « Fed / taux », « inflation CPI », \
« résultats Nvidia », « géopolitique Moyen-Orient », « emploi NFP », « technique / aucun »).
- explication : 1-3 phrases liant les news à la chronologie de l'épisode (début → prolongation).
- declencheurs : indices [k] (1 à 3) des tweets qui ont DÉCLENCHÉ le mouvement — publiés avant \
le début ou au plus tard 3 h après, cohérents avec son sens. Pour un rebond après une chute, le \
déclencheur est la news d'apaisement / de retournement, pas la mauvaise news précédente. Liste \
vide si aucun.
- Ignore toute news sans lien avec les marchés ou la macro (sport, faits divers, politique non \
économique).
- prolongations : indices [k] (0 à 6) des tweets publiés PENDANT l'épisode qui l'ont prolongé \
dans le même sens. Ne garde que des news macro/marché réellement importantes, pas de bruit.
- confiance : haute / moyenne / basse selon la clarté du lien news → mouvement.
N'invente rien : n'utilise que les indices présents."""


def load():
    ep = pd.read_csv(EP, parse_dates=["start", "end"])
    ep["episode_id"] = ep["side"] + "_" + ep["start"].dt.strftime("%Y%m%d%H")
    c = pd.read_parquet(CNBC).set_index("date").sort_index()
    t = pd.read_parquet(TWEETS)
    t = t[~t["is_retweet"]].sort_values("created_at").reset_index(drop=True)
    return ep, c, t


def et_date(ts):
    return ts.tz_convert("America/New_York").normalize().tz_localize(None)


def build(r, c, t):
    d0, d1 = et_date(r.start), et_date(r.end)
    sess = c.index[(c.index >= d0) & (c.index <= d1)]
    if len(sess) == 0 or sess[0] != d0:
        nxt = c.index[c.index >= d0][:1]  # séance de début absente → la suivante
        sess = sess.union(nxt)
    parts = []
    for i, s in enumerate(sess):
        row = c.loc[s]
        txt = row.body if i == 0 else row.summary
        parts.append(f"### CNBC clôture {s.date()} — {row.title}\n{txt}")
    tw = t[(t.created_at >= r.start - pd.Timedelta(hours=PRE_H)) & (t.created_at <= r.end)]
    lines = [f"[{k}] {ts:%Y-%m-%d %H:%M} {txt.replace(chr(10), ' ')}"
             for k, (ts, txt) in enumerate(zip(tw.created_at, tw.text))]
    sens = "BAISSIER" if r.side == "down" else "HAUSSIER"
    prompt = (f"Épisode {sens} : {r.amplitude_pct:+.2f} % du {r.start:%Y-%m-%d %H:%M} UTC "
              f"({r.start_price:.0f}) au {r.end:%Y-%m-%d %H:%M} UTC ({r.end_price:.0f}), "
              f"durée {r.duration_h:.0f} h.\n\n" + "\n\n".join(parts) +
              f"\n\n### Tweets DeltaOne (début − {PRE_H} h → fin)\n" + ("\n".join(lines) or "(aucun)"))
    return prompt, tw, list(sess)


def pick(tw, idx):
    out = []
    for k in idx:
        if 0 <= k < len(tw):
            x = tw.iloc[k]
            out.append({"time": f"{x.created_at:%Y-%m-%d %H:%M}", "text": x.text, "url": x.url})
    return out


def analyse(client, model, r, c, t):
    prompt, tw, sess = build(r, c, t)
    msg = client.messages.create(
        model=model, max_tokens=8000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM, messages=[{"role": "user", "content": prompt}],
    )
    if msg.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason={msg.stop_reason}")
    d = json.loads(text_of(msg))
    return {
        "episode_id": r.episode_id, "side": r.side, "start": str(r.start), "end": str(r.end),
        "amplitude_pct": round(r.amplitude_pct, 2), "duration_h": r.duration_h,
        "cnbc_sessions": [str(s.date()) for s in sess],
        "cnbc_urls": [c.loc[s].url for s in sess],
        "resume_cnbc": d["resume_cnbc"], "catalyseur": d["catalyseur"],
        "explication": d["explication"], "confiance": d["confiance"],
        "news_declencheur": pick(tw, d["declencheurs"]),
        "news_prolongation": pick(tw, d["prolongations"]),
        "n_tweets": len(tw),
        "usage": {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens},
    }


def to_csv():
    rows = [json.loads(x) for x in open(OUT_JSONL)]
    df = pd.DataFrame(rows)
    fmt = lambda L: " || ".join(f"{n['time']} {n['text']}" for n in L)  # noqa: E731
    df["news_declencheur"] = df["news_declencheur"].map(fmt)
    df["news_prolongation"] = df["news_prolongation"].map(fmt)
    df["cnbc_urls"] = df["cnbc_urls"].map(" ".join)
    df["cnbc_sessions"] = df["cnbc_sessions"].map(" ".join)
    df["abs_amp"] = df["amplitude_pct"].abs()
    df = df.drop(columns=["usage"]).sort_values(["side", "abs_amp"], ascending=[True, False])
    df.drop(columns=["abs_amp"]).to_csv(OUT_CSV, index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ep, c, t = load()
    done = set()
    if os.path.exists(OUT_JSONL):
        done = {json.loads(x)["episode_id"] for x in open(OUT_JSONL)}
    todo = ep[~ep.episode_id.isin(done)]
    # ordre : alternance des plus fortes baisses / hausses
    todo = todo.assign(rk=todo.groupby("side")["amplitude_pct"].transform(lambda s: s.abs().rank(ascending=False)))
    todo = todo.sort_values(["rk", "side"])
    if a.limit:
        todo = todo.head(a.limit)
    print(f"{len(done)} déjà faits, {len(todo)} à faire")
    client, model = make_client(timeout=300.0, max_retries=5), model_id("claude-haiku-5-5")
    lock, tin, tout, err = threading.Lock(), 0, 0, 0
    with ThreadPoolExecutor(a.workers) as ex, open(OUT_JSONL, "a") as f:
        futs = {ex.submit(analyse, client, model, r, c, t): r.episode_id for r in todo.itertuples()}
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                res = fu.result()
            except Exception as e:  # noqa: BLE001
                err += 1
                print(f"  ERREUR {futs[fu]}: {e}")
                continue
            with lock:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
                tin += res["usage"]["in"]
                tout += res["usage"]["out"]
            if n % 20 == 0 or n == len(futs):
                cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
                print(f"  {n}/{len(futs)}  in={tin} out={tout}  ≈ {cost:.3f} $  erreurs={err}")
    cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
    print(f"Run : in={tin} out={tout} tokens ≈ {cost:.3f} $ (tarif API), erreurs={err}")
    rows = to_csv()
    print(f"{len(rows)} épisodes → {OUT_JSONL} / {OUT_CSV}")


if __name__ == "__main__":
    main()
