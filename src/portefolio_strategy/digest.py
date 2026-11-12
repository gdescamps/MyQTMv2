"""Condensé factuel des news d'un (titre, jour) par Claude Haiku (API Anthropic) — étape sans jugement.

Rôle de Haiku : seulement réduire plusieurs articles longs à trois courtes rubriques factuelles
(COMPANY / COMPETITORS / GLOBAL, les trois catégories de ref/MyQTM) qui tiennent dans le contexte de
2 048 tokens du LLM point-in-time. Haiku ne note rien : la note de sentiment est donnée ensuite par
Diamegs/PIT-4B-FT-201912 (sentiment.py), dont les connaissances s'arrêtent en 2019-12.

Risque résiduel de fuite : Haiku connaît la suite des événements et pourrait colorer son résumé. La
consigne l'interdit (faits de l'article seulement, aucun jugement, aucune connaissance extérieure), mais
ce n'est pas vérifiable article par article.

Reprenable : un enregistrement par (titre, jour) dans data/portefolio_strategy/llm/haiku/digests.jsonl.

    python -m src.portefolio_strategy.digest --limit 20     # smoke test (quelques centimes)
    python -m src.portefolio_strategy.digest --full         # tout le corpus : voir estimate.py AVANT
"""
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import config as C
from . import news as N

DIGESTS = C.HAIKU_DIR / "digests.jsonl"
SECTIONS = ("COMPANY", "COMPETITORS", "GLOBAL")

SYSTEM = """You condense financial news for a downstream classifier. You receive news articles published \
on a single day that mention one listed company. Using ONLY what the articles state, write exactly three \
sections:

COMPANY: facts about the company itself - products, sales, earnings, guidance, deals, management, lawsuits \
(at most 60 words).
COMPETITORS: facts about its competitors or industry peers (at most 40 words).
GLOBAL: facts about global factors - politics, taxes, regulation, wars, diseases, elections, the economy \
(at most 40 words).

Rules:
- Report stated facts and figures only, in neutral wording.
- Add no knowledge that is not in the articles. Never mention what happened later.
- Do not judge whether the news is good or bad, and do not predict anything.
- Leave out share price moves unless the article gives their cause.
- The word limits are hard limits: keep only the most important facts, never list every article.
- Do not cite article numbers or sources.
- If the articles say nothing for a section, write exactly: none
- Output the three sections and nothing else."""


def user_prompt(name, ticker, articles):
    parts = [f"Company: {name} ({ticker})", ""]
    for i, (title, content) in enumerate(articles, 1):
        parts.append(f"--- Article {i} ---\n{title}\n\n{content}")
    return "\n".join(parts)


def parse_digest(text):
    """→ {company, competitors, global} ; « none » si la rubrique manque."""
    out = {}
    for s in SECTIONS:
        m = re.search(rf"{s}\s*:\s*(.*?)(?=\n\s*(?:{'|'.join(SECTIONS)})\s*:|\Z)", text or "", re.S | re.I)
        v = " ".join(m.group(1).split()) if m else ""
        out[s.lower()] = v if v and v.lower().strip(".") != "none" else "none"
    return out


def make_client():
    """Client de l'API Anthropic (clé ANTHROPIC_API du .env, jamais affichée)."""
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(C.ROOT / ".env")
    key = os.environ.get("ANTHROPIC_API")
    if not key:
        raise RuntimeError("ANTHROPIC_API absent de l'environnement (.env)")
    return anthropic.Anthropic(api_key=key, max_retries=5)


def call_haiku(client, model, name, ticker, articles):
    r = client.messages.create(model=model, max_tokens=C.HAIKU_MAX_TOKENS, system=SYSTEM,
                               thinking={"type": "disabled"}, output_config={"effort": "low"},
                               messages=[{"role": "user", "content": user_prompt(name, ticker, articles)}])
    text = "".join(b.text for b in r.content if b.type == "text")
    return text, r.usage.input_tokens, r.usage.output_tokens, r.stop_reason


def load_digests():
    if not DIGESTS.exists():
        return pd.DataFrame(columns=["day", "symbol", "company", "competitors", "global", "in_tok", "out_tok"])
    d = pd.read_json(DIGESTS, lines=True, convert_dates=["day"])
    return d.drop_duplicates(["day", "symbol"], keep="last")


def pending(news, done):
    keys = news[["day", "symbol"]].drop_duplicates()
    if len(done):
        keys = keys.merge(done[["day", "symbol"]].assign(_d=1), how="left", on=["day", "symbol"])
        keys = keys[keys["_d"].isna()].drop(columns="_d")
    return keys


def cost(in_tok, out_tok):
    return (in_tok * C.HAIKU_PRICE_IN + out_tok * C.HAIKU_PRICE_OUT) / 1e6


def run(limit=None, seed=0, workers=C.WORKERS):
    from . import universe as U
    news = N.load_news()
    names = U.load_universe().drop_duplicates("symbol").set_index("symbol")["name"].to_dict()
    todo = pending(news, load_digests())
    if limit:
        todo = todo.sample(min(limit, len(todo)), random_state=seed)
    groups = {k: g for k, g in news.merge(todo, on=["day", "symbol"]).groupby(["day", "symbol"])}
    client, model = make_client(), C.HAIKU_MODEL
    C.HAIKU_DIR.mkdir(parents=True, exist_ok=True)
    lock, tot, t0 = threading.Lock(), [0, 0, 0], time.time()

    def one(key):
        day, sym = key
        g = groups[key]
        arts = list(zip(g["title"], g["content"]))
        try:
            text, i, o, stop = call_haiku(client, model, names.get(sym, sym), sym.replace("_old", ""), arts)
        except Exception as e:                      # noqa: BLE001 — on journalise et on continue (reprenable)
            print(f"  échec {sym} {day:%Y-%m-%d} : {type(e).__name__} {str(e)[:120]}", flush=True)
            return
        rec = {"day": f"{day:%Y-%m-%d}", "symbol": sym, **parse_digest(text), "n_articles": len(arts),
               "in_tok": i, "out_tok": o, "stop": stop}
        with lock:
            with open(DIGESTS, "a") as f:
                f.write(json.dumps(rec) + "\n")
            tot[0] += i
            tot[1] += o
            tot[2] += 1
            if tot[2] % 500 == 0:
                print(f"  {tot[2]}/{len(groups)}  {cost(tot[0], tot[1]):.2f} $  {time.time() - t0:.0f} s", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, list(groups)))
    n = max(tot[2], 1)
    if tot[2]:
        (C.HAIKU_DIR / "timing.json").write_text(json.dumps(
            {"calls": tot[2], "seconds": round(time.time() - t0, 1), "workers": workers}))
    print(f"{tot[2]} condensés ; entrée {tot[0]} tokens ({tot[0] / n:.0f}/appel), sortie {tot[1]} "
          f"({tot[1] / n:.0f}/appel) ; coût {cost(tot[0], tot[1]):.4f} $ ; {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--full", action="store_true", help="traiter tout le corpus (payant : estimer d'abord)")
    ap.add_argument("--workers", type=int, default=C.WORKERS, help="appels simultanés (réseau, pas des cœurs)")
    a = ap.parse_args()
    if not a.limit and not a.full:
        ap.error("préciser --limit N (smoke test) ou --full (corpus complet, après lecture des estimations)")
    run(limit=a.limit, workers=a.workers)


if __name__ == "__main__":
    main()
