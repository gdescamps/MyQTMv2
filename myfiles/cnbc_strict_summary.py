"""Résumés STRICTS (anglais) des articles CNBC de clôture par Haiku 5.5, pour alimenter le LLM PIT.

But : réduire le contexte d'entrée de PIT-4B (fenêtre 2048 tokens) sans y injecter la connaissance
de Haiku. Consigne : ne restituer QUE ce que dit l'article (chiffres, faits, citations), aucune
connaissance extérieure, aucune interprétation ni conséquence non écrite.
Deux longueurs : short (≤ 40 mots, pour les séances précédentes) et long (≤ 150 mots, séance du jour).

Sortie : data/text/cnbc/strict_summary/days/YYYY-MM-DD.json + strict_summary.parquet
    python myfiles/cnbc_strict_summary.py --limit 3
    python myfiles/cnbc_strict_summary.py --start 2021-12-01
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

CNBC = "data/text/cnbc/cnbc_close.parquet"
OUT_DIR = "data/text/cnbc/strict_summary/days"
OUT_PQ = "data/text/cnbc/strict_summary/strict_summary.parquet"
PRICE_IN, PRICE_OUT = 0.10, 0.50

SCHEMA = {
    "type": "object",
    "properties": {"short": {"type": "string"}, "long": {"type": "string"}},
    "required": ["short", "long"], "additionalProperties": False,
}

SYSTEM = """You compress a US stock market close article into two strictly factual English summaries.

STRICT RULES:
- Use ONLY information written in the article. Add NO outside knowledge, no background, no context \
you know from elsewhere, no hindsight, no consequences or interpretations that the article does not state.
- Keep the article's own numbers (index levels and % moves, data releases vs expectations, earnings \
figures, rates) and attribute opinions to their source ("analyst X said...").
- Do not mention the date or weekday unless needed for a scheduled event the article cites.
- Telegraphic style allowed, no filler.

short: at most 40 words — the index moves and the 1-2 main drivers.
long: at most 150 words — index moves, every economic data release, earnings, policy decision or \
official statement mentioned, the main stock movers with their reason, and upcoming events the \
article says the market is waiting for."""


def out_path(d):
    return os.path.join(OUT_DIR, f"{d:%Y-%m-%d}.json")


def call(client, model, text):
    for attempt in range(5):
        try:
            msg = client.messages.create(
                model=model, max_tokens=2000,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                system=SYSTEM, messages=[{"role": "user", "content": text}],
            )
            if msg.stop_reason != "end_turn":
                raise RuntimeError(f"stop_reason={msg.stop_reason}")
            return json.loads(text_of(msg)), msg.usage
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            print(f"    retry {attempt + 1}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))


def one(client, model, d, row):
    res, u = call(client, model, f"# {row.title}\n\n{row.body}")
    with open(out_path(d), "w") as f:
        json.dump({"date": f"{d:%Y-%m-%d}", "url": row.url, **res,
                   "usage": {"in": u.input_tokens, "out": u.output_tokens}}, f, ensure_ascii=False, indent=1)
    return u.input_tokens, u.output_tokens


def aggregate():
    rows = [json.load(open(os.path.join(OUT_DIR, f))) for f in sorted(os.listdir(OUT_DIR))]
    df = pd.DataFrame([{k: r[k] for k in ("date", "url", "short", "long")} for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(OUT_PQ, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-12-01")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    c = pd.read_parquet(CNBC).set_index("date").sort_index()
    c = c[c.index >= a.start]
    todo = [(d, r) for d, r in c.iterrows() if not os.path.exists(out_path(d))]
    if a.limit:
        todo = todo[:a.limit]
    client, model = make_client(timeout=300.0, max_retries=3), model_id("claude-haiku-5-5")
    tin = tout = n = 0
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(one, client, model, d, r) for d, r in todo]):
            i, o = fut.result()
            tin, tout, n = tin + i, tout + o, n + 1
            if n % 100 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  in={tin} out={tout}  ≈ {tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT:.3f} $"
                      f"  {time.time() - t0:.0f} s", flush=True)
    df = aggregate()
    print(f"{len(df)} résumés → {OUT_PQ}")


if __name__ == "__main__":
    main()
