"""Phase « generate » des news structurées (voir events.py) — à lancer avec le venv vLLM :

    venv_vllm/bin/python src/portefolio_strategy/events_vllm.py [--proto]

Lit data/portefolio_strategy/llm/llama31/texts[_proto].parquet, génère avec Llama 3.1 8B Instruct (vLLM,
glouton) une ligne par événement « bloc|type|sens|citation exacte », écrit raw[_proto].parquet (day, symbol,
raw). Reprise : les (titre, jour) déjà présents dans raw.parquet sont sautés ; sauvegarde par paquets.
Autonome (pandas + vllm seulement) : n'importe pas le paquet src.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "portefolio_strategy" / "llm" / "llama31"
REPO = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW = 160
CHUNK = 4000

SYSTEM = """You extract factual events from financial news about one company, called "Company X".
Most news items contain NO event: general market commentary, stock-picking articles, price moves, opinions,
predictions and articles mainly about other companies are NOT events for Company X.

Output one line per event, exactly in this format:
block|type|sign|exact quote
- block: company (an event of Company X itself), competitors (an event of a competitor or peer of Company X),
  or macro (economy or policy).
- type for company and competitors (use exactly one of these words):
  earnings (reported results: revenue, EPS, margins, billings vs expectations or vs last year),
  guidance (company outlook or forecast raised / cut), analyst (rating or price target change by an analyst firm),
  contract (contract, order, deal or partnership with a customer or partner), product (launch of a new product),
  mna (merger, acquisition, divestment), legal_regulatory (lawsuit, fine, investigation, export ban, approval),
  management (executive appointment or departure), capital (buyback, dividend, share or debt issuance).
- sign for company and competitors: + if the fact is favourable to that company (results above expectations,
  guidance raised, analyst upgrade or price target raised, new contract or partnership won, acquisition completed,
  buyback), - if unfavourable (results below expectations, guidance cut, downgrade or target cut, lawsuit, fine,
  investigation, executive departure, share dilution).
- type for macro: tariffs_trade, inflation, interest_rates, employment, growth, geopolitics, sector_regulation,
  fiscal_budget; sign for macro: up, down or announced.
- exact quote: 4 to 15 consecutive words copied exactly from the news text (not from these instructions) that
  state the fact.
Rules: only facts explicitly stated as having happened; no opinion, no prediction, no outside knowledge; never
write names of people; at most 8 lines; if there is no event, output exactly: NONE"""

EXAMPLES = [
    ("- Company X stock jumps as Morgan Stanley upgrades it to Top Pick. Morgan Stanley upgraded the shares and "
     "raised its price target to $250. Rival Acme cut its full-year outlook.",
     "company|analyst|+|Morgan Stanley upgraded the shares and raised its price target\n"
     "competitors|guidance|-|Rival Acme cut its full-year outlook"),
    ("- 3 AI Stocks to Buy Now. These companies could soar over the next decade as AI spending grows, including "
     "Company X and Nvidia.", "NONE"),
    ("- Company X shares fall after the government announced new tariffs on chip imports. Company X reported "
     "quarterly revenue below analysts' estimates.",
     "macro|tariffs_trade|up|the government announced new tariffs on chip imports\n"
     "company|earnings|-|reported quarterly revenue below analysts' estimates"),
]


def messages(text):
    m = [{"role": "system", "content": SYSTEM}]
    for t, a in EXAMPLES:
        m += [{"role": "user", "content": f"News:\n{t}"}, {"role": "assistant", "content": a}]
    return m + [{"role": "user", "content": f"News:\n{text}"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", action="store_true")
    proto = ap.parse_args().proto
    sfx = "_proto" if proto else ""
    texts = pd.read_parquet(OUT_DIR / f"texts{sfx}.parquet")
    raw_path = OUT_DIR / f"raw{sfx}.parquet"
    done = pd.read_parquet(raw_path) if raw_path.exists() and not proto else pd.DataFrame(columns=["day", "symbol", "raw"])
    if len(done):
        seen = set(zip(done["symbol"], pd.to_datetime(done["day"])))
        texts = texts[[(s, d) not in seen for s, d in zip(texts["symbol"], pd.to_datetime(texts["day"]))]]
    print(f"{len(texts)} (titre, jour) à générer", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=REPO, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.6,
              enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, stop=["\n\n\n"])
    t0, n_in, n_out, parts = time.time(), 0, 0, [done]
    for i in range(0, len(texts), CHUNK):
        part = texts.iloc[i:i + CHUNK]
        outs = llm.chat([messages(t) for t in part["text"]], sp, use_tqdm=False)
        n_in += sum(len(o.prompt_token_ids) for o in outs)
        n_out += sum(len(o.outputs[0].token_ids) for o in outs)
        parts.append(pd.DataFrame({"day": part["day"].values, "symbol": part["symbol"].values,
                                   "raw": [o.outputs[0].text for o in outs]}))
        pd.concat(parts, ignore_index=True).to_parquet(raw_path, index=False)
        dt, n = time.time() - t0, i + len(part)
        print(f"  {n}/{len(texts)}  {dt:.0f} s  {n / dt:.2f} jours/s  entrée {n_in / dt:.0f} tok/s  sortie "
              f"{n_out / dt:.0f} tok/s  reste ≈ {(len(texts) - n) / (n / dt) / 3600:.1f} h", flush=True)
    (OUT_DIR / f"timing_vllm{sfx}.json").write_text(json.dumps(dict(days=len(texts), seconds=time.time() - t0,
                                                                    tokens_in=n_in, tokens_out=n_out)))


if __name__ == "__main__":
    main()
