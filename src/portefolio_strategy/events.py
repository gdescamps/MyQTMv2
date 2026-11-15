"""News structurées par (titre, jour) avec Llama 3.1 8B Instruct (coupure des données : décembre 2023).

Pour chaque (titre, jour) de news (news.load_news : titres + début des articles, 6 au plus), un appel génère un
JSON à catégories fixes, sans texte libre :
- entreprise  : événements de l'entreprise étudiée (« Company X ») : type + sens (+ / −) ;
- concurrents : mêmes types, sens vu du concurrent ;
- macro       : droits de douane, inflation, taux directeurs, … : type + sens (hausse / baisse / annonce).

Anonymisation AVANT le LLM :
- l'entreprise étudiée (nom, nom court, ticker) → « Company X » ;
- dirigeants publics connus → leur fonction à la date de la news (ex. « the President of the United States ») ;
- autres noms de personnes (NER dslim/bert-base-NER) → « a person ».
Le LLM ne doit rapporter que des faits écrits dans le texte (pas d'appréciation d'importance ni d'impact).

Look-ahead : la coupure du LLM (2023-12) rend 2024 et après sans fuite ; sur 2020-2023 il connaît la suite,
d'où les faits seuls et l'anonymisation.

Sorties : data/portefolio_strategy/llm/llama31/events.parquet (day, symbol, json brut, ok) + timing.json ;
reprise automatique (les (titre, jour) déjà traités sont sautés).

    python -m src.portefolio_strategy.events --proto             # ~300 jours (gros trades connus + hasard)
    python -m src.portefolio_strategy.events                     # tout, de 2020 à la fin
"""
import argparse
import json
import re
import time

import pandas as pd

from . import config as C
from . import news as N

REPO = "meta-llama/Llama-3.1-8B-Instruct"
NER_REPO = "dslim/bert-base-NER"
OUT_DIR = C.DATA / "llm" / "llama31"
EVENTS = OUT_DIR / "events.parquet"
PROTO = OUT_DIR / "events_proto.parquet"
TIMING = OUT_DIR / "timing.json"
ART_CHARS = 600                # début de chaque article
MAX_ART = 6
MAX_NEW = 200
BATCH = 24

COMPANY_TYPES = ["earnings", "guidance", "analyst", "contract", "product", "mna", "legal_regulatory",
                 "management", "capital"]
MACRO_TYPES = ["tariffs_trade", "inflation", "interest_rates", "employment", "growth", "geopolitics",
               "sector_regulation", "fiscal_budget"]

SYSTEM = f"""You extract structured, factual events from financial news. The company under study is called "Company X".
Return ONLY a compact JSON object with three lists:
{{"company": [...], "competitors": [...], "macro": [...]}}
- "company": events about Company X itself. Each item: {{"type": T, "sign": "+" or "-"}} with T in {COMPANY_TYPES}.
  sign "+" = favourable fact for Company X (results above expectations, guidance raised, analyst upgrade or price
  target raised, new contract or partnership, product launch, acquisition completed, buyback), "-" = unfavourable
  fact (results below expectations, guidance cut, downgrade or target cut, lawsuit, fine, investigation, executive
  departure, share dilution).
- "competitors": the same kind of events about competitors or peers of Company X, sign from the competitor's point
  of view.
- "macro": economy or policy events mentioned in the text. Each item: {{"type": M, "direction": "up", "down" or
  "announced"}} with M in {MACRO_TYPES}. Example: new tariffs → {{"type": "tariffs_trade", "direction": "up"}};
  central bank rate cut → {{"type": "interest_rates", "direction": "down"}}.
Rules: report only facts explicitly written in the text; no opinion, no importance rating, no prediction; do not
use outside knowledge; never output names of people or companies; use empty lists when nothing applies; at most
6 items per list; one item per distinct event."""

# Fonctions publiques : nom → [(début, fin, fonction)] (fin exclue) ; défaut hors périodes = « a former official ».
OFFICIALS = {
    "Donald Trump": [("2017-01-20", "2021-01-20", "the President of the United States"),
                     ("2025-01-20", "2029-01-20", "the President of the United States")],
    "Joe Biden": [("2021-01-20", "2025-01-20", "the President of the United States")],
    "Jerome Powell": [("2018-02-05", "2026-05-15", "the Chair of the US Federal Reserve")],
    "Janet Yellen": [("2021-01-26", "2025-01-20", "the US Treasury Secretary")],
    "Scott Bessent": [("2025-01-28", "2029-01-20", "the US Treasury Secretary")],
    "Christine Lagarde": [("2019-11-01", "2027-10-31", "the President of the European Central Bank")],
    "Xi Jinping": [("2013-03-14", "2033-03-14", "the President of China")],
    "Kamala Harris": [("2021-01-20", "2025-01-20", "the Vice President of the United States")],
    "JD Vance": [("2025-01-20", "2029-01-20", "the Vice President of the United States")],
    "Howard Lutnick": [("2025-02-21", "2029-01-20", "the US Commerce Secretary")],
    "Gina Raimondo": [("2021-03-03", "2025-01-20", "the US Commerce Secretary")],
}
FORMER = {"Donald Trump": "the former President of the United States",
          "Joe Biden": "the former President of the United States"}
# Dirigeants d'entreprise très cités : fonction générique (sans nom d'entreprise, pour ne rien désanonymiser).
EXECUTIVES = ["Elon Musk", "Jensen Huang", "Tim Cook", "Mark Zuckerberg", "Sam Altman", "Satya Nadella",
              "Sundar Pichai", "Lisa Su", "Warren Buffett", "Jeff Bezos", "Andy Jassy", "Cathie Wood", "Lip-Bu Tan",
              "Pat Gelsinger", "Hock Tan", "Sanjay Mehrotra", "Cristiano Amon", "Jamie Dimon", "Michael Dell",
              "Larry Ellison", "Safra Catz", "Bill Gates", "Reed Hastings", "Alex Karp", "Ken Xie"]
SURNAMES = {n.split()[-1]: n for n in OFFICIALS}       # « Trump », « Powell », …


def official_role(name, day):
    if name in EXECUTIVES:
        return "a business executive"
    for a, b, role in OFFICIALS[name]:
        if a <= day < b:
            return role
    return FORMER.get(name, "a former official")


class Anonymizer:
    def __init__(self, device=0):
        from transformers import pipeline
        self.ner = pipeline("ner", model=NER_REPO, aggregation_strategy="simple", device=device)
        names = sorted(list(OFFICIALS) + list(SURNAMES) + EXECUTIVES, key=len, reverse=True)
        self.rx_off = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")(?:'s)?\b")

    @staticmethod
    def company(text, symbol, name):
        pats = {re.escape(symbol)}
        base = re.sub(r"[,.]|\b(Inc|Corp|Corporation|Incorporated|Ltd|Plc|Holdings|Company|Co|Class [A-C])\b", " ",
                      name or "")
        base = " ".join(base.split())
        if base:
            pats.add(re.escape(base))
            if len(base.split()[0]) > 3:
                pats.add(re.escape(base.split()[0]))
        rx = re.compile(r"\b(" + "|".join(sorted(pats, key=len, reverse=True)) + r")(?:'s)?\b", re.I)
        return rx.sub("Company X", text)

    def officials(self, text, day):
        return self.rx_off.sub(lambda m: official_role(SURNAMES.get(m.group(1), m.group(1)), day), text)

    def persons(self, texts):
        """Remplace les noms de personnes restants (NER, fragments fusionnés, étendus au mot entier)."""
        out = []
        for t, ents in zip(texts, self.ner(texts, batch_size=32)):
            spans = []
            for e in ents:
                if e["entity_group"] != "PER":
                    continue
                s, f = e["start"], e["end"]
                while s > 0 and t[s - 1].isalnum():
                    s -= 1
                while f < len(t) and (t[f].isalnum() or t[f] in "-'"):
                    f += 1
                if spans and s <= spans[-1][1] + 1:
                    spans[-1][1] = max(spans[-1][1], f)
                else:
                    spans.append([s, f])
            for s, f in reversed(spans):
                t = t[:s] + "a person" + t[f:]
            out.append(t)
        return out


def day_text(g):
    parts = []
    for r in g.sort_values("ts").head(MAX_ART).itertuples():
        body = " ".join(str(r.content).split())[:ART_CHARS]
        parts.append(f"- {r.title}. {body}")
    return "\n".join(parts)


def parse(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    ok = lambda xs, key, vals, k2, v2: [x for x in xs if isinstance(x, dict) and x.get(key) in vals  # noqa: E731
                                        and x.get(k2) in v2] if isinstance(xs, list) else []
    return {"company": ok(d.get("company"), "type", COMPANY_TYPES, "sign", ("+", "-")),
            "competitors": ok(d.get("competitors"), "type", COMPANY_TYPES, "sign", ("+", "-")),
            "macro": ok(d.get("macro"), "type", MACRO_TYPES, "direction", ("up", "down", "announced"))}


class Llama:
    def __init__(self):
        import torch
        import transformers
        self.torch = torch
        self.tok = transformers.AutoTokenizer.from_pretrained(REPO)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = transformers.AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16).to("cuda").eval()

    def prompt(self, text):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"News:\n{text}\n\nJSON:"}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, texts):
        enc = self.tok([self.prompt(t) for t in texts], return_tensors="pt", padding=True,
                       add_special_tokens=False).to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None,
                                      pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return ([self.tok.decode(g, skip_special_tokens=True) for g in gen], int(enc["attention_mask"].sum()),
                int((gen != self.tok.pad_token_id).sum()))


def todo_keys(proto):
    news = N.load_news()
    news["day"] = pd.to_datetime(news["day"])
    news = news[news["day"] >= C.START_DATE]
    keys = news[["symbol", "day"]].drop_duplicates()
    if proto:
        known = [("INTC", "2026-04-07", "2026-04-16"), ("AMD", "2026-04-14", "2026-04-22"),
                 ("NVDA", "2025-05-13", "2025-05-19"), ("MU", "2026-04-18", "2026-04-27"),
                 ("FTNT", "2026-05-06", "2026-05-11"), ("TEAM", "2024-10-08", "2024-10-11")]
        k = pd.concat([keys[(keys["symbol"] == s) & keys["day"].between(a, b)] for s, a, b in known])
        rnd = keys[keys["day"] >= "2024-01-01"].sample(240, random_state=0)
        keys = pd.concat([k, rnd]).drop_duplicates()
    return news, keys.sort_values(["day", "symbol"]).reset_index(drop=True)


def run(proto=False, chunk=480):
    from .sentiment import company_names
    path = PROTO if proto else EVENTS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = pd.read_parquet(path) if path.exists() and not proto else pd.DataFrame(columns=["day", "symbol"])
    news, keys = todo_keys(proto)
    if len(done):
        seen = set(zip(done["symbol"], pd.to_datetime(done["day"])))
        keys = keys[[(s, d) not in seen for s, d in zip(keys["symbol"], keys["day"])]]
    print(f"{len(keys)} (titre, jour) à traiter", flush=True)
    names = company_names()
    grp = news.set_index(["symbol", "day"]).sort_index()
    anon, llm = Anonymizer(), Llama()
    t0, n_in, n_out, rows = time.time(), 0, 0, []
    for i in range(0, len(keys), chunk):
        part = keys.iloc[i:i + chunk]
        texts = []
        for s, d in zip(part["symbol"], part["day"]):
            t = Anonymizer.company(day_text(grp.loc[(s, d)].reset_index()), s, names.get(s, s))
            texts.append(anon.officials(t, f"{d:%Y-%m-%d}"))
        texts = anon.persons(texts)
        order = sorted(range(len(texts)), key=lambda j: len(texts[j]))
        raws = [None] * len(texts)
        for j in range(0, len(order), BATCH):
            idx = order[j:j + BATCH]
            out, a, b = llm.generate([texts[k] for k in idx])
            n_in, n_out = n_in + a, n_out + b
            for k, o in zip(idx, out):
                raws[k] = o
        for (s, d), t, r in zip(zip(part["symbol"], part["day"]), texts, raws):
            p = parse(r)
            rows.append(dict(day=d, symbol=s, text=t if proto else None, raw=r, ok=p is not None,
                             events=json.dumps(p) if p else None))
        new = pd.DataFrame(rows)
        pd.concat([done, new], ignore_index=True).to_parquet(path, index=False)
        dt = time.time() - t0
        n = i + len(part)
        print(f"  {n}/{len(keys)}  {dt:.0f} s  {n / dt:.2f} jours/s  entrée {n_in / dt:.0f} tok/s  "
              f"sortie {n_out / dt:.0f} tok/s  JSON valides {new['ok'].mean():.1%}  "
              f"reste ≈ {(len(keys) - n) / (n / dt) / 3600:.1f} h", flush=True)
    TIMING.write_text(json.dumps(dict(days=len(rows), seconds=time.time() - t0, tokens_in=n_in, tokens_out=n_out)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", action="store_true")
    run(proto=ap.parse_args().proto)


if __name__ == "__main__":
    main()
