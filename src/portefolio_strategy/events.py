"""News structurées par (titre, jour) avec Llama 3.1 8B Instruct (coupure des données : décembre 2023).

Trois phases (la génération tourne dans un venv séparé avec vLLM, venv_vllm/) :
1. prepare (venv du projet) : pour chaque (titre, jour) de news, titres + début des articles (6 au plus),
   anonymisés → llm/llama31/texts[_proto].parquet ;
   - l'entreprise étudiée (nom, nom court, ticker) → « Company X » ;
   - dirigeants publics connus → leur fonction à la date de la news (« the President of the United States ») ;
   - grands dirigeants d'entreprise → « a business executive » ; autres personnes (NER) → « a person ».
2. generate (venv_vllm : venv_vllm/bin/python src/portefolio_strategy/events_vllm.py) : une ligne par
   événement « bloc|type|sens|citation exacte » → llm/llama31/raw[_proto].parquet.
3. parse (venv du projet) : lignes valides (catégories fixes) ET citation retrouvée mot pour mot dans le texte
   (sinon l'événement est rejeté : garde-fou contre les événements inventés) → llm/llama31/events[_proto].parquet,
   une ligne par événement (day, symbol, block, type, sign).

Blocs : company (Company X), competitors (sens vu du concurrent), macro (sens = up / down / announced).
Le LLM ne rapporte que des faits écrits ; aucun nom ne sort (catégories fixes ; la citation sert seulement à la
vérification et n'est pas utilisée comme feature).
Look-ahead : coupure du LLM 2023-12 → 2024 et après sans fuite ; 2020-2023 : faits seuls + anonymisation.

    python -m src.portefolio_strategy.events prepare [--proto]
    venv_vllm/bin/python src/portefolio_strategy/events_vllm.py [--proto]
    python -m src.portefolio_strategy.events parse [--proto]
"""
import argparse
import re
import unicodedata

import pandas as pd

from . import config as C
from . import news as N

REPO = "meta-llama/Llama-3.1-8B-Instruct"
NER_REPO = "dslim/bert-base-NER"
OUT_DIR = C.DATA / "llm" / "llama31"
ART_CHARS = 600
MAX_ART = 6

COMPANY_TYPES = ("earnings", "guidance", "analyst", "contract", "product", "mna", "legal_regulatory",
                 "management", "capital")
MACRO_TYPES = ("tariffs_trade", "inflation", "interest_rates", "employment", "growth", "geopolitics",
               "sector_regulation", "fiscal_budget")
KNOWN = [("INTC", "2026-04-07", "2026-04-16"), ("AMD", "2026-04-14", "2026-04-22"), ("NVDA", "2025-05-13", "2025-05-19"),
         ("MU", "2026-04-18", "2026-04-27"), ("FTNT", "2026-05-06", "2026-05-11"), ("TEAM", "2024-10-08", "2024-10-11")]


def path(kind, proto):
    return OUT_DIR / f"{kind}{'_proto' if proto else ''}.parquet"


# --------------------------------------------------------------------------- anonymisation
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
EXECUTIVES = ["Elon Musk", "Jensen Huang", "Tim Cook", "Mark Zuckerberg", "Sam Altman", "Satya Nadella",
              "Sundar Pichai", "Lisa Su", "Warren Buffett", "Jeff Bezos", "Andy Jassy", "Cathie Wood", "Lip-Bu Tan",
              "Pat Gelsinger", "Hock Tan", "Sanjay Mehrotra", "Cristiano Amon", "Jamie Dimon", "Michael Dell",
              "Larry Ellison", "Safra Catz", "Bill Gates", "Reed Hastings", "Alex Karp", "Ken Xie"]
SURNAMES = {n.split()[-1]: n for n in OFFICIALS}


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
        out = []
        for t, ents in zip(texts, self.ner(texts, batch_size=64)):
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


def prepare(proto=False, chunk=2000):
    from .sentiment import company_names
    news = N.load_news()
    news["day"] = pd.to_datetime(news["day"])
    news = news[news["day"] >= C.START_DATE]
    keys = news[["symbol", "day"]].drop_duplicates()
    if proto:
        k = pd.concat([keys[(keys["symbol"] == s) & keys["day"].between(a, b)] for s, a, b in KNOWN])
        keys = pd.concat([k, keys[keys["day"] >= "2024-01-01"].sample(240, random_state=0)]).drop_duplicates()
    keys = keys.sort_values(["day", "symbol"]).reset_index(drop=True)
    names = company_names()
    grp = news.set_index(["symbol", "day"]).sort_index()
    anon, out = Anonymizer(), []
    for i in range(0, len(keys), chunk):
        part = keys.iloc[i:i + chunk]
        texts = [anon.officials(Anonymizer.company(day_text(grp.loc[(s, d)].reset_index()), s, names.get(s, s)),
                                f"{d:%Y-%m-%d}") for s, d in zip(part["symbol"], part["day"])]
        out.append(part.assign(text=anon.persons(texts)))
        print(f"  {i + len(part)}/{len(keys)} textes anonymisés", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(out, ignore_index=True).to_parquet(path("texts", proto), index=False)


# --------------------------------------------------------------------------- vérification des sorties
def _norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[‘’“”\"'`]", "", s)
    return " ".join(re.sub(r"[^\w%$.,+-]", " ", s).split())


TYPE_ALIASES = {"revenue": "earnings", "billings": "earnings", "eps": "earnings", "results": "earnings",
                "outlook": "guidance", "partnership": "contract", "deal": "contract", "order": "contract",
                "buyback": "capital", "dividend": "capital", "acquisition": "mna", "merger": "mna", "legal": "legal_regulatory",
                "regulatory": "legal_regulatory", "rating": "analyst", "upgrade": "analyst", "downgrade": "analyst"}


def _found(q, tn):
    """Citation retrouvée dans le texte : mot pour mot, ou ≥ 85 % de ses mots (≥ 4) dans une fenêtre du texte
    de même longueur + 50 % (tolère une ponctuation ou un mot changés, pas une paraphrase)."""
    qw = q.split()
    if len(qw) < 3:
        return False
    if q in tn:
        return True
    if len(qw) < 4:
        return False
    tw, need = tn.split(), set(qw)
    span = int(len(qw) * 1.5) + 1
    for i in range(0, max(len(tw) - len(qw) + 1, 1)):
        win = set(tw[i:i + span])
        if len(need & win) >= 0.85 * len(need):
            return True
    return False


def parse_lines(raw, text):
    """→ [(block, type, sign)] dont la citation (≥ 3 mots) figure mot pour mot dans le texte ; doublons fusionnés."""
    tn, out = _norm(text), []
    for line in (raw or "").splitlines():
        p = [x.strip() for x in line.strip().strip("`").split("|")]
        if len(p) < 4:
            continue
        block, typ, sign, quote = p[0].lower(), p[1].lower(), p[2].lower(), "|".join(p[3:])
        if block != "macro":
            typ = TYPE_ALIASES.get(typ, typ)
        if block in ("company", "competitors"):
            if typ not in COMPANY_TYPES or sign not in ("+", "-"):
                continue
        elif block == "macro":
            if typ not in MACRO_TYPES or sign not in ("up", "down", "announced"):
                continue
        else:
            continue
        if not _found(_norm(quote), tn):
            continue
        if (block, typ, sign) not in out:
            out.append((block, typ, sign))
    return out


def parse(proto=False):
    t = pd.read_parquet(path("texts", proto))
    r = pd.read_parquet(path("raw", proto))
    m = t.merge(r, on=["symbol", "day"])
    rows, stats = [], []
    for s in m.itertuples():
        ev = parse_lines(s.raw, s.text)
        n_lines = sum(1 for x in (s.raw or "").splitlines() if x.count("|") >= 3)
        stats.append((n_lines, len(ev)))
        rows += [(s.day, s.symbol, b, ty, sg) for b, ty, sg in ev]
    ev = pd.DataFrame(rows, columns=["day", "symbol", "block", "type", "sign"])
    ev.to_parquet(path("events", proto), index=False)
    st = pd.DataFrame(stats, columns=["lignes", "gardees"])
    print(f"{len(m)} (titre, jour) ; lignes produites {st['lignes'].sum()}, événements gardés {len(ev)} "
          f"({len(ev) / max(st['lignes'].sum(), 1):.0%}) ; jours avec ≥ 1 événement {(st['gardees'] > 0).mean():.0%}")
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "parse"])
    ap.add_argument("--proto", action="store_true")
    a = ap.parse_args()
    (prepare if a.phase == "prepare" else parse)(proto=a.proto)


if __name__ == "__main__":
    main()
