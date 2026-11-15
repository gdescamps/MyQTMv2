"""Étape 4 : prototype d'extraction d'événements structurés avec PIT-4B (coupure 2019-12, pas de look-ahead).

Entrée : titres + début des articles bruts du (titre, jour) (news.load_news), nom de l'entreprise et ticker
remplacés par « Company X ». Pour chaque type d'événement, question fermée → P(Yes) lue sur les logits du
premier token (comme sentiment.py), sans génération. Question témoin absurde pour mesurer le biais « Yes ».
Échantillon : jours de news avant les gros trades connus + jours tirés au hasard sur 2024-2026.
Sortie : myfiles/pit_events_proto.parquet + vitesse mesurée.

    python myfiles/pit_events_proto.py [--n-random=150]
"""
import re
import sys
import time

import pandas as pd

sys.path.insert(0, ".")
from src.portefolio_strategy import news as N  # noqa: E402
from src.portefolio_strategy import sentiment as S  # noqa: E402

EVENTS = {
    "contrat": "a new major contract, order or partnership for Company X",
    "relevement_analyste": "an analyst upgrading Company X or raising its price target",
    "degradation_analyste": "an analyst downgrading Company X or cutting its price target",
    "prevision_relevee": "Company X raising its financial guidance or outlook",
    "prevision_abaissee": "Company X lowering its financial guidance or outlook",
    "resultats_au_dessus": "Company X reporting quarterly results above expectations",
    "resultats_en_dessous": "Company X reporting quarterly results below expectations",
    "fusion_acquisition": "a merger or acquisition involving Company X",
    "probleme_juridique": "a regulatory, legal or lawsuit problem for Company X",
    "nouveau_produit": "Company X launching a new product",
    "temoin_absurde": "Company X opening a restaurant on the Moon",
}
QUESTION = "Question: Does this news report {event}? Answer with exactly one word, Yes or No."
KNOWN = [("INTC", "2026-04-07", "2026-04-16"), ("AMD", "2026-04-14", "2026-04-22"), ("NVDA", "2025-05-13", "2025-05-19"),
         ("MU", "2026-04-18", "2026-04-27"), ("TEAM", "2024-10-08", "2024-10-11"), ("FTNT", "2026-05-06", "2026-05-08"),
         ("AVGO", "2025-05-13", "2025-05-14"), ("QCOM", "2026-04-21", "2026-04-27")]
ART_CHARS = 260
MAX_ART = 6


def anonymize(text, symbol, name):
    pats = {re.escape(symbol)}
    base = re.sub(r"[,.]|\b(Inc|Corp|Corporation|Incorporated|Ltd|Plc|Holdings|Company|Co|Class [A-C])\b", " ", name)
    base = " ".join(base.split())
    if base:
        pats.add(re.escape(base))
        first = base.split()[0]
        if len(first) > 3:
            pats.add(re.escape(first))
    rx = re.compile(r"\b(" + "|".join(sorted(pats, key=len, reverse=True)) + r")(?:'s)?\b", re.I)
    return rx.sub("Company X", text)


def day_text(g, symbol, name):
    parts = []
    for r in g.sort_values("ts").head(MAX_ART).itertuples():
        body = " ".join(str(r.content).split())[:ART_CHARS]
        parts.append(f"- {r.title}. {body}")
    return anonymize("\n".join(parts), symbol, name)


def main():
    n_random = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--n-random=")), 150))
    news = N.load_news([2024, 2025, 2026])
    news["day"] = pd.to_datetime(news["day"])
    names = S.company_names()
    keys = set()
    for sym, a, b in KNOWN:
        d = news[(news["symbol"] == sym) & news["day"].between(a, b)]
        keys |= set(zip(d["symbol"], d["day"]))
    allk = news[["symbol", "day"]].drop_duplicates()
    rnd = allk.sample(n_random, random_state=0)
    keys |= set(zip(rnd["symbol"], rnd["day"]))
    grp = news.set_index(["symbol", "day"]).sort_index()
    pit = S.Pit()
    rows, prompts, meta = [], [], []
    for sym, day in sorted(keys):
        g = grp.loc[(sym, day)].reset_index()
        text = day_text(g, sym, names.get(sym, sym))
        rows.append(dict(symbol=sym, day=day, n_articles=len(g), text=text))
        for ev, desc in EVENTS.items():
            prompts.append(S.format_prompt(f"News about Company X:\n{text}\n\n" + QUESTION.format(event=desc)))
            meta.append((sym, day, ev))
    t0 = time.time()
    res, ntok, _ = pit.p_yes_many(prompts, batch_size=8)
    dt = time.time() - t0
    out = pd.DataFrame([(s, d, e, p, m) for (s, d, e), (p, m) in zip(meta, res)],
                       columns=["symbol", "day", "event", "p_yes", "masse"])
    wide = out.pivot_table(index=["symbol", "day"], columns="event", values="p_yes").reset_index()
    base = pd.DataFrame(rows)[["symbol", "day", "n_articles", "text"]]
    wide = base.merge(wide, on=["symbol", "day"])
    wide["echantillon"] = ["aléatoire" if (s, d) in set(zip(rnd["symbol"], rnd["day"])) else "gros trade"
                           for s, d in zip(wide["symbol"], wide["day"])]
    wide.to_parquet("myfiles/pit_events_proto.parquet", index=False)
    print(f"{len(wide)} jours de news, {len(prompts)} passes, {ntok} tokens en {dt:.0f} s "
          f"→ {ntok / dt:.0f} tokens/s, {len(prompts) / dt:.1f} passes/s ; tokens moyens par passe {ntok / len(prompts):.0f}")
    print(f"masse Yes+No médiane : {out['masse'].median():.2f}")


if __name__ == "__main__":
    main()
