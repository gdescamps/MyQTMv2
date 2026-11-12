"""Estimations AVANT de lancer le traitement complet du corpus : coût Haiku et temps GPU du gb10.

Fondées sur des mesures réelles : volumes de data/portefolio_strategy/news, tokens facturés
sur les condensés déjà faits (smoke test), vitesse mesurée du LLM PIT sur le GPU.

    python -m src.portefolio_strategy.estimate
"""
import json

import numpy as np

from . import config as C
from . import digest as D
from . import news as N


def compute():
    news = N.load_news()
    news["chars"] = news["title"].str.len() + news["content"].str.len()
    pairs = news.groupby(["day", "symbol"]).agg(chars=("chars", "sum"), n=("chars", "size")).reset_index()
    done = D.load_digests()
    e = {"pairs": len(pairs), "articles": len(news), "chars": int(pairs["chars"].sum()),
         "pairs_by_year": {int(k): int(v) for k, v in pairs.groupby(pairs["day"].dt.year).size().items()},
         "done": len(done)}
    if len(done) >= 10:
        m = done.merge(pairs, on=["day", "symbol"])
        slope, const = np.polyfit(m["chars"], m["in_tok"], 1)          # tokens = const + chars × slope
        todo = pairs.merge(done[["day", "symbol"]].assign(_d=1), how="left", on=["day", "symbol"])
        todo = todo[todo["_d"].isna()]
        e["chars_per_token"] = round(1 / slope, 2)
        e["in_tok"] = int((const + slope * todo["chars"]).sum())
        # la sortie croît avec le nombre d'articles : moyenne par nombre d'articles, mesurée
        out_by_n = m.groupby("n")["out_tok"].mean()
        e["out_tok"] = int(todo["n"].map(out_by_n).fillna(m["out_tok"].mean()).sum())
        e["out_tok_max"] = int(len(todo) * C.HAIKU_MAX_TOKENS)
        e["cost"] = D.cost(e["in_tok"], e["out_tok"])
        e["cost_max"] = D.cost(e["in_tok"], e["out_tok_max"])
        e["todo"] = len(todo)
        e["none_share"] = {c: round(float((done[c] == "none").mean()), 2) for c in C.SENT_CATEGORIES}
        t = C.HAIKU_DIR / "timing.json"
        if t.exists():
            h = json.loads(t.read_text())
            e["haiku_s_per_call"] = h["seconds"] * h["workers"] / h["calls"]
        p = C.PIT_DIR / "timing.json"
        if p.exists():
            g = json.loads(p.read_text())
            e["pit_passes_per_digest"] = g["passes"] / g["digests"]
            e["pit_s_per_pass"] = g["seconds"] / g["passes"]
            e["pit_tokens_per_pass"] = g["tokens"] / g["passes"]
            e["pit_passes"] = int(len(pairs) * e["pit_passes_per_digest"])
            e["pit_hours"] = e["pit_passes"] * e["pit_s_per_pass"] / 3600
    return e


def _n(x):
    return f"{x:,}".replace(",", " ")


def report(e):
    L = ["# Estimations avant traitement complet du corpus de news", "",
         f"Corpus retenu (univers Nasdaq dynamique, depuis {C.START_DATE}) : **{_n(e['pairs'])} couples (titre, jour)**, "
         f"{_n(e['articles'])} articles, {e['chars'] / 1e6:.0f} M de caractères.", "",
         "| Année | Couples (titre, jour) |", "|---|---|"]
    L += [f"| {y} | {_n(n)} |" for y, n in e["pairs_by_year"].items()]
    L.append("")
    if "cost" not in e:
        L.append("Pas assez de condensés mesurés : lancer d'abord `python -m src.portefolio_strategy.digest --limit 40`.")
        return "\n".join(L)
    L += ["## Coût Haiku (condensés factuels, digest.py)", "",
          f"Mesuré sur {e['done']} appels réels : {e['chars_per_token']} caractères par token d'entrée.", "",
          "| | Tokens | Tarif ($/MTok) | Coût |", "|---|---|---|---|",
          f"| Entrée | {e['in_tok'] / 1e6:.0f} M | {C.HAIKU_PRICE_IN} | {e['in_tok'] * C.HAIKU_PRICE_IN / 1e6:.1f} $ |",
          f"| Sortie (attendue) | {e['out_tok'] / 1e6:.1f} M | {C.HAIKU_PRICE_OUT} | {e['out_tok'] * C.HAIKU_PRICE_OUT / 1e6:.1f} $ |",
          f"| **Total attendu** | | | **{e['cost']:.0f} $** |",
          f"| Plafond (toutes les sorties à {C.HAIKU_MAX_TOKENS} tokens) | | | {e['cost_max']:.0f} $ |", "",
          f"- Reste à traiter : {_n(e['todo'])} couples.",
          "- Tarif = tarif de l'API Anthropic pour Claude Haiku 5.5 (appels directs, clé ANTHROPIC_API). L'API "
          "Message Batches ferait −50 % (résultats sous 24 h) ; digest.py ne l'utilise pas.",
          "- Durée : " + (f"{e['todo'] * e['haiku_s_per_call'] / C.WORKERS / 3600:.1f} h avec {C.WORKERS} appels simultanés "
                          f"({e['haiku_s_per_call']:.1f} s par appel mesuré) ; {e['todo'] * e['haiku_s_per_call'] / 20 / 3600:.1f} h "
                          f"avec `--workers 20` si les limites de débit le permettent (appels réseau, pas des cœurs)."
                          if "haiku_s_per_call" in e else "non mesurée."), ""]
    if "pit_hours" in e:
        L += ["## Temps GPU du gb10 (notes de sentiment PIT-4B-FT-201912, sentiment.py)", "",
              f"Mesuré : {e['pit_s_per_pass'] * 1000:.0f} ms par passe avant ({e['pit_tokens_per_pass']:.0f} tokens en moyenne), "
              f"{e['pit_passes_per_digest']:.2f} passes par condensé (2 questions par rubrique non vide ; "
              f"rubriques vides : {e['none_share']}).", "",
              f"- {_n(e['pit_passes'])} passes avant → **{e['pit_hours']:.1f} h de GPU** (une seule passe par question, "
              "aucun token généré).",
              "- Mémoire GPU : environ 8 Go (poids en bfloat16). 1 cœur CPU.", ""]
    return "\n".join(L)


def main():
    e = compute()
    txt = report(e)
    C.OUT.mkdir(parents=True, exist_ok=True)
    (C.OUT / "ESTIMATES.md").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
