"""Fiche de contexte de marché quotidienne, chaînée, générée par le LLM point-in-time
Diamegs/PIT-4B-FT-201912 (coupure 12/2019) — même fiche que cnbc_market_context.py (Haiku), en anglais.

Pourquoi « une question courte par champ » : testé le 2026-11-08, le modèle 4B décroche avec le long
prompt multi-consignes de Haiku (~1500 tokens) — il ignore les textes et invente une fiche « 2019-2020 »
(Brexit, guerre commerciale). Avec UN résumé et UNE question courte, il s'ancre correctement.

Entrées (résumés stricts Haiku, myfiles/cnbc_strict_summary.py — Haiku ne fait que compresser l'article) :
- résumé LONG de J ; résumés COURTS des séances J−4…J−1 (tendance) ;
- fiche J−1 (chaînage : ses thèmes et attentes servent de mémoire pour évolution / continuité).
Champs (même structure que la fiche Haiku) :
  mood     : choix par vraisemblance parmi 6 humeurs (score des options, pas de génération libre)
  trend    : 1 phrase sur les mouvements d'indices (résumés J−4…J)
  themes   : 3-5 sujets (≤ 8 mots), avec mémoire des thèmes J−1 ; evolution = nouveau / stable / eteint
             par recouvrement de mots avec J−1 ; intensite = rang (5 pour le 1er…)
  attentes : événements à venir cités par le texte, + attentes J−1 non échues
  tensions : 1-2 inquiétudes du marché selon le texte
  resume   : 3 phrases
Séquentiel (chaînage), décodage glouton. Reprise : jours existants relus et sautés.
Sortie : data/text/cnbc/pit_context/days/YYYY-MM-DD.json (+ pit_context.parquet)
    python myfiles/pit_market_context.py --limit 5        # test
    python myfiles/pit_market_context.py                  # 2022-01-03 → fin
"""
import argparse
import json
import os
import re
import sys
import time

import pandas as pd
import torch
import transformers

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from test_pit_llm import END_MARKER, PIT_REPO, PIT_REVISION, format_prompt  # noqa: E402

SUMS = "data/text/cnbc/strict_summary/strict_summary.parquet"
FICHES_HAIKU = "data/text/cnbc/context/days"
OUT_DIR = "data/text/cnbc/pit_context/days"
OUT_PQ = "data/text/cnbc/pit_context/pit_context.parquet"
MOODS = ["risk-on", "risk-off", "rotation", "wait-and-see", "panic", "euphoria"]
STOP = {"the", "a", "an", "of", "and", "to", "in", "on", "for", "with", "by", "us", "u.s.", "stocks", "market"}


class Pit:
    def __init__(self):
        self.tok = transformers.AutoTokenizer.from_pretrained(PIT_REPO, revision=PIT_REVISION, trust_remote_code=True)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            PIT_REPO, revision=PIT_REVISION, trust_remote_code=True, dtype=torch.bfloat16).to(self.dev).eval()
        self.n_out = 0

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    @torch.no_grad()
    def ask(self, instruction, max_new):
        ids = self.enc(format_prompt(instruction))
        x = torch.tensor([ids], device=self.dev)
        out = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new,
                                  do_sample=False, repetition_penalty=1.1, pad_token_id=self.tok.eos_token_id)
        new = out[0, len(ids):]
        self.n_out += len(new)
        return self.tok.decode(new).split(END_MARKER)[0].strip()

    @torch.no_grad()
    def choose(self, instruction, options):
        """Option la plus vraisemblable (log-prob totale de la réponse)."""
        base = self.enc(format_prompt(instruction))
        best, scores = None, {}
        for o in options:
            oi = self.enc(o)
            x = torch.tensor([base + oi], device=self.dev)
            lp = torch.log_softmax(self.model(input_ids=x, use_cache=False).logits[0, len(base) - 1:-1].float(), -1)
            scores[o] = float(lp[torch.arange(len(oi)), torch.tensor(oi, device=self.dev)].sum())
        best = max(scores, key=scores.get)
        return best, scores


def lines(text, k):
    out = []
    for ln in text.splitlines():
        ln = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", ln).strip().rstrip(".")
        if 2 < len(ln) < 200 and ln not in out:
            out.append(ln)
    return out[:k]


def words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2}


def fiche_for_day(p, s, i, prev):
    d, today = s.index[i], s.iloc[i].long
    recent = "\n".join(f"- {s.index[j]:%a %d %b}: {s.iloc[j].short}" for j in range(max(0, i - 4), i))
    t0 = f"Text (today's market close):\n{today}\n\n"
    prev_themes = [t["theme"] for t in prev["themes"]] if prev else []
    mood, mood_scores = p.choose(t0 + "Which word best describes the stock market mood today: risk-on, risk-off, "
                                 "rotation, wait-and-see, panic or euphoria? Answer with one word.", MOODS)
    trend = p.ask(f"Previous sessions:\n{recent}\n\n{t0}In one sentence, describe the recent trend of the "
                  "stock indexes (levels, % moves, streak).", 70)
    mem = ("Topics that were moving the market in previous days: " + "; ".join(prev_themes) + ".\n") if prev_themes else ""
    th = lines(p.ask(t0 + mem + "List the 3 to 5 main topics moving the stock market now according to the text, "
                     "one per line, each in under 8 words.", 70), 5)
    themes = []
    for r, t in enumerate(th):
        w = words(t)
        match = max(prev_themes, key=lambda x: len(w & words(x)), default=None)
        evo = "stable" if match and len(w & words(match)) >= 2 else "nouveau"
        detail = p.ask(t0 + f"In one factual sentence, what does the text say about: {t}?", 40)
        themes.append({"theme": t, "intensite": 5 - r, "evolution": evo, "detail": detail})
    kept = {t["theme"] for t in themes}
    for pt in prev_themes:   # thème J−1 disparu → marqué « eteint » une fois
        if not any(len(words(pt) & words(k)) >= 2 for k in kept) and not pt.startswith("[eteint]"):
            themes.append({"theme": pt, "intensite": 1, "evolution": "eteint", "detail": ""})
    ex = lines(p.ask(t0 + "Which future events mentioned in the text are investors waiting for? "
                     "One per line, under 12 words each.", 60), 4)
    attentes = [{"evenement": e, "echeance": "", "attente": ""} for e in ex
                if not re.match(r"(none|no |there (are|is) no)", e.lower())]
    tn = lines(p.ask(t0 + "In one sentence, what are investors worried about according to the text?", 50), 1)
    tensions = [{"sujet": x[:60], "justification": x} for x in tn]
    resume = p.ask(t0 + "Summarize the market context at today's close in 3 sentences.", 110)
    return {"humeur": mood, "humeur_scores": mood_scores, "tendance": trend, "themes": themes,
            "attentes": attentes, "tensions": tensions, "resume": resume}


def day_path(d):
    return os.path.join(OUT_DIR, f"{d:%Y-%m-%d}.json")


def aggregate():
    rows = []
    for f in sorted(os.listdir(OUT_DIR)):
        r = json.load(open(os.path.join(OUT_DIR, f)))
        fi = r["fiche"]
        rows.append({"date": pd.Timestamp(r["date"]), "humeur": fi["humeur"], "tendance": fi["tendance"],
                     "resume": fi["resume"], "themes": json.dumps(fi["themes"]), "attentes": json.dumps(fi["attentes"]),
                     "tensions": json.dumps(fi["tensions"]), "n_out": r["n_out"], "secs": r["secs"]})
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PQ, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-03")
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    s = pd.read_parquet(SUMS).set_index("date").sort_index()
    haiku_days = {f[:10] for f in os.listdir(FICHES_HAIKU)}
    idx = [i for i, d in enumerate(s.index) if d >= pd.Timestamp(a.start) and f"{d:%Y-%m-%d}" in haiku_days
           and (a.end is None or d <= pd.Timestamp(a.end))]
    if a.limit:
        idx = idx[:a.limit]
    p = Pit()
    prev = None
    before = [day_path(d) for d in s.index[:idx[0]] if os.path.exists(day_path(d))] if idx else []
    if before:
        prev = json.load(open(before[-1]))["fiche"]
    t0, n = time.time(), 0
    for i in idx:
        d = s.index[i]
        if os.path.exists(day_path(d)):
            prev = json.load(open(day_path(d)))["fiche"]
            continue
        t, k0 = time.time(), p.n_out
        fiche = fiche_for_day(p, s, i, prev)
        rec = {"date": f"{d:%Y-%m-%d}", "fiche": fiche, "n_out": p.n_out - k0, "secs": round(time.time() - t, 1)}
        with open(day_path(d), "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        prev = {**fiche, "themes": [x for x in fiche["themes"] if x["evolution"] != "eteint"]}
        n += 1
        if n % 10 == 0 or i == idx[-1]:
            el = time.time() - t0
            left = sum(1 for j in idx if not os.path.exists(day_path(s.index[j])))
            print(f"  {d:%Y-%m-%d}  {n} fiches  {el / n:.1f} s/fiche  reste {left} ≈ {left * el / n / 3600:.1f} h",
                  flush=True)
    df = aggregate()
    print(f"{len(df)} fiches PIT → {OUT_PQ}")


if __name__ == "__main__":
    main()
