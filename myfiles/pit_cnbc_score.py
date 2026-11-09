"""Note de sentiment « fondamental » + embeddings des clôtures CNBC avec le LLM point-in-time
Diamegs/PIT-4B-FT-201912 (coupure des connaissances : décembre 2019), pour comparaison avec Haiku.

Modèle/gabarit/révision : tests/test_pit_llm.py. Le modèle ne connaît rien après 2019 → aucune
fuite de l'avenir sur 2022-2026 (contrairement à Haiku, coupure 06/2026).

Entrée : l'article CNBC de clôture (anglais, title + body) des séances qui ont une fiche Haiku
(data/text/cnbc/context/days). On note l'article source et non la fiche (française, rédigée par
Haiku → contaminée par sa connaissance).

Par article, trois passes avant (aucune génération) :
1. emb_raw : embedding « Point-In-Time-LLM » (DjoFE2021, embeddings/experiments/4b/main.py) =
   article brut sans gabarit, tronqué à 2048 tokens, état caché du DERNIER token à l'entrée de
   lm_head (après la rms_norm finale), float32, non normalisé.
2. p_up : prompt d'instruction « les fondamentaux se sont-ils AMÉLIORÉS (faits tangibles seulement,
   humeur et cours ignorés) ? Answer Yes or No. » → P(Yes)/(P(Yes)+P(No)) sur le 1er token de
   réponse ; emb_q = état caché du dernier token du prompt (= représentation juste avant la génération).
3. p_dn : même question « DÉTÉRIORÉS ».
score = p_up − p_dn ∈ [−1, 1].

Sorties : data/text/cnbc/pit/pit_201912.parquet (date, p_up, p_dn, masses, score, n_tokens)
          data/text/cnbc/pit/emb_raw.npy, emb_q.npy (n × 4096, float32), dates dans le parquet.
    python myfiles/pit_cnbc_score.py --limit 3     # test
    python myfiles/pit_cnbc_score.py
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import transformers

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from test_pit_llm import PIT_REPO, PIT_REVISION, format_prompt  # noqa: E402

CNBC = "data/text/cnbc/cnbc_close.parquet"
FICHES = "data/text/cnbc/context/days"
OUT_DIR = "data/text/cnbc/pit"
CTX = 2048
ARTICLE_MAX_Q = 1700   # tokens d'article dans le prompt d'instruction (reste = consigne + réponse)

QUESTION = ("Based only on tangible economic facts reported in this market close article (economic data "
            "releases versus expectations, company earnings and guidance, central bank or government policy "
            "decisions, credit events), and ignoring market mood, fears, analyst opinions and the stock price "
            "moves themselves: did the economic fundamentals {verb} on this day? Answer Yes or No.")


class Pit:
    def __init__(self):
        self.tok = transformers.AutoTokenizer.from_pretrained(PIT_REPO, revision=PIT_REVISION,
                                                              trust_remote_code=True)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            PIT_REPO, revision=PIT_REVISION, trust_remote_code=True, dtype=torch.bfloat16,
        ).to(self.dev).eval()
        self.cap = {}
        self.model.lm_head.register_forward_pre_hook(lambda _m, args: self.cap.__setitem__("h", args[0].detach()))
        ids = lambda ws: sorted({self.tok.encode(w, add_special_tokens=False)[0] for w in ws})  # noqa: E731
        self.yes, self.no = ids(["Yes", " Yes", "yes"]), ids(["No", " No", "no"])

    @torch.no_grad()
    def last(self, ids):
        """Forward sur une séquence → (état caché du dernier token à l'entrée de lm_head, logits)."""
        x = torch.tensor([ids], device=self.dev)
        out = self.model(input_ids=x, logits_to_keep=1, use_cache=False)
        return self.cap["h"][0, -1].float().cpu().numpy(), out.logits[0, -1].float()

    def p_yes(self, article_ids, verb):
        q = format_prompt("Article:\n" + self.tok.decode(article_ids[:ARTICLE_MAX_Q]) + "\n\n"
                          + QUESTION.format(verb=verb))
        h, logits = self.last(self.tok.encode(q, add_special_tokens=False)[-CTX:])
        p = torch.softmax(logits, -1)
        y, n = float(p[self.yes].sum()), float(p[self.no].sum())
        return y / (y + n), y + n, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    dates = sorted(f[:10] for f in os.listdir(FICHES))
    c = pd.read_parquet(CNBC).set_index("date").sort_index()
    c = c[c.index.strftime("%Y-%m-%d").isin(dates)]
    if a.limit:
        c = c.iloc[:a.limit]
    pit = Pit()
    rows, e_raw, e_q = [], [], []
    t0 = time.time()
    for i, (d, r) in enumerate(c.iterrows()):
        art = pit.tok.encode(f"{r.title}\n\n{r.body}", add_special_tokens=False)
        h_raw, _ = pit.last(art[:CTX])
        pu, mu, h_q = pit.p_yes(art, "improve")
        pd_, md, _ = pit.p_yes(art, "deteriorate")
        rows.append({"date": d, "p_up": pu, "p_dn": pd_, "mass_up": mu, "mass_dn": md,
                     "score": pu - pd_, "n_tokens": len(art)})
        e_raw.append(h_raw)
        e_q.append(h_q)
        if (i + 1) % 50 == 0 or i + 1 == len(c):
            print(f"  {i + 1}/{len(c)}  {d:%Y-%m-%d}  {(time.time() - t0) / (i + 1):.2f} s/article", flush=True)
    df = pd.DataFrame(rows)
    sfx = "" if not a.limit else "_test"
    df.to_parquet(f"{OUT_DIR}/pit_201912{sfx}.parquet", index=False)
    np.save(f"{OUT_DIR}/emb_raw{sfx}.npy", np.stack(e_raw).astype(np.float32))
    np.save(f"{OUT_DIR}/emb_q{sfx}.npy", np.stack(e_q).astype(np.float32))
    print(df.describe().round(3).to_string())
    print(f"→ {OUT_DIR}/pit_201912{sfx}.parquet + emb_raw{sfx}.npy / emb_q{sfx}.npy")


if __name__ == "__main__":
    main()
