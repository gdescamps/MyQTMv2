"""Notes de sentiment par (titre, jour) avec le LLM point-in-time Diamegs/PIT-4B-FT-201912 (GPU du gb10).

Le modèle ne génère rien : pour chaque question fermée, une passe avant et lecture des logits du premier
token de réponse → P(Yes) / (P(Yes) + P(No)). Ses connaissances s'arrêtent en 2019-12 : aucune fuite de
l'avenir sur 2020 et après.

Pour chacune des trois catégories de ref/MyQTM (company, competitors, global), deux questions sur la
rubrique correspondante du condensé (digest.py) : « bonne nouvelle pour la valeur de l'entreprise ? » et
« mauvaise nouvelle ? ». note = 5 + 5·(P_bonne − P_mauvaise) ∈ [0, 10] (5 = neutre, comme ref/MyQTM).
Rubrique « none » → 5 sans inférence.

    python -m src.portefolio_strategy.sentiment --limit 20    # smoke test
    python -m src.portefolio_strategy.sentiment               # tous les condensés non encore notés
"""
import argparse
import time

import numpy as np
import pandas as pd

from . import config as C
from . import digest as D
from .features import SENTIMENTS

# Formulation retenue après essai de 7 variantes sur des cas témoins (4 positifs, 4 négatifs, 2 neutres) :
# c'est celle qui sépare le mieux (signe correct sur les 8 cas, masse Yes+No = 0,62 du premier token).
# Le modèle répond « Yes » par défaut : l'écart P_bonne − P_mauvaise reste petit (±0,2), d'où des notes
# resserrées autour de 5. Sans effet sur XGBoost (arbres : seul l'ordre compte).
QUESTION = "Question: Is this {word} news for {name}? Answer with exactly one word, Yes or No."
POLARITIES = (("pos", "good"), ("neg", "bad"))
SECTION_LABEL = {"company": "", "competitors": "News about the competitors of {name}:\n",
                 "global": "News about global factors (politics, regulation, economy) affecting {name}:\n"}


def format_prompt(instruction):
    return f"<|user|>\n{instruction}\n<|assistant|>\n"


def build_prompt(name, category, text, word):
    return format_prompt(f"{SECTION_LABEL[category].format(name=name)}{text}\n\n"
                         + QUESTION.format(word=word, name=name))


def p_yes(logits, yes_ids, no_ids):
    """logits (vocab,) du premier token de réponse → (P(Yes) normalisée, masse Yes+No)."""
    z = np.asarray(logits, dtype=np.float64)
    p = np.exp(z - z.max())
    p /= p.sum()
    y, n = p[yes_ids].sum(), p[no_ids].sum()
    return float(y / (y + n)), float(y + n)


def score(p_pos, p_neg):
    return round(5.0 + 5.0 * (p_pos - p_neg), 3)


class Pit:
    def __init__(self):
        import torch
        import transformers
        self.torch = torch
        self.tok = transformers.AutoTokenizer.from_pretrained(C.PIT_REPO, revision=C.PIT_REVISION,
                                                              trust_remote_code=True)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            C.PIT_REPO, revision=C.PIT_REVISION, trust_remote_code=True, dtype=torch.bfloat16).to(self.dev).eval()
        ids = lambda ws: sorted({self.tok.encode(w, add_special_tokens=False)[0] for w in ws})  # noqa: E731
        self.yes, self.no = ids(["Yes", " Yes", "yes"]), ids(["No", " No", "no"])

    def encode(self, prompt):
        return self.tok.encode(prompt, add_special_tokens=False)[-C.PIT_CTX:]

    def logits(self, batch):
        """batch : séquences de même longueur → logits du dernier token (B, vocab)."""
        with self.torch.no_grad():
            x = self.torch.tensor(batch, device=self.dev)
            out = self.model(input_ids=x, logits_to_keep=1, use_cache=False)
            return out.logits[:, -1].float().cpu().numpy()

    def p_yes_many(self, prompts, batch_size=16):
        """Passe avant par lots de séquences de même longueur (pas de padding) → [(p_yes, masse)]."""
        enc = [self.encode(p) for p in prompts]
        res = [None] * len(enc)
        by_len = {}
        for i, e in enumerate(enc):
            by_len.setdefault(len(e), []).append(i)
        for idx in by_len.values():
            for j in range(0, len(idx), batch_size):
                chunk = idx[j:j + batch_size]
                for i, z in zip(chunk, self.logits([enc[i] for i in chunk])):
                    res[i] = p_yes(z, self.yes, self.no)
        return res, sum(len(e) for e in enc)


def load_scores():
    return pd.read_parquet(SENTIMENTS) if SENTIMENTS.exists() else None


def run(limit=None, chunk=400):
    from . import universe as U
    dig = D.load_digests()
    names = U.load_universe().drop_duplicates("symbol").set_index("symbol")["name"].to_dict()
    done = load_scores()
    if done is not None and len(done):
        k = done[["day", "symbol"]].assign(_d=1)
        dig = dig.merge(k, how="left", on=["day", "symbol"])
        dig = dig[dig["_d"].isna()].drop(columns="_d")
    dig = dig.sort_values(["day", "symbol"])
    if limit:
        dig = dig.head(limit)
    if dig.empty:
        print("rien à noter")
        return
    pit = Pit()
    rows, n_fwd, n_tok, t0 = [], 0, 0, time.time()
    recs = dig.to_dict("records")
    for a in range(0, len(recs), chunk):
        part, prompts, where = recs[a:a + chunk], [], []
        for i, r in enumerate(part):
            for cat in C.SENT_CATEGORIES:
                if r[cat] and r[cat] != "none":
                    for pol, word in POLARITIES:
                        prompts.append(build_prompt(names.get(r["symbol"], r["symbol"]), cat, r[cat], word))
                        where.append((i, cat, pol))
        res, tok = pit.p_yes_many(prompts) if prompts else ([], 0)
        n_fwd += len(prompts)
        n_tok += tok
        vals = [{} for _ in part]
        for (i, cat, pol), (p, mass) in zip(where, res):
            vals[i][f"p_{pol}_{cat}"] = p
            vals[i][f"mass_{cat}"] = mass
        for r, v in zip(part, vals):
            row = {"day": pd.Timestamp(r["day"]), "symbol": r["symbol"]}
            for cat in C.SENT_CATEGORIES:
                pp, pn = v.get(f"p_pos_{cat}"), v.get(f"p_neg_{cat}")
                row[cat] = score(pp, pn) if pp is not None else 5.0
                row[f"p_pos_{cat}"], row[f"p_neg_{cat}"] = pp, pn
                row[f"mass_{cat}"] = v.get(f"mass_{cat}")
            rows.append(row)
        out = pd.DataFrame(rows)
        if done is not None and len(done):
            out = pd.concat([done, out], ignore_index=True)
        out.to_parquet(SENTIMENTS, index=False)
        dt = time.time() - t0
        print(f"  {a + len(part)}/{len(recs)} condensés, {n_fwd} passes, {dt:.0f} s "
              f"({dt / max(n_fwd, 1) * 1000:.0f} ms/passe, {n_tok / max(n_fwd, 1):.0f} tokens/passe)", flush=True)
    dt = time.time() - t0
    new = pd.DataFrame(rows)
    print(new[list(C.SENT_CATEGORIES)].describe().round(2).to_string())
    import json
    (C.DATA / "pit_timing.json").write_text(json.dumps(
        {"passes": n_fwd, "seconds": round(dt, 1), "digests": len(recs), "tokens": n_tok}))
    print(f"TIMING passes={n_fwd} secondes={dt:.1f} condensés={len(recs)} tokens={n_tok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    run(limit=ap.parse_args().limit)


if __name__ == "__main__":
    main()
