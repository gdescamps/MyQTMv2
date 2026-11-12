"""Notes de sentiment par (titre, jour) avec le LLM point-in-time Diamegs/PIT-4B-FT-201912 (GPU du gb10).

Le modèle ne génère rien : pour chaque question fermée, une passe avant et lecture des logits du premier
token de réponse → P(Yes) / (P(Yes) + P(No)). Ses connaissances s'arrêtent en 2019-12 : aucune fuite de
l'avenir sur 2020 et après.

Pour chacune des trois catégories de ref/MyQTM (company, competitors, global), deux questions sur la
rubrique correspondante du condensé (digest.py) : « bonne nouvelle pour la valeur de l'entreprise ? » et
« mauvaise nouvelle ? ». note = 5 + 5·(P_bonne − P_mauvaise) ∈ [0, 10] (5 = neutre, comme ref/MyQTM).
Rubrique « none » → 5 sans inférence.

Chaque passe avant sauvegarde aussi l'embedding « juste avant la génération » : état caché du dernier token
du prompt à l'entrée de lm_head (float16), dans data/portefolio_strategy/llm/pit/sentiment_emb/part_NNNNN.npy,
avec un index part_NNNNN.parquet (day, symbol, category, polarity, row). Les notes vont dans
llm/pit/sentiment.parquet. Même schéma et mêmes noms que les notes de Qwen (qwen.py, llm/qwen25/).

    python -m src.portefolio_strategy.sentiment --limit 20    # smoke test
    python -m src.portefolio_strategy.sentiment               # tous les condensés non encore notés
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from . import config as C
from . import digest as D
from .features import SENTIMENTS

EMB_DIR = C.PIT_DIR / "sentiment_emb"
TIMING = C.PIT_DIR / "timing.json"

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
        # état caché du dernier token à l'entrée de lm_head = embedding « juste avant la génération »
        self.cap = {}
        self.model.lm_head.register_forward_pre_hook(lambda _m, args: self.cap.__setitem__("h", args[0].detach()))
        ids = lambda ws: sorted({self.tok.encode(w, add_special_tokens=False)[0] for w in ws})  # noqa: E731
        self.yes, self.no = ids(["Yes", " Yes", "yes"]), ids(["No", " No", "no"])

    def encode(self, prompt):
        return self.tok.encode(prompt, add_special_tokens=False)[-C.PIT_CTX:]

    def logits(self, batch):
        """batch : séquences de même longueur → (logits du dernier token (B, vocab), embedding (B, dim))."""
        with self.torch.no_grad():
            x = self.torch.tensor(batch, device=self.dev)
            out = self.model(input_ids=x, logits_to_keep=1, use_cache=False)
            return out.logits[:, -1].float().cpu().numpy(), self.cap["h"][:, -1].float().cpu().numpy()

    def p_yes_many(self, prompts, batch_size=16):
        """Passe avant par lots de séquences de même longueur (pas de padding)
        → ([(p_yes, masse)], nombre de tokens, embeddings (N, dim) float16 dans l'ordre des prompts)."""
        enc = [self.encode(p) for p in prompts]
        res, emb = [None] * len(enc), [None] * len(enc)
        by_len = {}
        for i, e in enumerate(enc):
            by_len.setdefault(len(e), []).append(i)
        for idx in by_len.values():
            for j in range(0, len(idx), batch_size):
                chunk = idx[j:j + batch_size]
                zs, hs = self.logits([enc[i] for i in chunk])
                for i, z, h in zip(chunk, zs, hs):
                    res[i] = p_yes(z, self.yes, self.no)
                    emb[i] = h.astype(np.float16)
        return res, sum(len(e) for e in enc), (np.stack(emb) if emb else np.zeros((0, 0), np.float16))


# --------------------------------------------------------------------------- stockage commun (PIT, Qwen)
def load_scores(path=SENTIMENTS):
    return pd.read_parquet(path) if path.exists() else None


def pending_digests(done, limit=None):
    """Condensés Haiku pas encore notés dans `done`, par ordre chronologique."""
    dig = D.load_digests()
    if done is not None and len(done):
        dig = dig.merge(done[["day", "symbol"]].assign(_d=1), how="left", on=["day", "symbol"])
        dig = dig[dig["_d"].isna()].drop(columns="_d")
    dig = dig.sort_values(["day", "symbol"])
    return dig.head(limit) if limit else dig


def prune_orphans(emb_dir, done):
    """Supprime les lots d'embeddings écrits juste avant une interruption (leurs notes n'ont pas été sauvées)."""
    keys = set(zip(done["day"], done["symbol"])) if done is not None and len(done) else set()
    for f in sorted(emb_dir.glob("part_*.parquet")):
        ix = pd.read_parquet(f, columns=["day", "symbol"])
        if not set(zip(ix["day"], ix["symbol"])) <= keys:
            print(f"  lot d'embeddings orphelin supprimé : {f.stem}")
            f.unlink()
            f.with_suffix(".npy").unlink(missing_ok=True)


def save_emb(emb_dir, emb, part, where):
    """Un lot : part_NNNNN.npy (passes × dim, float16) + part_NNNNN.parquet (index : une ligne par passe avant).
    `where` : [(position du condensé dans `part`, catégorie, polarité)] dans l'ordre des lignes de `emb`."""
    emb_dir.mkdir(parents=True, exist_ok=True)
    no = max([int(f.stem[5:]) for f in emb_dir.glob("part_*.npy")], default=-1) + 1
    np.save(emb_dir / f"part_{no:05d}.npy", np.asarray(emb, dtype=np.float16))
    pd.DataFrame({"day": [pd.Timestamp(part[i]["day"]) for i, _, _ in where],
                  "symbol": [part[i]["symbol"] for i, _, _ in where],
                  "category": [c for _, c, _ in where], "polarity": [p for _, _, p in where],
                  "row": range(len(where))}).to_parquet(emb_dir / f"part_{no:05d}.parquet", index=False)


def score_rows(part, where, res):
    """→ une ligne par condensé : note par catégorie (5 si rubrique vide) + p_pos / p_neg / masse."""
    vals = [{} for _ in part]
    for (i, cat, pol), (p, mass) in zip(where, res):
        vals[i][f"p_{pol}_{cat}"] = p
        vals[i][f"mass_{cat}"] = mass
    rows = []
    for r, v in zip(part, vals):
        row = {"day": pd.Timestamp(r["day"]), "symbol": r["symbol"]}
        for cat in C.SENT_CATEGORIES:
            pp, pn = v.get(f"p_pos_{cat}"), v.get(f"p_neg_{cat}")
            row[cat] = score(pp, pn) if pp is not None else 5.0
            row[f"p_pos_{cat}"], row[f"p_neg_{cat}"] = pp, pn
            row[f"mass_{cat}"] = v.get(f"mass_{cat}")
        rows.append(row)
    return rows


def append_save(done, rows, path):
    out = pd.DataFrame(rows)
    if done is not None and len(done):
        out = pd.concat([done, out], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def company_names():
    from . import universe as U
    return U.load_universe().drop_duplicates("symbol").set_index("symbol")["name"].to_dict()


# --------------------------------------------------------------------------- passe PIT
def run(limit=None, chunk=400):
    names = company_names()
    done = load_scores()
    dig = pending_digests(done, limit)
    if dig.empty:
        print("rien à noter")
        return
    prune_orphans(EMB_DIR, done)
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
        res, tok, emb = pit.p_yes_many(prompts) if prompts else ([], 0, None)
        if prompts:
            save_emb(EMB_DIR, emb, part, where)
        n_fwd += len(prompts)
        n_tok += tok
        rows += score_rows(part, where, res)
        append_save(done, rows, SENTIMENTS)
        dt = time.time() - t0
        print(f"  {a + len(part)}/{len(recs)} condensés, {n_fwd} passes, {dt:.0f} s "
              f"({dt / max(n_fwd, 1) * 1000:.0f} ms/passe, {n_tok / max(n_fwd, 1):.0f} tokens/passe)", flush=True)
    dt = time.time() - t0
    print(pd.DataFrame(rows)[list(C.SENT_CATEGORIES)].describe().round(2).to_string())
    TIMING.write_text(json.dumps({"passes": n_fwd, "seconds": round(dt, 1), "digests": len(recs), "tokens": n_tok}))
    print(f"TIMING passes={n_fwd} secondes={dt:.1f} condensés={len(recs)} tokens={n_tok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    run(limit=ap.parse_args().limit)


if __name__ == "__main__":
    main()
