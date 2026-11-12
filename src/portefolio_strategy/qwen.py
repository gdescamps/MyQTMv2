"""Types de news et notes de sentiment par (titre, jour) avec Qwen2.5-7B-Instruct (GPU du gb10).

Même principe que sentiment.py (LLM PIT) : aucune génération, une passe avant par question fermée et lecture
des logits du premier token de réponse → P(Yes) / (P(Yes) + P(No)). Entrée : les condensés Haiku
(llm/haiku/digests.jsonl). Sorties dans data/portefolio_strategy/llm/qwen25/, mêmes noms et mêmes clés
(day, symbol) que llm/pit/ :

- sentiment.parquet     : même schéma que les notes PIT — pour chaque rubrique non vide (company, competitors,
                          global), « bonne nouvelle ? » et « mauvaise nouvelle ? » → note = 5 + 5·(P_bonne − P_mauvaise) ;
- sentiment_emb/        : embedding de chaque passe de sentiment (état caché du dernier token à l'entrée de
                          lm_head, float16), part_NNNNN.npy + index part_NNNNN.parquet ;
- categories.parquet    : p_<type> pour les types de news de config.NEWS_TYPES (une question par type, sur la
                          rubrique company ; NaN si elle est vide) ;
- timing.json.

ATTENTION, fuite temporelle : Qwen2.5 connaît les événements jusqu'à 2024 environ. Ses notes ne sont pas
point-in-time, contrairement à celles du LLM PIT — à comparer, pas à utiliser telles quelles dans un backtest
antérieur à sa date de coupure.

Vitesse : les questions d'une même rubrique partagent leur préfixe (texte du condensé) ; il est calculé une
seule fois (cache clé-valeur), puis chaque question ne coûte que ses ~25 tokens.

    python -m src.portefolio_strategy.qwen --limit 40     # smoke test
    python -m src.portefolio_strategy.qwen                # tous les condensés non encore traités
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from . import config as C
from . import sentiment as S

SENTIMENTS = C.QWEN_DIR / "sentiment.parquet"
CATEGORIES = C.QWEN_DIR / "categories.parquet"
EMB_DIR = C.QWEN_DIR / "sentiment_emb"
TIMING = C.QWEN_DIR / "timing.json"

ANSWER = " Answer with exactly one word, Yes or No."
Q_SENT = "Question: Is this {word} news for the company?" + ANSWER
Q_TYPE = "Question: Is this news about {desc}?" + ANSWER
MAX_TEXT_TOKENS = 1024


def section_text(name, category, text):
    return f"{S.SECTION_LABEL[category].format(name=name)}{text}\n\n"


def questions(category):
    """[(sorte, clé, question)] posées sur une rubrique : sentiment partout, types de news sur company."""
    q = [("sent", pol, Q_SENT.format(word=word)) for pol, word in S.POLARITIES]
    if category == "company":
        q += [("type", t, Q_TYPE.format(desc=d)) for t, d in C.NEWS_TYPES.items()]
    return q


class Qwen:
    def __init__(self):
        import torch
        import transformers
        self.torch = torch
        self.tok = transformers.AutoTokenizer.from_pretrained(C.QWEN_REPO, revision=C.QWEN_REVISION)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            C.QWEN_REPO, revision=C.QWEN_REVISION, dtype=torch.bfloat16).to(self.dev).eval()
        tpl = self.tok.apply_chat_template([{"role": "user", "content": "\x00"}], add_generation_prompt=True,
                                           tokenize=False)
        self.head, self.tail = tpl.split("\x00")
        ids = lambda ws: sorted({self.tok.encode(w, add_special_tokens=False)[0] for w in ws})  # noqa: E731
        self.yes, self.no = ids(["Yes", " Yes", "yes"]), ids(["No", " No", "no"])
        self._suffix = {}

    def prefix_ids(self, text):
        return (self.tok.encode(self.head, add_special_tokens=False)
                + self.tok.encode(text, add_special_tokens=False)[:MAX_TEXT_TOKENS])

    def suffix_ids(self, question):
        if question not in self._suffix:
            self._suffix[question] = self.tok.encode(question + self.tail, add_special_tokens=False)
        return self._suffix[question]

    def _read(self, hidden):
        """état caché du dernier token (B, dim) → (P(Yes) normalisée, masse Yes+No, embedding float16)."""
        p = self.torch.softmax(self.model.lm_head(hidden).float(), dim=-1)
        y, n = p[:, self.yes].sum(1), p[:, self.no].sum(1)
        return (y / (y + n)).cpu().numpy(), (y + n).cpu().numpy(), hidden.float().cpu().numpy().astype(np.float16)

    def ask(self, prefixes, qs):
        """prefixes : séquences de tokens (complétées à gauche à la même longueur, masquées) ; qs : questions
        → {question: (p_yes, masse, emb)}. Le préfixe passe une fois dans le modèle ; chaque question repart
        de son cache clé-valeur."""
        t = self.torch
        n = max(len(p) for p in prefixes)
        pad = self.tok.pad_token_id
        with t.no_grad():
            x = t.tensor([[pad] * (n - len(p)) + p for p in prefixes], device=self.dev)
            m = t.tensor([[0] * (n - len(p)) + [1] * len(p) for p in prefixes], device=self.dev)
            past = self.model.model(input_ids=x, attention_mask=m, use_cache=True).past_key_values
            out = {}
            for q in qs:
                sx = t.tensor([self.suffix_ids(q)] * len(prefixes), device=self.dev)
                o = self.model.model(input_ids=sx, attention_mask=t.cat([m, t.ones_like(sx)], 1),
                                     past_key_values=past, use_cache=True)
                out[q] = self._read(o.last_hidden_state[:, -1])
                past = o.past_key_values
                past.crop(-sx.shape[1])               # retire la question du cache avant la suivante
            return out

    def ask_plain(self, prefix, question):
        """Référence sans cache (tests) : une passe sur préfixe + question."""
        with self.torch.no_grad():
            x = self.torch.tensor([prefix + self.suffix_ids(question)], device=self.dev)
            return self._read(self.model.model(input_ids=x).last_hidden_state[:, -1])


def process(qwen, part, names, batch_size=16):
    """part : condensés → (lignes de sentiment, lignes de types, embeddings, where, passes, tokens)."""
    items = []                                        # (position du condensé, rubrique, tokens du préfixe)
    for i, r in enumerate(part):
        for cat in C.SENT_CATEGORIES:
            if r[cat] and r[cat] != "none":
                items.append((i, cat, qwen.prefix_ids(section_text(names.get(r["symbol"], r["symbol"]), cat, r[cat]))))
    res, where, emb, types = [], [], [], [{} for _ in part]
    n_fwd = n_tok = 0
    for is_company in (True, False):                  # lots de longueurs voisines (peu de remplissage)
        idx = sorted((k for k, it in enumerate(items) if (it[1] == "company") == is_company),
                     key=lambda k: len(items[k][2]))
        qs = questions("company" if is_company else "global")
        for j in range(0, len(idx), batch_size):
            ch = idx[j:j + batch_size]
            ans = qwen.ask([items[k][2] for k in ch], [q for _, _, q in qs])
            n_fwd += len(ch) * len(qs)
            n_tok += sum(len(items[k][2]) for k in ch) + len(ch) * sum(len(qwen.suffix_ids(q)) for _, _, q in qs)
            for kind, key, q in qs:
                p, mass, h = ans[q]
                for b, k in enumerate(ch):
                    i, cat, _ = items[k]
                    if kind == "sent":
                        where.append((i, cat, key))
                        res.append((float(p[b]), float(mass[b])))
                        emb.append(h[b])
                    else:
                        types[i][f"p_{key}"] = round(float(p[b]), 5)
    order = sorted(range(len(where)), key=lambda k: (where[k][0], C.SENT_CATEGORIES.index(where[k][1]), where[k][2] != "pos"))
    where, res, emb = [where[k] for k in order], [res[k] for k in order], [emb[k] for k in order]
    type_rows = [{"day": pd.Timestamp(r["day"]), "symbol": r["symbol"],
                  **{f"p_{t}": v.get(f"p_{t}", np.nan) for t in C.NEWS_TYPES}} for r, v in zip(part, types)]
    return S.score_rows(part, where, res), type_rows, emb, where, n_fwd, n_tok


def run(limit=None, chunk=400):
    names = S.company_names()
    done = S.load_scores(SENTIMENTS)
    done_types = S.load_scores(CATEGORIES)
    if done_types is not None and done is not None:   # types écrits juste avant une interruption
        done_types = done_types.merge(done[["day", "symbol"]], on=["day", "symbol"])
    elif done is None:
        done_types = None
    dig = S.pending_digests(done, limit)
    if dig.empty:
        print("rien à traiter")
        return
    S.prune_orphans(EMB_DIR, done)
    qwen = Qwen()
    rows, type_rows, n_fwd, n_tok, t0 = [], [], 0, 0, time.time()
    recs = dig.to_dict("records")
    for a in range(0, len(recs), chunk):
        part = recs[a:a + chunk]
        r, t, emb, where, f, k = process(qwen, part, names)
        if where:
            S.save_emb(EMB_DIR, np.stack(emb), part, where)
        rows += r
        type_rows += t
        n_fwd += f
        n_tok += k
        S.append_save(done_types, type_rows, CATEGORIES)
        S.append_save(done, rows, SENTIMENTS)         # écrit en dernier : c'est lui qui marque le lot comme fait
        dt = time.time() - t0
        print(f"  {a + len(part)}/{len(recs)} condensés, {n_fwd} questions, {dt:.0f} s "
              f"({dt / max(a + len(part), 1) * 1000:.0f} ms/condensé, {n_tok / max(dt, 1e-9):.0f} tokens/s)", flush=True)
    dt = time.time() - t0
    print(pd.DataFrame(rows)[list(C.SENT_CATEGORIES)].describe().round(2).to_string())
    print(pd.DataFrame(type_rows)[[f"p_{t}" for t in C.NEWS_TYPES]].describe().round(2).to_string())
    TIMING.write_text(json.dumps({"passes": n_fwd, "seconds": round(dt, 1), "digests": len(recs), "tokens": n_tok}))
    print(f"TIMING questions={n_fwd} secondes={dt:.1f} condensés={len(recs)} tokens={n_tok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    run(limit=ap.parse_args().limit)


if __name__ == "__main__":
    main()
