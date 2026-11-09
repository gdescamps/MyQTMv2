"""Embeddings point-in-time des clôtures CNBC avec le BON millésime PIT pour chaque année (sans fuite).

Principe (Diamegs/PIT-4B-FT-YYYYMM, coupure fin YYYY-1 pour rejouer l'année YYYY) :
- l'année de test Y est lue par le millésime coupé à fin Y−1 (2026 → 202412, le plus récent) ;
- les espaces d'embedding diffèrent d'un millésime à l'autre → chaque millésime encode AUSSI tout
  l'historique 2007 → fin de ses années de test, pour qu'une régression soit apprise sur le passé
  DANS SON PROPRE ESPACE puis appliquée à son année de test (walk-forward par millésime).
Embedding = état caché du dernier token (entrée de lm_head) du prompt-question « fundamentals improve ? »
(comme emb_q de pit_cnbc_score.py : 0.55 de corrélation avec Haiku contre 0.24 pour l'article brut),
+ p_up (P(Yes)) gratuit sur la même passe.

Sécurité : révisions HF figées (SHA ci-dessous) ; code distant modeling/configuration_pit.py identique
(même blob) sur les 12 millésimes à celui relu pour 201912 — revérifié avant chargement.
Disque : chaque millésime (~17.8 Go) est supprimé du cache HF après usage, sauf 201912 (déjà présent).
Sortie : data/text/cnbc/pit_vintage/{vintage}.npz (dates, emb float16 n×4096, p_up, test_years). Reprise.
    python -m myfiles.pit_vintage_embed                 # tous (201912 d'abord)
    python -m myfiles.pit_vintage_embed --only 201912   # pilote
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import transformers
from huggingface_hub import HfApi, scan_cache_dir, snapshot_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from test_pit_llm import format_prompt  # noqa: E402

CNBC = "data/text/cnbc/cnbc_close.parquet"
OUT = "data/text/cnbc/pit_vintage"
KEEP = {"201912"}
CODE_BLOBS = {"configuration_pit.py": "359d6c04c0", "modeling_pit.py": "3cdba597fc"}
REVS = {
    "201312": "e3e475e329f268cc87ffb5b4fb149184168e6025",
    "201412": "b523e465bfa4eacdbd26d37a66e4223b013f3483",
    "201511": "32bd64b647eb840a284be6cd80857d6148c38748",
    "201612": "2c65885f4a85c6b6bb1dc8226954e5108515d1ff",
    "201712": "b03125e2759268dafed8b64b7fe3e3e41ae95e66",
    "201812": "a93820f680b7dd91732ed769e92b47b8fe3986c0",
    "201912": "dd482b6a647f86b963626355a8e78d5588f63803",
    "202012": "faa015fb0354736b905879b3ce992a4e6872db59",
    "202112": "efe304f98759af121eaaf33f659c12f1b8d12099",
    "202212": "fc4eaefde5022a4c9aa6f201c4803e838583a91e",
    "202312": "bf786611e28561eabd38d17d145411eed640fe6c",
    "202412": "c82f0c11e583b5a95202ccf1ecba09ab8463c75a",
}
CTX, ARTICLE_MAX = 2048, 1700
QUESTION = ("Based only on tangible economic facts reported in this market close article (economic data "
            "releases versus expectations, company earnings and guidance, central bank or government policy "
            "decisions, credit events), and ignoring market mood, fears, analyst opinions and the stock price "
            "moves themselves: did the economic fundamentals improve on this day? Answer Yes or No.")


def test_years(v):
    y = int(v[:4]) + 1
    return [2025, 2026] if v == "202412" else [y]


def check_code(repo, rev):
    info = HfApi().model_info(repo, revision=rev, files_metadata=True)
    blobs = {s.rfilename: (s.lfs.sha256 if s.lfs else s.blob_id)[:10] for s in info.siblings}
    bad = {k: blobs.get(k) for k, h in CODE_BLOBS.items() if blobs.get(k) != h}
    if bad:
        raise SystemExit(f"{repo}: code distant différent de la version relue {bad} — arrêt")


def drop_cache(repo):
    for r in scan_cache_dir().repos:
        if r.repo_id == repo:
            shutil.rmtree(r.repo_path)
            print(f"    cache supprimé : {repo} ({r.size_on_disk_str})", flush=True)


def embed_vintage(v, c):
    repo, rev = f"Diamegs/PIT-4B-FT-{v}", REVS[v]
    out = f"{OUT}/{v}.npz"
    if os.path.exists(out):
        print(f"{v} : déjà fait", flush=True)
        return
    ty = test_years(v)
    sub = c[c.index.year <= max(ty)]
    check_code(repo, rev)
    t0 = time.time()
    path = snapshot_download(repo, revision=rev)
    print(f"{v} : téléchargé en {time.time() - t0:.0f} s, {len(sub)} articles (test {ty})", flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                                              dtype=torch.bfloat16).to("cuda").eval()
    cap = {}
    model.lm_head.register_forward_pre_hook(lambda _m, a: cap.__setitem__("h", a[0].detach()))
    yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")})
    no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")})
    emb, pu = np.zeros((len(sub), 4096), np.float16), np.zeros(len(sub), np.float32)
    t1 = time.time()
    with torch.no_grad():
        for i, (_, r) in enumerate(sub.iterrows()):
            art = tok.encode(f"{r.title}\n\n{r.body}", add_special_tokens=False)[:ARTICLE_MAX]
            q = format_prompt("Article:\n" + tok.decode(art) + "\n\n" + QUESTION)
            x = torch.tensor([tok.encode(q, add_special_tokens=False)[-CTX:]], device="cuda")
            lg = model(input_ids=x, logits_to_keep=1, use_cache=False).logits[0, -1].float()
            p = torch.softmax(lg, -1)
            y_, n_ = float(p[yes].sum()), float(p[no].sum())
            emb[i] = cap["h"][0, -1].float().cpu().numpy().astype(np.float16)
            pu[i] = y_ / (y_ + n_)
            if (i + 1) % 500 == 0:
                print(f"    {v} {i + 1}/{len(sub)}  {(time.time() - t1) / (i + 1):.2f} s/article", flush=True)
    np.savez(out, dates=sub.index.values.astype("datetime64[D]"), emb=emb, p_up=pu, test_years=np.array(ty))
    print(f"{v} : {len(sub)} embeddings en {(time.time() - t1) / 60:.1f} min → {out}", flush=True)
    del model
    torch.cuda.empty_cache()
    if v not in KEEP:
        drop_cache(repo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    c = pd.read_parquet(CNBC)
    c["date"] = pd.to_datetime(c["date"])
    c = c.set_index("date").sort_index()
    order = ["201912"] + [v for v in REVS if v != "201912"]
    for v in ([a.only] if a.only else order):
        embed_vintage(v, c)


if __name__ == "__main__":
    main()
