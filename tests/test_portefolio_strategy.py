"""Smoke tests de src/portefolio_strategy (sélection longue dans l'univers Nasdaq dynamique, data/eodhd).

Deux familles :
- **hors données** (toujours exécutés) : logique pure sur données synthétiques — règle de reconstitution de
  l'univers, datation des news, analyse du condensé Haiku, P(Yes) depuis des logits, features invariantes
  quand on tronque l'avenir, fusion « dernier connu », labels purgés, portefeuille et frais ;
- **sur données** (ignorés si data/portefolio_strategy n'est pas construit) : univers, absence de fuite
  temporelle sur un vrai titre, scores tous postérieurs à leur date de coupe.

Aucun appel payant ni GPU ici (Haiku : marqueur ``llm`` ; modèle PIT : marqueur ``pit``).

    pytest tests/test_portefolio_strategy.py
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.portefolio_strategy import config as C
from src.portefolio_strategy import digest as D
from src.portefolio_strategy import features as F
from src.portefolio_strategy import news as N
from src.portefolio_strategy import sentiment as S
from src.portefolio_strategy import labels as L
from src.portefolio_strategy import portfolio as PF
from src.portefolio_strategy import universe as U


def synthetic_prices(n=900, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n) + 0.004 * np.sin(np.arange(n) / 25)))
    return pd.DataFrame({"o": c * (1 + rng.normal(0, 0.003, n)), "c": c,
                         "v": rng.integers(1_000_000, 5_000_000, n).astype(float)}, index=dates)


# --------------------------------------------------------------------------- univers
def test_reconstitute_buffer_rule():
    ranked = [f"S{i}" for i in range(1, 151)]
    first = U.reconstitute(ranked, set())
    assert first == ranked[:100]
    # un membre classé 90e reste ; un non-membre classé 90e n'entre pas tant que la place est prise
    prev = set(ranked[:100]) - {"S90"} | {"S140"}
    new = U.reconstitute(ranked, prev)
    assert len(new) == 100 and "S140" not in new          # rang 140 > 100 : sorti
    assert "S90" in new                                     # complète la liste par le rang
    prev2 = set(ranked[10:110])                             # membres classés 101-110 : sortent, top 10 entre
    new2 = U.reconstitute(ranked, prev2)
    assert set(new2) == set(ranked[:100])


def test_effective_date_is_monday_after_third_friday():
    d = U.effective_date(2020)
    assert d == pd.Timestamp("2019-12-23") and d.dayofweek == 0


# --------------------------------------------------------------------------- news
def test_news_day_cutoff():
    assert N.news_day("2020-03-16T19:59:00+00:00") == pd.Timestamp("2020-03-16")
    assert N.news_day("2020-03-16T20:00:00+00:00") == pd.Timestamp("2020-03-17")   # après la clôture US
    assert N.news_day("2020-03-16T02:00:00+00:00") == pd.Timestamp("2020-03-16")


def test_news_select_dedup_and_cap():
    rows = [(pd.Timestamp("2021-01-04"), "AAA", "t", f"Title {i % 9}", "x" * 300, 300 + i, 1 + i % 3, "s")
            for i in range(20)]
    df = pd.DataFrame(rows, columns=["day", "symbol", "ts", "title", "content", "full_len", "n_symbols", "site"])
    out = N.select(df)
    assert len(out) == C.NEWS_MAX_ARTICLES and out["title"].is_unique
    assert out["n_symbols"].iloc[0] == 1                    # les plus ciblés d'abord


# --------------------------------------------------------------------------- Haiku (hors ligne)
def test_parse_digest():
    txt = "COMPANY: Revenue rose 12%.\nGuidance raised.\nCOMPETITORS: none\nGLOBAL: New tariffs announced."
    d = D.parse_digest(txt)
    assert d == {"company": "Revenue rose 12%. Guidance raised.", "competitors": "none",
                 "global": "New tariffs announced."}
    assert D.parse_digest("") == {"company": "none", "competitors": "none", "global": "none"}


def test_haiku_client_uses_anthropic_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API", "sk-test")
    c = D.make_client()
    assert type(c).__name__ == "Anthropic" and c.api_key == "sk-test" and C.HAIKU_MODEL == "claude-haiku-5-5"


def test_digest_prompt_has_no_judgement_request():
    assert "Do not judge" in D.SYSTEM and "Never mention what happened later" in D.SYSTEM
    p = D.user_prompt("Acme", "ACME", [("T1", "body1"), ("T2", "body2")])
    assert "Acme (ACME)" in p and "Article 2" in p


# --------------------------------------------------------------------------- LLM PIT (hors ligne)
def test_p_yes_from_logits_and_score():
    z = np.full(100, -10.0)
    z[[3, 4]] = [2.0, 1.0]        # Yes
    z[[7]] = [1.0]                # No
    p, mass = S.p_yes(z, [3, 4], [7])
    e = np.exp([2.0, 1.0, 1.0])
    assert p == pytest.approx((e[0] + e[1]) / e.sum(), rel=1e-6) and 0.99 < mass <= 1.0
    assert S.score(0.9, 0.1) == 9.0 and S.score(0.5, 0.5) == 5.0 and S.score(0.0, 1.0) == 0.0


def test_pit_prompt_is_single_closed_question():
    p = S.build_prompt("Acme Corp", "company", "Revenue rose.", "good")
    assert p.startswith("<|user|>\n") and p.endswith("<|assistant|>\n")
    assert "Answer with exactly one word, Yes or No." in p and "Acme Corp" in p


# --------------------------------------------------------------------------- pas de fuite temporelle
def test_forward_returns_and_purged_labels():
    dates = pd.bdate_range("2021-01-04", periods=8)
    o = pd.DataFrame({"A": [10, 10, 11, 12, 13, 14, 15, 16], "B": [5, 5, 5, 5, 5, np.nan, np.nan, np.nan],
                      "C": [20.0] * 8}, index=dates)
    f = L.forward_returns(o, 2)
    assert f["A"].iloc[0] == pytest.approx(12 / 10 - 1)            # ouverture J+1 → ouverture J+3
    assert f["A"].iloc[-3:].isna().all()                           # fenêtre au-delà des données : inconnu
    assert f["B"].iloc[2] == pytest.approx(0.0) and f["B"].iloc[3] == pytest.approx(0.0)   # radié : dernier cours
    assert np.isnan(f["B"].iloc[4])                                # plus coté le lendemain : pas de label
    df = o.stack().rename("o").rename_axis(["date", "symbol"]).reset_index().assign(member=True)
    lab = L.add_labels(df, [2])
    day0 = lab[lab["date"] == dates[0]].set_index("symbol")["y_2"]
    assert day0["A"] == pytest.approx(0.5) and day0.max() == day0["A"] and (lab["y_2"].dropna().abs() <= 0.5).all()
    # label du jour de rang pos connu à l'ouverture de pos+1+h : utilisable seulement si cette date est < coupe
    assert L.label_known(pd.Series([0, 1, 2]), 2, cutoff_pos=4).tolist() == [True, False, False]


def test_train_split_never_uses_labels_overlapping_the_cutoff():
    from src.portefolio_strategy import train as TR
    dates = pd.bdate_range("2021-01-04", periods=40)
    df = pd.DataFrame({"date": np.repeat(dates, 2), "symbol": ["A", "B"] * 40, "member": True,
                       "o": np.tile([10.0, 5.0], 40) + np.repeat(np.arange(40), 2)})
    df = L.add_labels(df, [5])
    cut, nxt = dates[30], dates[35]
    tr, te = TR.split(df, 5, cut, nxt)
    assert df.loc[tr, "date"].max() == dates[30 - 1 - 5 - 1]       # dernier label connu avant la coupe
    assert df.loc[te, "date"].min() == cut and df.loc[te, "date"].max() == dates[34]


def test_price_features_do_not_depend_on_future():
    px = synthetic_prices()
    full = F.price_features(px)
    part = F.price_features(px.iloc[:600])
    pd.testing.assert_frame_equal(part, full.loc[part.index])


def test_asof_join_never_reads_ahead():
    block = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.to_datetime(["2021-01-05", "2021-01-12"]))
    dates = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-11", "2021-01-12"])
    j = F.asof_join(dates, block, "days")
    assert np.isnan(j["x"].iloc[0]) and j["x"].tolist()[1:] == [1.0, 1.0, 2.0]
    assert j["days"].tolist() == [-1, 0, 6, 0]


def test_key_metrics_dated_by_filing_not_period_end():
    fund = {"Financials": {"Balance_Sheet": {"quarterly": {
        "2019-12-31": {"filing_date": "2020-01-29", "shortLongTermDebtTotal": "100", "totalStockholderEquity": "50",
                       "totalAssets": "400"},
        "2019-09-30": {"filing_date": "2019-10-31", "shortLongTermDebtTotal": "80", "totalStockholderEquity": "40",
                       "totalAssets": "400"}}}}}
    km = F.key_metrics(fund)
    assert list(km.index) == [pd.Timestamp("2019-11-01"), pd.Timestamp("2020-01-30")]
    j = F.asof_join(pd.to_datetime(["2020-01-29", "2020-01-30"]), km, "km_days")
    assert j["ts_km_debtToEquity_0"].tolist() == [2.0, 2.0]          # 80/40 puis 100/50
    assert j["ts_km_debtToAssets_0"].tolist() == [0.2, 0.25]         # le trimestre de décembre n'arrive que le 30/01


def test_earnings_surprise_available_day_after_report():
    fund = {"Earnings": {"History": {
        "2020-03-31": {"reportDate": "2020-04-30", "epsActual": 1.1, "epsEstimate": 1.0, "surprisePercent": 10.0},
        "2030-03-31": {"reportDate": "2030-04-30", "epsActual": None, "epsEstimate": 2.0}}}}
    es = F.earnings_surprises(fund, today="2021-01-01")
    assert list(es.index) == [pd.Timestamp("2020-05-01")] and es["ts_es_surprise_0"].iloc[0] == 10.0


# --------------------------------------------------------------------------- portefeuille
def _frames(scores, opens):
    d = pd.bdate_range("2021-01-04", periods=len(opens))
    return pd.DataFrame(scores, index=d, columns=list("ABC")), pd.DataFrame(opens, index=d, columns=list("ABC"))


def test_select_top_n_with_buffer_and_eligibility():
    score = np.array([0.9, 0.5, 0.4, np.nan, 0.1])
    none = np.zeros(5, bool)
    assert PF.select(score, none, 2).tolist() == [0, 1]
    held = np.array([False, False, True, True, False])              # C détenu (rang 3), D détenu mais non éligible
    assert PF.select(score, held, 2, buffer=1.0).tolist() == [0, 1]
    assert PF.select(score, held, 2, buffer=2.0).tolist() == [2, 0]     # C gardé grâce au buffer, complété par A
    assert len(PF.select(score, none, None)) == 4                   # n_top None : tous les éligibles


def test_simulate_trades_next_open_with_fee_on_traded_amount():
    s, o = _frames([[3, 2, 1]] * 2 + [[1, 2, 3]] * 3, [[10, 10, 10], [10, 10, 10], [11, 10, 10], [12.1, 10, 9], [12.1, 10, 9]])
    r = PF.simulate(s, o, n_top=1, every=2, fee=0.01)
    v = r["values"]
    assert v.iloc[0] == 1.0 and v.iloc[1] == pytest.approx(0.99)        # achat de A à l'ouverture de J+1, 1 % de frais
    assert v.iloc[2] == pytest.approx(0.99 * 1.1)                       # A : 10 → 11
    # le soir de J2 le classement s'inverse (C en tête) : vente de A et achat de C à l'ouverture de J3
    assert v.iloc[3] == pytest.approx(0.99 * 1.1 * 1.1 * (1 - 0.02), rel=1e-9)
    assert r["turnover"].iloc[1] == pytest.approx(1.0) and r["turnover"].iloc[3] == pytest.approx(2.0)
    assert r["n"].tolist() == [0, 1, 1, 1, 1]
    free = PF.simulate(s, o, n_top=1, every=2, fee=0.0)["values"]
    assert free.iloc[-1] == pytest.approx(1.1 * 1.1) and (free >= v).all()


def test_simulate_keeps_last_price_of_unquoted_stock_and_equal_weights():
    s, o = _frames([[3, 2, 1]] * 4, [[10, 10, 10], [10, 10, 10], [np.nan, 12, 10], [np.nan, 12, 10]])
    r = PF.simulate(s, o, n_top=2, every=10, fee=0.0, record=True)
    assert r["weights"][0].to_dict() == {"A": 0.5, "B": 0.5}
    assert r["values"].iloc[-1] == pytest.approx(0.5 + 0.5 * 1.2)       # A figé à son dernier cours


def test_simulate_cap_weights_follow_size():
    s, o = _frames([[3, 2, 1]] * 3, [[10, 10, 10]] * 3)
    size = pd.DataFrame([[300.0, 100.0, 50.0]] * 3, index=s.index, columns=s.columns)
    r = PF.simulate(s, o, n_top=2, every=10, fee=0.0, record=True, size=size)
    assert r["weights"][0].to_dict() == pytest.approx({"A": 0.75, "B": 0.25})


def test_metrics_and_active():
    d = pd.to_datetime(["2021-01-01", "2021-07-01", "2022-01-01"])
    m = PF.metrics(pd.Series([100.0, 80.0, 121.0], index=d))
    assert m["return_pct"] == pytest.approx(21.0) and m["max_drawdown_pct"] == pytest.approx(-20.0)
    assert m["cagr_pct"] == pytest.approx(21.0, abs=0.1)          # exactement un an
    a = PF.active(pd.Series([100.0, 80.0, 121.0], index=d), pd.Series([100.0, 100.0, 110.0], index=d))
    assert a["excess_cagr_pct"] == pytest.approx(11.0, abs=0.1)


def test_walk_forward_choice_uses_only_past_years():
    from src.portefolio_strategy import benchmark as B
    d = pd.bdate_range("2021-01-04", "2023-12-29")
    bench = pd.Series(1.0, index=d)
    up = pd.Series(np.cumprod(np.r_[1.0, np.full(len(d) - 1, 1.001)]), index=d)
    default = tuple(C.DEFAULT_CONFIG[k] for k in B.KEYS)
    late = (5, 5, 10, 1.0, "equal")                                   # ne gagne qu'à partir de 2023
    lv = pd.Series(np.where(d.year < 2023, 1.0, 0.0), index=d)
    late_curve = pd.Series(np.cumprod(1 + 0.01 * lv.to_numpy() + np.where(d.year < 2023, -0.0005, 0)), index=d)
    strat, by_year = B.walk_forward({default: up, late: late_curve}, bench, fee=0.0)
    assert by_year[2021]["source"].startswith("défaut")
    assert tuple(by_year[2023][k] for k in B.KEYS) == default           # 2023 choisi sur 2021-2022 seulement
    assert strat.iloc[-1] == pytest.approx(up.iloc[-1], rel=1e-6)


# --------------------------------------------------------------------------- sur données réelles
needs_data = pytest.mark.skipif(not U.UNIVERSE.exists() or not F.FEATURES.exists(),
                                reason="data/portefolio_strategy non construit (python -m src.portefolio_strategy.run)")


@needs_data
def test_universe_is_dynamic_and_point_in_time():
    u = U.load_universe()
    for year in range(2020, 2027):
        m = U.members_on(u, f"{year}-02-01")
        assert len(m) == C.UNIVERSE_SIZE, year
        assert {"AAPL", "MSFT", "AMZN"} <= set(m)
    assert set(U.members_on(u, "2020-02-01")) != set(U.members_on(u, "2026-02-01"))     # la liste change
    caps = pd.read_parquet(U.CAPS)
    # reconstruire l'univers en s'arrêtant fin 2021 donne les mêmes membres en 2021 : pas d'information future
    past = U.build_universe(caps[caps["month"] <= "2021-11"], last_month="2021-11")
    assert U.members_on(past, "2021-06-01") == U.members_on(u, "2021-06-01")


@needs_data
def test_real_stock_features_do_not_depend_on_future():
    from src.portefolio_strategy import eod
    u = U.load_universe()
    index = pd.read_parquet(F.INDEX)
    cut = pd.Timestamp("2022-06-30")
    px, fund = eod.load_eod("AAPL"), eod.load_fund("AAPL")
    member = lambda dates: U.member_mask(u, "AAPL", dates)  # noqa: E731
    econ = F.econ_blocks()
    econ_cut = {k: b[b.index <= cut] for k, b in econ.items()}
    full = F.stock_features("AAPL", px, fund, F.market_features(index), econ, None, member)
    part = F.stock_features("AAPL", px[px.index <= cut], fund, F.market_features(index[index.index <= cut]),
                            econ_cut, None, member)
    a = full[full["date"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(part.reset_index(drop=True), a, check_dtype=False)


@needs_data
def test_features_only_news_after_pit_cutoff():
    assert pd.Timestamp(C.START_DATE) >= pd.Timestamp("2020-01-01")          # coupure du LLM PIT : 2019-12
    files = sorted(N.NEWS_DIR.glob("*.parquet"))
    if not files:
        pytest.skip("news non extraites")
    assert pd.read_parquet(files[0], columns=["day"])["day"].min() >= pd.Timestamp(C.START_DATE)


@needs_data
def test_predictions_are_out_of_sample():
    from src.portefolio_strategy.train import out_dir, predictions_path
    variant = next((v for v in F.VARIANTS if predictions_path(v).exists() and (out_dir(v) / "train_report.json").exists()), None)
    if variant is None:
        pytest.skip("entraînement non lancé")
    p = pd.read_parquet(predictions_path(variant))
    assert p["date"].min() >= pd.Timestamp(C.FIRST_OOS_DATE)
    assert not p.duplicated(["date", "symbol"]).any()
    assert p[[f"s_{h}" for h in C.HORIZONS]].notna().all().all()
    for c in json.loads((out_dir(variant) / "train_report.json").read_text())["cutoffs"]:
        assert all(c[f"h{h}"]["train_until"] < c["cutoff"] for h in C.HORIZONS)
    u = U.load_universe()                                    # on ne note que des membres de l'univers du jour
    s = p.sample(200, random_state=0)
    assert all(sym in U.members_on(u, d) for d, sym in zip(s["date"], s["symbol"]))


@needs_data
def test_feature_table_has_no_label_and_ranks_only_members():
    df = pd.read_parquet(F.FEATURES, columns=["member", "xs_ret_21"])
    assert df.loc[~df["member"], "xs_ret_21"].isna().all() and df.loc[df["member"], "xs_ret_21"].between(0, 1).all()
    import pyarrow.parquet as pq
    names = pq.ParquetFile(F.FEATURES).schema.names
    assert not [c for c in names if c.startswith(("fwd_", "y_", "s_"))]


# --------------------------------------------------------------------------- sentiment fournisseur + catégories
def test_vendor_categories_from_tags_and_title():
    from src.portefolio_strategy import vendor_sentiment as V
    assert V.categories(["EARNINGS", "INC"], "Acme beats estimates") == ["earnings", "analyst"]
    assert V.categories([], "FedEx launches new service") == ["product"]            # « FED » ne matche pas FedEx
    assert "macro" in V.categories(["INFLATION"], "Stocks slide") and V.categories([], "Hello world") == []
    assert "legal" in V.categories([], "Class action filed against Acme")           # titre seul (tags vides en 2024)


def test_vendor_aggregate_and_block_use_only_past_news_days():
    from src.portefolio_strategy import vendor_sentiment as V
    rows = []
    for i, (day, pol, earn) in enumerate([("2021-01-04", 0.8, True), ("2021-01-04", 0.2, False),
                                          ("2021-01-06", -0.5, True), ("2021-01-11", 0.1, False)]):
        r = {"day": pd.Timestamp(day), "symbol": "A", "key": f"k{i}", "polarity": pol, "pos": 0.1, "neg": 0.05, "neu": 0.85}
        r.update({c: (c == "earnings" and earn) for c in V.CATEGORIES})
        rows.append(r)
    agg = V.aggregate(pd.DataFrame(rows))
    assert agg["n"].tolist() == [2, 1, 1] and agg["polarity"].tolist() == pytest.approx([0.5, -0.5, 0.1])
    assert agg["cat_earnings_n"].tolist() == [1, 1, 0] and agg["cat_earnings_pol"].iloc[0] == pytest.approx(0.8)
    vb = F.vendor_block(agg)
    assert vb["ts_vs_polarity_0"].tolist() == pytest.approx([0.5, -0.5, 0.1])
    assert vb["ts_vs_polarity_1"].tolist()[1:] == pytest.approx([0.5, -0.5])        # jour de news précédent
    assert vb["vs_cat_earnings_pol5"].tolist() == pytest.approx([0.8, 0.15, 0.15])  # moyenne des articles passés
    assert vb["vs_pol5"].iloc[1] == pytest.approx((0.5 * 2 - 0.5) / 3, abs=1e-4)
    # tronquer l'avenir ne change pas le passé
    pd.testing.assert_frame_equal(F.vendor_block(agg.iloc[:2]), vb.iloc[:2])
    j = F.asof_join(pd.to_datetime(["2021-01-05", "2021-01-06"]), vb, "vs_days")
    assert j["ts_vs_polarity_0"].tolist() == pytest.approx([0.5, -0.5]) and j["vs_days"].tolist() == [1, 0]


def test_feature_variants_select_sentiment_families():
    cols = ["date", "symbol", "member", "o", "c", "rsi_14", "pit_pos_company", "xs_pit_pos_company_5", "news_days",
            "qt_earnings_5", "qs_pos_company", "ts_vs_polarity_0", "vs_cat_earnings_n5", "vs_days"]
    df = pd.DataFrame(columns=cols)
    assert F.feature_columns(df, "base") == ["rsi_14"]
    assert F.feature_columns(df, "vendor") == ["rsi_14", "ts_vs_polarity_0", "vs_cat_earnings_n5", "vs_days"]
    assert F.feature_columns(df, "pit") == ["rsi_14", "pit_pos_company", "xs_pit_pos_company_5", "news_days"]
    assert F.feature_columns(df, "pit_types") == F.feature_columns(df, "pit") + ["qt_earnings_5"]
    assert F.feature_columns(df, "qsent") == ["rsi_14", "qt_earnings_5", "qs_pos_company"]
    assert len(F.feature_columns(df, "all")) == 8                       # tout sauf le sentiment Qwen (non point-in-time)


def test_prob_block_uses_raw_probabilities_of_past_news_days_only():
    sent = pd.DataFrame({"day": pd.to_datetime(["2021-01-04", "2021-01-06", "2021-01-11"]), "symbol": "A",
                         **{f"p_{pol}_{cat}": [0.2, 0.6, 1.0] if cat == "company" else [np.nan] * 3
                            for cat in C.SENT_CATEGORIES for pol in ("pos", "neg")}})
    b = F.prob_block(sent, "pit_")
    assert b["pit_pos_company"].tolist() == [0.2, 0.6, 1.0]                     # P(Yes) brute, pas de note 0-10
    assert b["pit_pos_company_5"].tolist() == pytest.approx([0.2, 0.4, 0.6])
    assert b["pit_pos_global_5"].isna().all()                                   # rubrique vide : pas de valeur inventée
    pd.testing.assert_frame_equal(F.prob_block(sent.iloc[:2], "pit_"), b.iloc[:2])
    j = F.asof_join(pd.to_datetime(["2021-01-05", "2021-01-06"]), b, "news_days")
    assert j["pit_pos_company"].tolist() == [0.2, 0.6] and j["news_days"].tolist() == [1, 0]


def test_fundamentals_ttm_dated_by_filing():
    q = {f"20{y}-{m}": {"filing_date": f"20{y + (m == '12-31')}-{f}", "totalRevenue": "100", "netIncome": "10",
                        "operatingIncome": "20", "grossProfit": "50"}
         for y in (19, 20) for m, f in (("03-31", "05-01"), ("06-30", "08-01"), ("09-30", "11-01"), ("12-31", "02-01"))}
    t = F.fundamentals_ttm({"Financials": {"Income_Statement": {"quarterly": q}}})
    assert t.index[0] == pd.Timestamp("2020-02-02")                 # 4 trimestres connus le lendemain du 4e dépôt
    assert t["f_rev_ttm"].iloc[0] == 400 and t["op_margin"].iloc[0] == pytest.approx(0.2)
    assert np.isnan(t["sales_growth"].iloc[0]) and t["sales_growth"].iloc[-1] == pytest.approx(0.0)


# --------------------------------------------------------------------------- stockage LLM commun (PIT, Qwen)
def test_llm_outputs_share_names_across_models():
    from src.portefolio_strategy import qwen as Q
    assert D.DIGESTS == C.LLM / "haiku" / "digests.jsonl"
    assert (S.SENTIMENTS, S.EMB_DIR) == (C.LLM / "pit" / "sentiment.parquet", C.LLM / "pit" / "sentiment_emb")
    assert (Q.SENTIMENTS, Q.EMB_DIR) == (C.LLM / "qwen25" / "sentiment.parquet", C.LLM / "qwen25" / "sentiment_emb")
    assert Q.CATEGORIES == C.LLM / "qwen25" / "categories.parquet"


def test_score_rows_and_embedding_parts(tmp_path):
    part = [{"day": "2020-01-02", "symbol": "AAA"}, {"day": "2020-01-02", "symbol": "BBB"}]
    where = [(0, "company", "pos"), (0, "company", "neg")]
    rows = S.score_rows(part, where, [(0.9, 1.0), (0.1, 0.8)])
    assert rows[0]["company"] == 9.0 and rows[0]["global"] == 5.0 and rows[0]["mass_company"] == 0.8
    assert rows[1]["company"] == 5.0 and rows[1]["p_pos_company"] is None        # rubrique vide : neutre
    S.save_emb(tmp_path, np.ones((2, 4)), part, where)
    S.save_emb(tmp_path, np.zeros((1, 4)), part, [(1, "global", "pos")])
    assert sorted(f.name for f in tmp_path.iterdir()) == ["part_00000.npy", "part_00000.parquet",
                                                          "part_00001.npy", "part_00001.parquet"]
    ix = pd.read_parquet(tmp_path / "part_00000.parquet")
    assert list(ix.columns) == ["day", "symbol", "category", "polarity", "row"] and list(ix["polarity"]) == ["pos", "neg"]
    assert np.load(tmp_path / "part_00000.npy").dtype == np.float16
    # reprise après interruption : le lot dont les notes n'ont pas été sauvées (BBB) est supprimé
    S.prune_orphans(tmp_path, pd.DataFrame({"day": [pd.Timestamp("2020-01-02")], "symbol": ["AAA"]}))
    assert sorted(f.name for f in tmp_path.iterdir()) == ["part_00000.npy", "part_00000.parquet"]


def test_qwen_questions():
    from src.portefolio_strategy import qwen as Q
    comp, glob = Q.questions("company"), Q.questions("global")
    assert [(k, key) for k, key, _ in glob] == [("sent", "pos"), ("sent", "neg")]
    assert [key for k, key, _ in comp if k == "type"] == list(C.NEWS_TYPES)       # types sur company seulement
    assert all(q.endswith("Yes or No.") for _, _, q in comp)
