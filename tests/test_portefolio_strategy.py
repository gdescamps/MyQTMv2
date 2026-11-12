"""Smoke tests du portage de ref/MyQTM sur data/eodhd (src/portefolio_strategy).

Deux familles :
- **hors données** (toujours exécutés) : logique pure sur données synthétiques — règle de reconstitution de
  l'univers, datation des news, analyse du condensé Haiku, P(Yes) depuis des logits, labels de tendance et
  features invariants quand on tronque l'avenir, fusion « dernier connu », moteur de trading ;
- **sur données** (ignorés si data/portefolio_strategy n'est pas construit) : univers, absence de fuite
  temporelle sur un vrai titre, prédictions toutes postérieures à leur date de coupe.

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
from src.portefolio_strategy import trade as T
from src.portefolio_strategy import universe as U
from src.portefolio_strategy.trends import trend_labels


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
def test_trend_labels_are_final_before_final_until():
    px = synthetic_prices()["c"]
    full, _ = trend_labels(px, **C.TREND)
    assert set(full.unique()) == {0, 1, 2}
    for cut in (300, 500, 777):
        part, final_until = trend_labels(px.iloc[:cut], **C.TREND)
        ok = part.index < final_until
        assert ok.sum() > cut - 60                          # on ne perd que les derniers jours
        assert (part[ok] == full.reindex(part.index)[ok]).all()


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


# --------------------------------------------------------------------------- moteur de trading
def _preds(rows):
    return pd.DataFrame(rows, columns=["date", "symbol", "o", "ybear", "yrange", "ybull"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


PRM = dict(long_open=0.6, long_close=0.4, short_open=0.6, short_close=0.4, increase=0.7)


def test_build_trade_data_index_long():
    p = _preds([("2021-01-04", "A", 10, .2, .6, .2), ("2021-01-05", "A", 10, .1, .2, .7),
                ("2021-01-06", "A", 11, .1, .2, .7), ("2021-01-07", "A", 12, .2, .6, .2)])
    td = T.build_trade_data(p)
    assert [td[d]["A"]["index_long"] for d in td] == [-1, 0, 1, 2]


def test_simulate_opens_next_open_and_closes_on_signal():
    p = _preds([("2021-01-04", "A", 10, .2, .6, .2), ("2021-01-05", "A", 10, .1, .2, .7),   # signal le 05 au soir
                ("2021-01-06", "A", 11, .1, .2, .7),                                        # achat à 11
                ("2021-01-07", "A", 12, .1, .6, .3),                                        # 0.3 < 0.4 → vente demain
                ("2021-01-08", "A", 13.2, .1, .6, .3), ("2021-01-11", "A", 14, .1, .6, .3)])
    r = T.simulate(T.build_trade_data(p), lambda d: PRM, capital=1200.0, max_positions=12, fee=0.0)
    h = r["history"]
    assert len(h) == 1 and h[0]["start"] == "2021-01-06" and h[0]["open_price"] == 11
    assert h[0]["end"] == "2021-01-08" and h[0]["close_price"] == 13.2 and h[0]["gain"] == pytest.approx(20.0)
    size = 0.5 * (1 - 0.7) * 12 * (1200 / 12)                 # facteur 1.8 × 100 $
    assert h[0]["size"] == pytest.approx(size)
    assert r["values"][-1] == pytest.approx(1200 + size * 0.2)
    assert r["values"][1] == 1200.0                            # rien n'est ouvert le jour du signal


def test_simulate_fee_and_missing_ticker_forces_close():
    p = _preds([("2021-01-04", "A", 10, .2, .6, .2), ("2021-01-05", "A", 10, .1, .2, .7),
                ("2021-01-06", "A", 10, .1, .2, .7), ("2021-01-04", "B", 5, .2, .6, .2),
                ("2021-01-05", "B", 5, .2, .6, .2), ("2021-01-06", "B", 5, .2, .6, .2),
                ("2021-01-07", "B", 5, .2, .6, .2), ("2021-01-08", "B", 5, .2, .6, .2)])   # A disparaît le 07
    r = T.simulate(T.build_trade_data(p), lambda d: PRM, capital=1200.0, fee=0.001)
    assert len(r["history"]) == 1 and r["history"][0]["end"] == "2021-01-08"
    assert r["history"][0]["close_price"] == 10               # dernier cours connu
    assert r["values"][-1] == pytest.approx(1200 - 180 * 0.001 - 180 * 0.999 * 0.001)   # deux frais, prix plat


def test_short_position_gains_when_price_falls():
    p = _preds([("2021-01-04", "A", 10, .2, .6, .2), ("2021-01-05", "A", 10, .7, .2, .1),
                ("2021-01-06", "A", 10, .7, .2, .1), ("2021-01-07", "A", 8, .1, .8, .1),
                ("2021-01-08", "A", 8, .1, .8, .1)])
    r = T.simulate(T.build_trade_data(p), lambda d: PRM, capital=1200.0, fee=0.0)
    assert r["history"][0]["type"] == "short" and r["history"][0]["gain"] == pytest.approx(20.0)


def test_metrics():
    d = pd.to_datetime(["2021-01-01", "2021-07-01", "2022-01-01"])
    m = T.metrics(d, [100, 80, 121])
    assert m["return_pct"] == pytest.approx(21.0) and m["max_drawdown_pct"] == pytest.approx(-20.0)
    assert m["cagr_pct"] == pytest.approx(21.0, abs=0.1)          # exactement un an


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
    from src.portefolio_strategy.train import PREDICTIONS
    rep = C.OUT / "train_report.json"
    if not PREDICTIONS.exists() or not rep.exists():
        pytest.skip("entraînement non lancé")
    p = pd.read_parquet(PREDICTIONS)
    assert p["date"].min() >= pd.Timestamp(C.FIRST_OOS_DATE)
    assert not p.duplicated(["date", "symbol"]).any()
    assert np.allclose(p[["ybear", "yrange", "ybull"]].sum(axis=1), 1.0, atol=1e-4)
    for c in json.loads(rep.read_text())["cutoffs"]:
        assert c["train_until"] < c["cutoff"]
    u = U.load_universe()                                    # on ne prédit que des membres de l'univers du jour
    s = p.sample(200, random_state=0)
    assert all(sym in U.members_on(u, d) for d, sym in zip(s["date"], s["symbol"]))
