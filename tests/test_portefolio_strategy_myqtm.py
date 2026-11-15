"""Tests de la méthode d'origine de ref/MyQTM (src/portefolio_strategy/myqtm), restaurée depuis 73c6534.

Hors données : labels de tendance et features invariants quand on tronque l'avenir, fusion « dernier connu »,
moteur de trading (long seulement par défaut, short en option). Sur données (ignorés si data/portefolio_strategy/myqtm n'est pas construit) :
absence de fuite sur un vrai titre, prédictions toutes postérieures à leur date de coupe.

    pytest tests/test_portefolio_strategy_myqtm.py
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.portefolio_strategy import universe as U
from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm import trade as T
from src.portefolio_strategy.myqtm.trends import trend_labels


def synthetic_prices(n=900, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n) + 0.004 * np.sin(np.arange(n) / 25)))
    return pd.DataFrame({"o": c * (1 + rng.normal(0, 0.003, n)), "c": c,
                         "v": rng.integers(1_000_000, 5_000_000, n).astype(float)}, index=dates)


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
    r = T.simulate(T.build_trade_data(p), lambda d: PRM, capital=1200.0, max_positions=12, fee=0.0,
                   close_rule="proba")
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
    r = T.simulate(T.build_trade_data(p), lambda d: PRM, capital=1200.0, fee=0.001, close_rule="proba")
    assert len(r["history"]) == 1 and r["history"][0]["end"] == "2021-01-08"
    assert r["history"][0]["close_price"] == 10               # dernier cours connu
    assert r["values"][-1] == pytest.approx(1200 - 180 * 0.001 - 180 * 0.999 * 0.001)   # deux frais, prix plat


def test_short_position_gains_when_price_falls():
    p = _preds([("2021-01-04", "A", 10, .2, .6, .2), ("2021-01-05", "A", 10, .7, .2, .1),
                ("2021-01-06", "A", 10, .7, .2, .1), ("2021-01-07", "A", 8, .1, .8, .1),
                ("2021-01-08", "A", 8, .1, .8, .1)])
    assert T.simulate(T.build_trade_data(p), lambda d: PRM, capital=1200.0, fee=0.0)["history"] == []   # long seul
    r = T.simulate(T.build_trade_data(p), lambda d: PRM, capital=1200.0, fee=0.0, allow_short=True)
    assert r["history"][0]["type"] == "short" and r["history"][0]["gain"] == pytest.approx(20.0)


def test_metrics():
    d = pd.to_datetime(["2021-01-01", "2021-07-01", "2022-01-01"])
    m = T.metrics(d, [100, 80, 121])
    assert m["return_pct"] == pytest.approx(21.0) and m["max_drawdown_pct"] == pytest.approx(-20.0)
    assert m["cagr_pct"] == pytest.approx(21.0, abs=0.1)          # exactement un an


# --------------------------------------------------------------------------- sur données réelles
needs_data = pytest.mark.skipif(not U.UNIVERSE.exists() or not F.FEATURES.exists(),
                                reason="data/portefolio_strategy non construit (python -m src.portefolio_strategy.run)")


needs_data = pytest.mark.skipif(not U.UNIVERSE.exists() or not F.FEATURES.exists(),
                                reason="features non construites (python -m src.portefolio_strategy.myqtm.run features)")


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
def test_predictions_are_out_of_sample():
    from src.portefolio_strategy.myqtm.train import PREDICTIONS
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


def test_feature_variants_select_sentiment_families():
    cols = ["date", "symbol", "member", "o", "c", "rsi_14", "ts_news_sentiment_company_0", "news_days",
            "sentiment_vs_rate", "ts_vs_n_0", "vs_cat_earnings_n5", "vs_days"]
    df = pd.DataFrame(columns=cols)
    assert F.feature_columns(df, "base") == ["rsi_14"]
    assert F.feature_columns(df, "vendor") == ["rsi_14", "ts_vs_n_0", "vs_cat_earnings_n5", "vs_days"]
    assert F.feature_columns(df, "pit") == ["rsi_14", "ts_news_sentiment_company_0", "news_days", "sentiment_vs_rate"]
    assert len(F.feature_columns(df, "all")) == 7


def test_vendor_block_has_no_vader_sentiment():
    from src.portefolio_strategy.vendor_sentiment import CATEGORIES
    days = pd.to_datetime(["2021-01-04", "2021-01-06", "2021-01-11"])
    vs = pd.DataFrame({"day": days, "n": [2, 1, 3], "polarity": [0.5, -0.2, 0.1], "pos": 0.3, "neg": 0.1, "neu": 0.6,
                       "polarity_min": -0.2, "polarity_max": 0.5,
                       **{f"cat_{c}_n": [1, 0, 2] for c in CATEGORIES}, **{f"cat_{c}_pol": 0.1 for c in CATEGORIES}})
    out = F.vendor_block(vs)
    assert not [c for c in out.columns if any(k in c for k in ("polarity", "pos", "neg", "neu", "pol5"))]
    assert out["vs_n5"].tolist() == [2, 3, 6] and out["ts_vs_n_1"].tolist()[1:] == [2, 1]
    assert out["vs_cat_earnings_n5"].tolist() == [1, 1, 3]


def test_trend_phase_marks_head_of_segments():
    px = synthetic_prices()["c"]
    y, phase, _ = trend_labels(px, with_phase=True, **C.TREND)
    assert phase[y == 1].isna().all() and phase[y != 1].between(0, 1, inclusive="left").all()
    starts = phase[phase == 0]
    assert len(starts) > 5 and (y[starts.index] != 1).all()
    head = phase < C.F1_TREND_HEAD
    assert 0.1 < head[y != 1].mean() < 0.3                  # ≈ 20 % des jours de tendance gardés pour le F1


def test_long_closed_by_ema_rule_next_open():
    c = [10, 11, 12, 13, 14, 15, 16, 15, 13, 11, 9, 8]
    d = pd.bdate_range("2021-01-04", periods=len(c))
    p = pd.DataFrame({"date": d, "symbol": "A", "o": c, "c": c, "ybear": .1, "yrange": .2, "ybull": .7})
    p.loc[0, ["yrange", "ybull"]] = [.7, .2]                    # passe haussier le 2e jour
    td = T.build_trade_data(p)
    exit_days = [k for k, v in td.items() if v["A"]["ema_exit"]]
    first = exit_days[0]
    assert T.ema_exit(pd.Series(c, index=d)).loc[first] and pd.Timestamp(first) > d[6]
    r = T.simulate(td, lambda d: PRM, capital=1200.0, fee=0.0, close_rule="ema")
    h = r["history"][0]
    assert h["end"] == list(td)[list(td).index(first) + 1]        # exécuté à l'ouverture suivante
