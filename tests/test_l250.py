import pandas as pd
import pytest

from src import live
from src.symbols import liquidity_universe as lu


def test_sector_cap_fills_from_other_sectors(monkeypatch):
    monkeypatch.setattr(live, "sectors", lambda: {"A": "IT", "B": "IT", "C": "IT", "D": "Energy", "E": "IT"})
    scores = pd.DataFrame({"symbol": list("ABCDEF"), "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]})
    assert live.select_top(scores, 4)["symbol"].tolist() == ["A", "B", "C", "D"]
    assert live.select_top(scores, 4, sector_cap=2)["symbol"].tolist() == ["A", "B", "D", "F"]  # F: no sector, not capped


def daily(days, symbol, dollar_volume, price=20.0):
    return pd.DataFrame({"day": days, "ticker": symbol, "close": price, "volume": pd.Series(dollar_volume, index=range(len(days))) / price if isinstance(dollar_volume, list) else dollar_volume / price})


def test_liquidity_rules():
    days = pd.bdate_range("2026-01-01", periods=45)
    types = pd.Series({"BIG": "C", "NEW": "C", "SPY": "Q", "MID": "A", "CHEAP": "C"})
    late = [150e6] * 30 + [300e6] * 15  # >= $125M for 45 days, >= $250M only recently
    bars = pd.concat([daily(days, "BIG", 400e6), daily(days, "SPY", 900e6), daily(days, "MID", 220e6),
                      daily(days, "CHEAP", 400e6, price=3.0), daily(days, "NEW", late)])
    liquid, on_way = lu.select(bars, types, previous_liquid={"MID"})
    assert liquid == {"BIG", "NEW", "MID"}  # MID stays (>= $200M), SPY is an ETF, CHEAP is below $5
    liquid, _ = lu.select(bars, types, previous_liquid=set())
    assert "MID" not in liquid  # $220M is not enough to enter
    fresh = daily(days[-10:], "IPO", 600e6)  # only 10 sessions of history
    liquid, on_way = lu.select(pd.concat([bars, fresh]), pd.concat([types, pd.Series({"IPO": "C"})]), set())
    assert "IPO" not in liquid and "IPO" in on_way  # downloaded first, a candidate after 15 sessions


def test_shadow_failure_never_stops_the_main_prediction(monkeypatch):
    calls = []
    monkeypatch.setattr(live, "shadow_predict", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setitem(live.PROFILES, "fake_main", {**live.PROFILES["b1542"]})
    monkeypatch.setitem(live.PROFILES, "fake_shadow", {**live.PROFILES["b1542_l250"], "shadow_of": "fake_main"})
    top = pd.DataFrame({"symbol": ["A"], "rank": [1]})
    tops = {"fake_main": top}
    for name in [n for n, p in live.PROFILES.items() if p.get("shadow_of") == "fake_main"]:
        try:
            tops[name] = live.step(f"shadow model {name}", lambda: live.shadow_predict())
        except Exception:
            calls.append(name)
    assert calls == ["fake_shadow"] and list(tops) == ["fake_main"]


def test_stocks_with_missing_features_are_not_picked():
    rows = pd.DataFrame({"symbol": ["OLD", "NEW"], "f1": [1.0, 1.0], "f2": [1.0, None], "f3": [1.0, None]})
    assert live.enough_history(rows, ["f1", "f2", "f3"], {"min_feature_share": 0.9}) == ["OLD"]


def test_young_listings_wait_and_renames_keep_their_history(monkeypatch):
    from types import SimpleNamespace
    from src.data.raw import ticker_renames

    first = pd.DataFrame({"symbol": ["OLD", "SPIN", "NEWNAME", "OLDNAME"],
                          "first_day": pd.to_datetime(["2024-09-03", "2026-10-01", "2026-09-30", "2024-09-03"])})
    store = SimpleNamespace(exists=lambda path: True, read_parquet=lambda path: first)
    monkeypatch.setattr(lu, "storage", lambda: store)
    monkeypatch.setattr(ticker_renames, "load_renames", lambda: (SimpleNamespace(symbol="NEWNAME", old="OLDNAME"),))
    allowed = live.old_enough(["OLD", "SPIN", "NEWNAME", "NOINFO"], pd.Timestamp("2026-10-08"), {"min_listing_days": 20})
    assert allowed == ["OLD", "NEWNAME", "NOINFO"]  # SPIN waits; a renamed company keeps its history
