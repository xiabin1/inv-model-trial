from __future__ import annotations

import pandas as pd
import pytest

from src.data.raw.corporate_actions import adjust_bars, find_candidates, nearest_split, parse_actions


def daily(symbol: str, closes: list[float], start: str = "2025-11-12") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    frame = pd.DataFrame(
        {
            "datetime_utc": dates.tz_localize("UTC"),
            "symbol": symbol,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1000.0,
        }
    )
    frame["vwap"] = frame["close"]
    frame["vwap_volume"] = frame["vwap"] * frame["volume"]
    return frame


def test_parse_split_and_factor():
    actions = parse_actions(
        [
            {"symbol": "nflx", "date": "2025-11-17", "split": 10},
            {"symbol": "AMCR", "date": "2026-01-15", "split": 0.2, "verified": True, "source": "OCC"},
            {"symbol": "DD", "date": "2025-11-03", "factor": 0.45},
        ]
    )
    assert [a.symbol for a in actions] == ["NFLX", "AMCR", "DD"]
    assert actions[0].factor == pytest.approx(0.1) and actions[0].kind == "split"
    assert actions[1].factor == pytest.approx(5) and actions[1].kind == "reverse_split" and actions[1].verified
    assert actions[2].factor == 0.45


@pytest.mark.parametrize(
    "entry",
    [{"symbol": "X", "date": "2025-01-02"}, {"symbol": "X", "date": "2025-01-02", "split": 2, "factor": 0.5},
     {"symbol": "X", "split": 2}, {"symbol": "X", "date": "2025-01-02", "split": -2}],
)
def test_parse_rejects_bad_entries(entry):
    with pytest.raises(ValueError):
        parse_actions([entry])


def test_adjust_scales_only_bars_before_the_action():
    bars = pd.concat([daily("NFLX", [1100, 1110, 1115, 112, 113]), daily("AAPL", [200] * 5)])
    actions = parse_actions([{"symbol": "NFLX", "date": "2025-11-17", "split": 10}])
    out = adjust_bars(bars, actions)
    nflx = out[out["symbol"] == "NFLX"]
    assert nflx["close"].tolist() == pytest.approx([110, 111, 111.5, 112, 113])
    assert nflx["volume"].tolist() == pytest.approx([10000, 10000, 10000, 1000, 1000])
    assert (nflx["vwap"] * nflx["volume"]).tolist() == pytest.approx(nflx["vwap_volume"].tolist())
    assert out[out["symbol"] == "AAPL"]["close"].tolist() == [200] * 5  # other symbols untouched
    assert bars[bars["symbol"] == "NFLX"]["close"].iloc[0] == 1100  # input not modified


def test_adjust_minute_bars_by_session_day():
    times = pd.to_datetime(["2025-11-14 20:59", "2025-11-17 14:30"]).tz_localize("UTC")  # Fri 15:59, Mon 09:30 ET
    bars = pd.DataFrame({"datetime_utc": times, "symbol": "NFLX", "open": [1115.0, 112.0], "high": [1115.0, 112.0],
                         "low": [1115.0, 112.0], "close": [1115.0, 112.0], "volume": [10.0, 100.0], "vwap": [1115.0, 112.0]})
    out = adjust_bars(bars, parse_actions([{"symbol": "NFLX", "date": "2025-11-17", "split": 10}]))
    assert out["close"].tolist() == pytest.approx([111.5, 112.0])


def test_two_actions_compound():
    bars = daily("DD", [30, 30, 90, 90, 90], start="2026-06-22")
    actions = parse_actions([{"symbol": "DD", "date": "2026-06-24", "split": 1 / 3},
                             {"symbol": "DD", "date": "2026-06-23", "factor": 0.5}])
    assert adjust_bars(bars, actions)["close"].tolist() == pytest.approx([45, 90, 90, 90, 90])


def test_nearest_split():
    assert nearest_split(0.1007) == 10
    assert nearest_split(0.0405) == 25
    assert nearest_split(5.0522) == pytest.approx(0.2)
    assert nearest_split(0.97) is None
    assert nearest_split(0.4065) is None


def test_find_candidates_flags_splits_and_large_moves_only():
    bars = pd.concat([daily("NFLX", [1100, 1115, 112, 113]), daily("XYZ", [100, 101, 55, 56]), daily("AAPL", [200, 204, 198, 201])])
    found = find_candidates(bars)
    assert set(found["symbol"]) == {"NFLX", "XYZ"}
    assert found.loc[found["symbol"] == "NFLX", "suggested_split"].item() == 10
    assert pd.isna(found.loc[found["symbol"] == "XYZ", "suggested_split"].item())
    adjusted = adjust_bars(bars, parse_actions([{"symbol": "NFLX", "date": "2025-11-14", "split": 10}]))
    assert "NFLX" not in set(find_candidates(adjusted)["symbol"])


def test_spin_off_factor_scales_prices_not_volume():
    bars = daily("CTVA", [77.67, 77.67, 12.57], start="2026-09-29")
    out = adjust_bars(bars, parse_actions([{"symbol": "CTVA", "date": "2026-10-01", "factor": 0.15642}]))
    assert out["close"].tolist() == pytest.approx([77.67 * 0.15642, 77.67 * 0.15642, 12.57])
    assert out["volume"].tolist() == [1000.0, 1000.0, 1000.0]  # same share count before and after
    assert (out["vwap"] * out["volume"]).tolist() == pytest.approx(out["vwap_volume"].tolist())


def test_gap_guard_excludes_split_like_and_large_gaps(tmp_path, monkeypatch):
    from src import live

    monkeypatch.setattr(live, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(live, "propose_actions", lambda *args, **kwargs: None)  # no real proposals / Telegram
    combined = pd.DataFrame({"symbol": ["CTVA", "AAA", "BBB", "CCC", "DDD"],
                             "min_ret_gap": [-0.814, 0.02, -0.50, 0.45, -0.13]})  # spin-off, normal, 1:2-like, +45%, -13%
    found = live.gap_guard(pd.Timestamp("2026-10-01"), combined)
    assert sorted(found["symbol"]) == ["BBB", "CCC", "CTVA"]
    assert "CTVA 2026-10-01" in (tmp_path / "logs/corporate_action_alerts.log").read_text()


def test_spinoff_factor_uses_ex_date_opens_so_the_overnight_move_is_the_holders_return():
    entry = {"symbol": "CTVA", "date": "2026-10-01", "spinoff": {"child": "VYLR", "ratio": 1, "parent_open": 14.44, "child_open": 66.00}}
    action = parse_actions([entry])[0]
    assert action.kind == "spin_off" and action.factor == pytest.approx(14.44 / 80.44)
    bars = daily("CTVA", [77.67, 14.44], start="2026-09-30")  # 9/30 close, 10/1 open (= close here)
    adjusted = adjust_bars(bars, [action])["close"].tolist()
    holder_return = (14.44 + 66.00) / 77.67 - 1  # parent + child at the ex-date open
    assert adjusted[1] / adjusted[0] - 1 == pytest.approx(holder_return)


def test_spinoff_requires_its_inputs():
    with pytest.raises(ValueError, match="spinoff needs"):
        parse_actions([{"symbol": "X", "date": "2026-01-02", "spinoff": {"child": "Y", "ratio": 1}}])


def test_repo_spinoff_factors():
    from src.data.raw.corporate_actions import load_actions

    factors = {(a.symbol, f"{a.date:%Y-%m-%d}"): a.factor for a in load_actions() if a.kind == "spin_off"}
    assert len(factors) >= 12  # grows as spin-offs are verified
    assert factors[("BDX", "2026-02-10")] == pytest.approx(0.78871, abs=1e-5)
    assert factors[("CTVA", "2026-10-01")] == pytest.approx(0.17951, abs=1e-5)
    assert factors[("DD", "2025-11-03")] == pytest.approx(0.40738, abs=1e-5)


def test_verified_spinoff_without_opens_is_not_applied_yet(tmp_path):
    from src.data.raw.corporate_actions import load_actions

    path = tmp_path / "actions.yaml"
    path.write_text("actions:\n"
                    "  - {symbol: AAA, date: 2026-01-05, spinoff: {child: BBB, ratio: 0.5}, verified: true}\n"
                    "  - {symbol: CCC, date: 2026-01-05, split: 2, verified: true}\n")
    load_actions.cache_clear()
    assert [a.symbol for a in load_actions(path)] == ["CCC"]  # no crash; AAA waits for its opens
    load_actions.cache_clear()
