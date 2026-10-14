from __future__ import annotations

import pandas as pd
import pytest

from src.broker import ibkr

STRATEGY = {"top_n": 3, "capital_per_day": 30000, "max_order_value": 15000, "order_ref": "test"}


def picks() -> pd.DataFrame:
    return pd.DataFrame(
        {"rank": [2, 1, 3, 4], "symbol": ["AAA", "BRK-B", "CCC", "DDD"], "close_cutoff": [100.0, 501.18, 1790.0, 50.0]}
    )


def test_plan_buys_equal_dollars_whole_shares_and_limit():
    plan = ibkr.plan_buys(picks(), STRATEGY, limit_buffer=0.003).set_index("symbol")
    assert list(plan.index) == ["BRK-B", "AAA", "CCC"]  # top 3 by rank
    assert plan.loc["AAA", "quantity"] == 100  # $10,000 / $100
    assert plan.loc["CCC", "quantity"] == 5  # floor(10,000 / 1,790)
    assert plan.loc["AAA", "limit_price"] == pytest.approx(100.30)
    assert plan.loc["BRK-B", "limit_price"] == pytest.approx(round(501.18 * 1.003, 2))


def test_plan_buys_caps_order_value():
    plan = ibkr.plan_buys(picks(), {**STRATEGY, "max_order_value": 5000}, limit_buffer=0.003).set_index("symbol")
    assert (plan["quantity"] * plan["limit_price"] <= 5000).all()


def test_ib_symbol_share_class():
    assert ibkr.ib_symbol("BRK-B") == "BRK B"
    assert ibkr.ib_symbol("AAPL") == "AAPL"


def test_open_lots_counts_only_sent_filled_orders():
    log = pd.DataFrame(
        {
            "side": ["BUY", "BUY", "SELL", "BUY"],
            "symbol": ["AAA", "BBB", "AAA", "CCC"],
            "filled_qty": [100, 50, 100, 30],
            "dry_run": [False, False, False, True],
        }
    )
    assert ibkr.open_lots(log).to_dict() == {"BBB": 50}


def test_connect_refuses_live_accounts(monkeypatch):
    class FakeIB:
        def connect(self, *args, **kwargs):
            pass

        def managedAccounts(self):
            return ["U1234567"]

        def disconnect(self):
            self.disconnected = True

    import ib_async

    monkeypatch.setattr(ib_async, "IB", FakeIB)
    config = {"connection": {"host": "127.0.0.1", "port": 4002, "client_id": 1, "timeout_seconds": 1, "paper_only": True}}
    with pytest.raises(RuntimeError, match="not all paper accounts"):
        ibkr.connect(config)


def fill_log() -> pd.DataFrame:
    rows = [
        ("2026-10-06", "BUY", "AAA", 100, 10.00, 1.00, False),
        ("2026-10-07", "SELL", "AAA", 100, 10.50, 1.00, False),   # +48 after commissions
        ("2026-10-07", "BUY", "BBB", 50, 20.00, 1.00, False),
        ("2026-10-08", "SELL", "BBB", 50, 19.00, 1.00, False),    # -52
    ]
    log = pd.DataFrame(rows, columns=["date", "side", "symbol", "filled_qty", "avg_fill_price", "commission", "dry_run"])
    log["date"] = pd.to_datetime(log["date"])
    return log


def capital_config(mode: str, since: str = "2026-10-06") -> dict:
    return {"strategy": {"capital": {"mode": mode, "amount": 50000, "since": since}}}


def test_compound_capital_adds_pnl_of_trades_sold_before_the_day():
    log = fill_log()
    assert ibkr.daily_capital(pd.Timestamp("2026-10-07"), capital_config("compound"), log)["capital"] == 50000  # nothing sold before 10/7
    assert ibkr.daily_capital(pd.Timestamp("2026-10-08"), capital_config("compound"), log)["capital"] == pytest.approx(50048)
    assert ibkr.daily_capital(pd.Timestamp("2026-10-09"), capital_config("compound"), log)["capital"] == pytest.approx(49996)
    # a reset: only trades bought on/after the new `since` count
    assert ibkr.daily_capital(pd.Timestamp("2026-10-09"), capital_config("compound", "2026-10-07"), log)["capital"] == pytest.approx(49948)
    assert ibkr.daily_capital(pd.Timestamp("2026-10-09"), capital_config("fixed"), log)["capital"] == 50000


def test_set_capital_rewrites_amount_and_since(tmp_path):
    path = tmp_path / "ibkr_config.yaml"
    path.write_text(ibkr.CONFIG_PATH.read_text())
    ibkr.set_capital(60000, pd.Timestamp("2026-11-02"), path)
    capital = ibkr.load_config(path)["strategy"]["capital"]
    assert capital["amount"] == 60000 and str(capital["since"]) == "2026-11-02"
    assert "# starting money" in path.read_text()  # comments kept


def test_sell_all_positions_sells_split_shares_and_spinoff_shares():
    import pandas as pd
    from src.broker import ibkr

    lots = pd.Series({"AAA": 100.0, "BRK-B": 10.0, "CCC": 50.0})
    held = {"AAA": 200.0, "BRK B": 10.0, "CCC": 30.0, "NEWCO": 33.0}  # AAA split 2:1, CCC partly filled, NEWCO spun off
    targets, notes = ibkr.sell_targets(lots, held, sell_all=True)
    assert targets == {"AAA": 200, "BRK-B": 10, "CCC": 30, "NEWCO": 33}
    assert set(notes) == {"AAA", "CCC", "NEWCO"} and "not bought by the strategy" in notes["NEWCO"]
    capped, _ = ibkr.sell_targets(lots, held, sell_all=False)
    assert capped == {"AAA": 100, "BRK-B": 10, "CCC": 30}  # old rule: the bought lots, capped at IBKR's position


def test_connect_refuses_another_account(monkeypatch):
    import sys, types
    import pytest
    from src.broker import ibkr

    class FakeIB:
        def connect(self, *a, **k): pass
        def managedAccounts(self): return ["DUX000001"]
        def disconnect(self): pass
    monkeypatch.setitem(sys.modules, "ib_async", types.SimpleNamespace(IB=FakeIB))
    config = {"connection": {"host": "h", "port": 1, "client_id": 1, "timeout_seconds": 1, "paper_only": True, "account": "DUR262794"}}
    with pytest.raises(RuntimeError, match="not in"):
        ibkr.connect(config)
