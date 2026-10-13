from __future__ import annotations

import pandas as pd
import pytest

from src.notify import telegram


def ibkr_log() -> pd.DataFrame:
    rows = [
        # date, side, symbol, filled, avg price, commission, dry_run
        ("2026-10-06", "BUY", "AAA", 100, 10.00, 1.00, False),
        ("2026-10-06", "BUY", "BBB", 50, 20.00, 1.00, False),
        ("2026-10-06", "BUY", "CCC", 10, 5.00, 0.0, True),  # dry run: ignored
        ("2026-10-07", "SELL", "AAA", 100, 10.50, 1.00, False),
        ("2026-10-07", "SELL", "BBB", 50, 19.00, 1.00, False),
    ]
    log = pd.DataFrame(rows, columns=["date", "side", "symbol", "filled_qty", "avg_fill_price", "commission", "dry_run"])
    log["date"] = pd.to_datetime(log["date"])
    return log


def test_trades_from_ibkr_matches_sells_to_buys_with_commissions():
    trades = telegram.trades_from_ibkr(ibkr_log()).set_index("symbol")
    assert list(trades.index) == ["AAA", "BBB"]
    assert trades.loc["AAA", "pnl"] == pytest.approx(100 * 0.50 - 2.00)
    assert trades.loc["BBB", "pnl"] == pytest.approx(-50 * 1.00 - 2.00)
    assert trades.loc["AAA", "net_return"] == pytest.approx(48 / 1000)


def test_summary_windows_and_short_history_note():
    trades = telegram.trades_from_ibkr(ibkr_log())
    trades["sell_date"] = pd.to_datetime(trades["sell_date"])
    lines = telegram.summary_lines(trades, [1, 5, 10, "all"])
    assert lines[1].startswith("1d ") and "-4" in lines[1]  # 48 - 52
    assert lines[2].startswith("5d*")
    assert lines[4].startswith("all ")
    assert lines[-1] == "* only 1 trading day(s) so far"


def test_messages_render():
    orders = pd.DataFrame({"rank": [2, 1], "symbol": ["BBB", "AAA"], "quantity": [50, 100], "limit_price": [20.06, 10.03]})
    text = telegram.buy_message(pd.Timestamp("2026-10-06"), orders, True, "b1542")
    assert "DRY RUN" in text and "AAA" in text and text.index("AAA") < text.index("BBB")
    trades = telegram.trades_from_ibkr(ibkr_log())
    trades["sell_date"] = pd.to_datetime(trades["sell_date"])
    text = telegram.sell_message(pd.Timestamp("2026-10-07"), trades, ["summary"], "ibkr")
    assert "1/2 up" in text and "10.00→10.50" in text and "+48" in text and "IBKR fills" in text


def test_send_without_token_does_not_raise(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    config = {"enabled": True, "token_env": "TELEGRAM_BOT_TOKEN", "chat_id_env": "TELEGRAM_CHAT_ID"}
    assert telegram.send("hello", config) is False


def test_picks_message_shows_overlap_and_sector():
    import pandas as pd
    from src.notify import telegram

    top = pd.DataFrame({"rank": [1, 2], "symbol": ["COHR", "XOM"], "close_cutoff": [335.49, 112.3]})
    main = pd.DataFrame({"symbol": ["COHR", "MU"]})
    text = telegram.picks_message(pd.Timestamp("2026-10-07"), top, "b1542_l250", main, {"COHR": "Information Technology", "XOM": "Energy"})
    assert "b1542_l250, paper" in text and "1/2 same as the traded model" in text and "Energy" in text


def test_sell_message_for_a_compared_ledger(monkeypatch):
    import pandas as pd
    from src.notify import telegram

    trades = pd.DataFrame({"buy_date": ["2026-10-06"], "sell_date": ["2026-10-07"], "symbol": ["AAA"], "shares": [10],
                           "buy_price": [100.0], "sell_price": [101.0], "pnl": [9.0], "net_return": [0.009]})
    asked, sent = [], []
    monkeypatch.setattr(telegram, "trades_from_ledger", lambda ledger: asked.append(ledger) or trades)
    monkeypatch.setattr(telegram, "send", lambda text, config=None: sent.append(text) or True)
    config = {"results_source": "ledger", "ledger": "live_b1542", "summary_windows": [1, "all"]}
    assert telegram.notify_sells(pd.Timestamp("2026-10-07"), config, ledger="live_b1542_l250", label="b1542_l250")
    assert asked == ["live_b1542_l250"] and "(b1542_l250)" in sent[0]
    assert telegram.compared({"compare": [{"ledger": "live_b1542_l250", "label": "b1542_l250"}]}) == {"live_b1542_l250": "b1542_l250"}


def test_compared_picks_go_to_telegram_except_the_traded_model(monkeypatch):
    import pandas as pd
    from src import live
    from src.broker import ibkr
    from src.notify import telegram

    sent = []
    monkeypatch.setattr(telegram, "compared", lambda config=None: {"live_mini": "mini", "live_b1542_l250": "b1542_l250"})
    monkeypatch.setattr(telegram, "notify_picks", lambda day, picks, label, main=None, sectors=None: sent.append((label, main is not None)))
    monkeypatch.setattr(ibkr, "load_config", lambda: {"strategy": {"profile": "b1542"}})
    monkeypatch.setattr(live.paper, "load_ledger", lambda ledger: pd.DataFrame({"trade_date": ["2026-10-09"], "symbol": ["A"]}))
    monkeypatch.setattr(live, "sectors", lambda: {})
    live.notify_compared_picks(pd.Timestamp("2026-10-09"), {"mini": pd.DataFrame({"symbol": ["A"]})})
    assert sent == [("mini", True)]  # mini's own run: compared with b1542's picks from its ledger
    sent.clear()
    live.notify_compared_picks(pd.Timestamp("2026-10-09"), {"b1542": pd.DataFrame({"symbol": ["A"]}), "b1542_l250": pd.DataFrame({"symbol": ["B"]})})
    assert sent == [("b1542_l250", True)]  # the traded model gets no picks message (it has the buy message)
