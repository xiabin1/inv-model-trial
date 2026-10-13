#!/usr/bin/env python3
"""Robustness checks on a walk-forward variant's saved predictions (no retraining).

1. month-by-month returns, 2. top 3/5/10/30/50, 3. where the profit comes from
(big days, volatile picks, up vs down market nights, repeated stocks), 4. trading-cost
sensitivity, 5. weekday effect and skipping days before weekends/holidays.
Optionally compares with a second variant.

    python3 scripts/analyze_walk_forward.py weekly_rolling120_threshold --compare weekly_rolling120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data.raw.util.market_calendar import market_sessions  # noqa: E402
from src.data.storage import Storage  # noqa: E402
from src.models.evaluate import calculate_strategy_metrics  # noqa: E402
from src.train import load_runtime_config  # noqa: E402

COSTS = (0.0005, 0.001, 0.0015, 0.002)


def picks(scored: pd.DataFrame, top_n: int) -> pd.DataFrame:
    return scored.sort_values(["date", "prediction"], ascending=[True, False]).groupby("date").head(top_n)


def daily(scored: pd.DataFrame, top_n: int, cost: float) -> pd.Series:
    return picks(scored, top_n).groupby("date")["future_return"].mean() - cost


def days_before_gap(days: pd.DatetimeIndex) -> pd.Series:
    """True for trading days whose next session is more than one calendar day later
    (Fridays, and days before a market holiday): positions are held through a closure."""
    sessions = market_sessions(days.min(), days.max() + pd.Timedelta(days=10))
    next_session = pd.Series(sessions[1:], index=sessions[:-1])
    gap = (next_session.reindex(days) - days).dt.days
    return pd.Series(gap.to_numpy() > 1, index=days)


def trading_rules(scored: pd.DataFrame, name: str, cost: float) -> None:
    days = pd.DatetimeIndex(sorted(scored["date"].unique()))
    before_gap = days_before_gap(days)
    friday = pd.Series(days.dayofweek == 4, index=days)
    print(f"\n{name}: average return per trading day by buy day (after {cost:.2%} cost)")
    rows = []
    for n in (10, 30):
        d = daily(scored, n, cost)
        by_day = d.groupby(d.index.day_name()).agg(["mean", "count"])
        rows.append({"top": n, **{day: f"{by_day.loc[day, 'mean']:+.3%} ({int(by_day.loc[day, 'count'])})"
                                   for day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday") if day in by_day.index}})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"{name}: trading rules (annual return, sharpe, max drawdown, t, trading days)")
    for n in (10, 30, 50):
        d = daily(scored, n, cost)
        for rule, skip in (("every day", None), ("skip Fridays", friday), ("skip days before weekends/holidays", before_gap)):
            kept = d if skip is None else d[~skip.reindex(d.index).to_numpy()]
            # Days not traded earn 0 and cost nothing; annualize over all calendar trading days.
            full = kept.reindex(d.index, fill_value=0.0)
            annual = np.prod(1 + full) ** (252 / len(full)) - 1
            sharpe = full.mean() / full.std() * np.sqrt(252) if full.std() else float("nan")
            drawdown = (np.cumprod(1 + full) / np.maximum.accumulate(np.cumprod(1 + full)) - 1).min()
            t_stat = kept.mean() / (kept.std() / np.sqrt(len(kept)))
            print(f"  top{n:>2} {rule:<36} {annual:+7.1%}  sharpe {sharpe:5.2f}  maxDD {drawdown:6.1%}  t {t_stat:5.2f}  days {len(kept)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", help="walk-forward variant name, e.g. weekly_rolling120_threshold")
    parser.add_argument("--compare", help="second variant for the month-by-month comparison")
    parser.add_argument("--cost", type=float, default=0.001, help="round-trip cost per day (default 0.001)")
    args = parser.parse_args()

    config = load_runtime_config(ROOT_DIR / "config/train/train_model.yaml")
    storage = Storage(config["storage"], ROOT_DIR)
    base = Path(config["storage"]["output"]["evaluation"]) / "walk_forward"
    scored = storage.read_parquet(base / args.variant / "predictions.parquet")
    other = storage.read_parquet(base / args.compare / "predictions.parquet") if args.compare else None
    market = scored.groupby("date")["future_return"].mean()
    cost = args.cost
    print(f"{args.variant}: {scored['date'].nunique()} days, {scored['date'].min():%Y-%m-%d}..{scored['date'].max():%Y-%m-%d}")

    print(f"\n=== 1. Month by month (sum of daily returns after {cost:.2%} cost) ===")
    series = {f"top{n}": daily(scored, n, cost) for n in (10, 30, 50)}
    if other is not None:
        series.update({f"{args.compare} top{n}": daily(other, n, cost) for n in (10, 30)})
    series["all stocks"] = market - cost
    table = pd.DataFrame({name: s.groupby(s.index.to_period("M")).sum() for name, s in series.items()})
    print(table.map(lambda x: f"{x:+.1%}").to_string())
    print("positive months: " + ", ".join(f"{name} {(table[name] > 0).sum()}/{len(table)}" for name in table))

    print("\n=== 2. Portfolio sizes ===")
    for n in (3, 5, 10, 30, 50):
        m = calculate_strategy_metrics(scored, n, cost)
        print(
            f"top{n:>2}: annual {m['strategy_annual_return']:+.1%}, sharpe {m['strategy_sharpe_ratio']:.2f}, "
            f"maxDD {m['strategy_max_drawdown']:.1%}, median day {m['strategy_median_daily']:+.3%}, "
            f"t {m['strategy_t_stat']:.2f}, worst day {daily(scored, n, cost).min():+.1%}"
        )

    print("\n=== 3. Where the profit comes from ===")
    for n in (10, 30, 50):
        d = daily(scored, n, cost)
        p = picks(scored, n)
        best_days = d.nlargest(5).sum() / d.sum()
        best_picks = (p["future_return"] / n).nlargest(10).sum() / d.sum()
        print(
            f"top{n}: best 5 days = {best_days:.0%} of total, best 10 single picks = {best_picks:.0%} of total; "
            f"mean without best 5 days {d.drop(d.nlargest(5).index).mean():+.3%}/day"
        )
    p10 = picks(scored, 10)
    print(
        f"top10 picks move {p10['future_return'].abs().mean():.2%} overnight vs {scored['future_return'].abs().mean():.2%} "
        f"for all stocks; trailing volatility {p10['overnight_vol'].median():.2%} vs {scored['overnight_vol'].median():.2%}"
    )
    up = market > 0
    print(f"market up nights {up.sum()}, down nights {(~up).sum()} (market = average of all stocks)")
    for n in (10, 30, 50):
        gross = daily(scored, n, 0.0)
        print(
            f"top{n} before cost: up nights {gross[up].mean():+.3%} (market {market[up].mean():+.3%}, "
            f"excess {(gross - market)[up].mean():+.3%}); down nights {gross[~up].mean():+.3%} "
            f"(market {market[~up].mean():+.3%}, excess {(gross - market)[~up].mean():+.3%})"
        )
    for n in (10, 30):
        p = picks(scored, n)
        counts = p["symbol"].value_counts()
        print(
            f"top{n}: {counts.size} different stocks; the 10 most frequent are {counts.head(10).sum() / len(p):.0%} "
            f"of picks: " + ", ".join(f"{symbol} {k}" for symbol, k in counts.head(6).items())
        )

    print("\n=== 4. Trading-cost sensitivity (annual return, t-stat) ===")
    rows = []
    for n in (10, 30, 50):
        gross = daily(scored, n, 0.0)
        row = {"top": n, "gross/day": f"{gross.mean():+.3%}", "break-even cost": f"{gross.mean():.3%}"}
        for c in COSTS:
            net = gross - c
            annual = np.prod(1 + net) ** (252 / len(net)) - 1
            row[f"cost {c:.2%}"] = f"{annual:+.0%} (t {net.mean() / (net.std() / np.sqrt(len(net))):.2f})"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 5. Weekday effect and skipping days before weekends/holidays ===")
    trading_rules(scored, args.variant, cost)
    if other is not None:
        trading_rules(other, args.compare, cost)


if __name__ == "__main__":
    main()
