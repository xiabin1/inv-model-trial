"""Daily live run: stream bars, build features, predict at the cutoff; refresh history overnight.

    python -m src.live predict               # ~15:56 ET: stream today's bars up to the cutoff -> top N
    python -m src.live predict --date 2026-09-25 --skip-pull   # replay a past day
    python -m src.live nightly               # after midnight ET: finish the previous session
    python -m src.live retrain               # weekly: refit the live model on recent labels

`predict` streams today's 1m bars from Databento's Live API (src/pull/live_bars.py; the
historical API is ~10 minutes behind during the session) and then uses exactly the batch
code (minute features, combined join) on that file, so live rows equal the rows the model
was trained on (scripts/check_leakage.py compares the two). `nightly` replaces that partial file with
the full session, which is only available from Databento after midnight ET, and adds
the previous session's label (it needs this session's open).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier

from .data.feature_loader import load_feature_data
from .data.features.combined_feature_manager import CombinedFeatureManager
from .data.features.daily_feature_manager import DailyFeatureManager
from .data.features.label_manager import LabelManager
from .data.features.minutes_feature_manager import MinutesFeatureManager, session_bounds
from .data.raw.dbn_store import load_stored_dbn_bars
from .data.raw.util.market_calendar import is_early_close, is_market_session, market_sessions
from .data.storage import Storage
from .pull import minutes_pull, session_bars
from . import paper
from .pull.live_bars import stream_until_cutoff
from .train import NON_FEATURE_COLUMNS, ROOT_DIR, load_runtime_config
from .walk_forward import MAX_TRAIN_MOVE, label_rows, trainable, training_rows, variant_label
from .models.model_registry import save as save_model

MARKET_TZ = "America/New_York"
# Parallel pipelines. "basic" is the original one (XNAS.BASIC minute features). "mini"
# computes the minute features from EQUS.MINI, whose live feed the plan includes, and keeps
# daily features, labels and paper prices on XNAS.BASIC; its nightly run must follow the
# basic one (which builds the shared session bars and daily features).
PROFILES = {
    "basic": {
        "minutes": ROOT_DIR / "config/feature/minutes_feature_config.yaml",
        "daily": ROOT_DIR / "config/feature/daily_feature_config.yaml",
        "labels": ROOT_DIR / "config/feature/label_config.yaml",
        "combined": ROOT_DIR / "config/feature/combined_feature_config.yaml",
        "train": ROOT_DIR / "config/train/train_model.yaml",
        "pull": ROOT_DIR / "config/data/minutes_config.yaml",
        "builds_daily": True,
        "ledger_suffix": "",
        "live_source": "stream",
    },
    "mini": {
        "minutes": ROOT_DIR / "config/feature/minutes_feature_config_mini.yaml",
        "daily": ROOT_DIR / "config/feature/daily_feature_config.yaml",
        "labels": ROOT_DIR / "config/feature/label_config_mini.yaml",
        "combined": ROOT_DIR / "config/feature/combined_feature_config_mini.yaml",
        "train": ROOT_DIR / "config/train/train_model_mini.yaml",
        "pull": ROOT_DIR / "config/data/minutes_equs_mini_config.yaml",
        "builds_daily": False,
        "ledger_suffix": "_mini",
        "live_source": "stream",
    },
    # XNAS.BASIC minute features up to 15:42: on the plan, same-day XNAS.BASIC is 15 minutes
    # delayed, so the bars before 15:42 are downloadable at ~15:57 and the 15:58 decision needs
    # no live licence. Labels are the XNAS.BASIC 15:57 close -> next open (labels/mini_v1).
    "b1542": {
        "minutes": ROOT_DIR / "config/feature/minutes_feature_config_b1542.yaml",
        "daily": ROOT_DIR / "config/feature/daily_feature_config.yaml",
        "labels": ROOT_DIR / "config/feature/label_config_mini.yaml",
        "combined": ROOT_DIR / "config/feature/combined_feature_config_b1542.yaml",
        "train": ROOT_DIR / "config/train/train_model_b1542.yaml",
        "pull": ROOT_DIR / "config/data/minutes_config.yaml",
        "builds_daily": False,
        "nightly_pull": False,  # the basic nightly already downloads the same XNAS.BASIC day
        "ledger_suffix": "_b1542",
        "live_source": "delayed",
    },
    # b1542 features and cutoff, trained on the S&P 500 / Nasdaq-100 stocks plus liquid US stocks
    # (>= $250M median daily dollar volume, src/symbols/liquidity_universe.py); picks only from the
    # current stocks, at most `live.sector_cap` per sector. A shadow of b1542: at the cutoff it scores
    # the rows b1542 has just built (same feature definitions), so it needs no data of its own then.
    # Its nightly run extends its own training data (lists, minute download, features, labels).
    "b1542_l250": {
        "minutes": ROOT_DIR / "config/feature/minutes_feature_config_l250.yaml",
        "daily": ROOT_DIR / "config/feature/daily_feature_config_l250.yaml",
        "labels": ROOT_DIR / "config/feature/label_config_l250.yaml",
        "combined": ROOT_DIR / "config/feature/combined_feature_config_l250.yaml",
        "train": ROOT_DIR / "config/train/train_model_b1542_l250.yaml",
        "pull": ROOT_DIR / "config/data/minutes_config_l250.yaml",
        "session": ROOT_DIR / "config/data/session_bars_config_l250.yaml",
        "builds_daily": True,
        "universe": True,
        "shadow_of": "b1542",
        "ledger_suffix": "_b1542_l250",
        "live_source": "delayed",
    },
    # mini features (EQUS.MINI to 15:58) trained on the same wider universe; picks from the current
    # stocks, at most 3 per sector. A shadow of mini: at the cutoff it scores the rows mini has just
    # built. Its nightly run (after b1542_l250's, which builds the lists, daily features and labels)
    # adds the EQUS.MINI minute data and mini-style features of the wider universe.
    "mini_l250": {
        "minutes": ROOT_DIR / "config/feature/minutes_feature_config_mini_l250.yaml",
        "daily": ROOT_DIR / "config/feature/daily_feature_config_l250.yaml",
        "labels": ROOT_DIR / "config/feature/label_config_l250.yaml",
        "combined": ROOT_DIR / "config/feature/combined_feature_config_mini_l250.yaml",
        "train": ROOT_DIR / "config/train/train_model_mini_l250.yaml",
        "pull": ROOT_DIR / "config/data/minutes_equs_mini_config_l250.yaml",
        "builds_daily": False,
        "shadow_of": "mini",
        "ledger_suffix": "_mini_l250",
        "live_source": "stream",
    },
}
SECTORS_PATH = ROOT_DIR / "config/symbols/sectors.csv"


def sectors() -> dict[str, str]:
    """GICS sector per symbol (config/symbols/sectors.csv); missing symbols are not capped."""
    if not SECTORS_PATH.exists():
        return {}
    table = pd.read_csv(SECTORS_PATH)
    return dict(zip(table["symbol"], table["sector"]))


def enough_history(rows: pd.DataFrame, features: list[str], live: dict) -> list[str]:
    """Symbols with at least `live.min_feature_share` (default 0.9) of their features present.

    A stock with a short history (new listing, spin-off, IPO, back from a long halt) lacks the
    20-60 day averages and volatilities, so its score rests on part of the information; such
    stocks are scored but not picked (e.g. VYLR on 2026-10-08, six sessions after its spin-off:
    168 of 329 features)."""
    share = rows[features].notna().mean(axis=1)
    minimum = float(live.get("min_feature_share", 0.9))
    short = rows.loc[share < minimum, "symbol"]
    if len(short):
        print(f"Not picked (less than {minimum:.0%} of the features, too little history): "
              + ", ".join(f"{s} {v:.0%}" for s, v in zip(short, share[share < minimum])))
    return rows.loc[share >= minimum, "symbol"].tolist()


def old_enough(symbols: list[str], day: pd.Timestamp, live: dict) -> list[str]:
    """Symbols that have traded for at least `live.min_listing_days` (default 20) sessions before `day`.

    First trading days come from the all-market daily bars (raw/daily_all/v1/first_seen.parquet),
    so a long-listed stock that has just joined the index is not held back; a renamed ticker
    counts from its old ticker's first day. Without that file the rule is skipped."""
    from .data.raw.ticker_renames import load_renames
    from .data.raw.util.market_calendar import market_sessions
    from .symbols import liquidity_universe

    minimum = int(live.get("min_listing_days", 20))
    store = liquidity_universe.storage()
    path = liquidity_universe.RAW_DIR / "first_seen.parquet"
    if not minimum or not store.exists(path):
        print("Listing-age rule skipped (no first_seen table)")
        return list(symbols)
    first = dict(zip(*store.read_parquet(path)[["symbol", "first_day"]].to_numpy().T))
    for rename in load_renames():
        if rename.symbol in first and rename.old in first:
            first[rename.symbol] = min(first[rename.symbol], first[rename.old])
    sessions = pd.DatetimeIndex(market_sessions(pd.Timestamp(day) - pd.Timedelta(days=60), pd.Timestamp(day)))
    sessions = sessions[sessions < pd.Timestamp(day)]
    cutoff = sessions[-minimum] if len(sessions) >= minimum else sessions[0]  # must have traded on or before this session
    young = {s: first[s] for s in symbols if s in first and pd.Timestamp(first[s]) > cutoff}
    if young:
        print(f"Not picked (listed for less than {minimum} sessions): "
              + ", ".join(f"{s} since {pd.Timestamp(d):%Y-%m-%d}" for s, d in sorted(young.items())))
    return [s for s in symbols if s not in young]


def select_top(scores: pd.DataFrame, top_n: int, sector_cap: int | None = None) -> pd.DataFrame:
    """Best `top_n` rows by score; with `sector_cap`, at most that many per sector (the next best
    stock of another sector takes the place). Stocks without a known sector are never capped."""
    ranked = scores.sort_values("score", ascending=False)
    if not sector_cap:
        return ranked.head(top_n).reset_index(drop=True)
    sector = sectors()
    unknown = sorted(set(ranked["symbol"].head(3 * top_n)) - set(sector))
    if unknown:
        print(f"Sector cap: no sector for {unknown} (add them to {SECTORS_PATH.relative_to(ROOT_DIR)}); not capped")
    chosen, count = [], {}
    for index, symbol in zip(ranked.index, ranked["symbol"]):
        group = sector.get(symbol)
        if group is not None and count.get(group, 0) >= sector_cap:
            continue
        chosen.append(index)
        if group is not None:
            count[group] = count.get(group, 0) + 1
        if len(chosen) == top_n:
            break
    return ranked.loc[chosen].reset_index(drop=True)


def today() -> pd.Timestamp:
    return pd.Timestamp.now(tz=MARKET_TZ).tz_localize(None).normalize()


def previous_session(day: pd.Timestamp) -> pd.Timestamp:
    return [session for session in market_sessions(day - pd.Timedelta(days=10), day) if session < day][-1]


def step(name: str, action) -> object:
    started = time.perf_counter()
    result = action()
    print(f"[{pd.Timestamp.now(tz=MARKET_TZ):%H:%M:%S}] {name} ({time.perf_counter() - started:.1f}s)", flush=True)
    return result


def stream_bars(day: pd.Timestamp, cutoff_utc: pd.Timestamp, live: dict, pull_config: Path) -> None:
    """Stream today's bars up to the cutoff; refuse to predict if most symbols lack the last bar."""
    summary = stream_until_cutoff(day, cutoff_utc, float(live["grace_seconds"]), float(live["max_wait_seconds"]), pull_config)
    fresh = summary["with_last_bar"] / max(summary["instruments"], 1)
    if summary["errors"] or fresh < float(live["min_fresh_share"]):
        raise RuntimeError(
            f"Only {fresh:.0%} of symbols have the bar before the cutoff (gateway errors: {summary['errors'] or 'none'}); "
            "not predicting on incomplete data (features would silently use an earlier price)"
        )


def gap_guard(day: pd.Timestamp, combined: pd.DataFrame, max_move: float = 0.40) -> pd.DataFrame:
    """Stocks whose overnight gap into `day` looks like an unrecorded split or spin-off.

    `min_ret_gap` (today's open / previous close - 1) is computed after the recorded corporate
    actions, so a split-like ratio (1/2, 1/3, 2, ... within 8%) or a move beyond `max_move` means
    an action is missing: the features would read it as a crash or a jump. Such stocks are left
    out of the day's picks and written to logs/corporate_action_alerts.log."""
    from .data.raw.corporate_actions import nearest_split

    ratio = 1 + combined.set_index("symbol")["min_ret_gap"].astype(float)
    split_like = ratio.map(lambda value: nearest_split(value) is not None if pd.notna(value) else False)
    flagged = ratio[split_like | ((ratio - 1).abs() > max_move)]
    found = pd.DataFrame({"symbol": flagged.index, "ratio": flagged.to_numpy()})
    if not found.empty:
        alert = ROOT_DIR / "logs/corporate_action_alerts.log"
        alert.parent.mkdir(parents=True, exist_ok=True)
        with alert.open("a") as file:
            for _, row in found.iterrows():
                file.write(f"{pd.Timestamp.now(tz=MARKET_TZ):%Y-%m-%d %H:%M} {row['symbol']} {day:%Y-%m-%d}: excluded from "
                           f"the prediction, open/prev close {row['ratio']:.4f} - verify and add to corporate_actions.yaml\n")
        print(f"GAP GUARD: excluded {found['symbol'].tolist()} (open/prev close {found['ratio'].round(3).tolist()})")
        propose_actions(day, found, "15:58 gap guard")
    return found


def propose_actions(day: pd.Timestamp, found: pd.DataFrame, found_by: str) -> None:
    """Pending entries + Telegram Verify/Reject messages for unexplained gaps (never stops the caller)."""
    try:
        from .corporate_action_flow import propose

        for _, row in found.iterrows():
            propose(row["symbol"], day, float(row["ratio"]), found_by)
    except Exception as error:
        print(f"Corporate action proposal failed: {error}")


def delayed_bars(day: pd.Timestamp, cutoff_utc: pd.Timestamp, live: dict, pull_config: Path) -> None:
    """Download the day's bars up to the cutoff once the data delay has passed; refuse incomplete data.

    For profiles whose cutoff is early enough that the delayed historical feed already has every
    bar before it at decision time (b1542: bars before 15:42, delayed 15 minutes)."""
    delay = pd.Timedelta(minutes=float(live.get("data_delay_minutes", 15)))
    ready = cutoff_utc + delay + pd.Timedelta(seconds=5)
    deadline = ready + pd.Timedelta(seconds=float(live.get("max_wait_seconds", 45)))
    wait = (ready - pd.Timestamp.now(tz="UTC")).total_seconds()
    if wait > 0:
        print(f"Waiting {wait:.0f}s until {ready.tz_convert(MARKET_TZ):%H:%M:%S} for the delayed bars")
        time.sleep(wait)
    config = minutes_pull.load_config(pull_config)
    storage = Storage(config["storage"], ROOT_DIR)
    path = Path(config["version"]["data"]) / minutes_pull.format_day_filename(day, config["schedule"]["interval"])
    last_bar = cutoff_utc - pd.Timedelta(minutes=1)
    while True:
        minutes_pull.main(end_utc=cutoff_utc, rolling=True, config_path=pull_config)
        bars = load_stored_dbn_bars(storage, [path], adjust=False)
        symbols = max(bars["symbol"].nunique(), 1)
        fresh = bars.loc[bars["datetime_utc"] == last_bar, "symbol"].nunique() / symbols
        print(f"Bars at {last_bar.tz_convert(MARKET_TZ):%H:%M}: {fresh:.0%} of {symbols} symbols")
        if fresh >= float(live["min_fresh_share"]):
            return
        if pd.Timestamp.now(tz="UTC") >= deadline:
            raise RuntimeError(
                f"Only {fresh:.0%} of symbols have the {last_bar.tz_convert(MARKET_TZ):%H:%M} bar by "
                f"{deadline.tz_convert(MARKET_TZ):%H:%M:%S}; not predicting on incomplete data"
            )
        time.sleep(5)


def predict(
    day: pd.Timestamp, skip_pull: bool, model_path: Path | None = None, ledger: str = "live", profile: str = "basic",
    trade: bool = False,
) -> pd.DataFrame | None:
    """Top-N list for `day`; the top picks are recorded as mock buys in the paper `ledger`."""
    if not is_market_session(day) or is_early_close(day):
        print(f"{day:%Y-%m-%d}: no regular session (or a half day, which the model skips); nothing to predict")
        return None
    paths = PROFILES[profile]
    kind = ledger  # "live" (real time) or e.g. "late"; the profile suffix only names the ledger file
    ledger = ledger + paths["ledger_suffix"]
    config = load_runtime_config(paths["train"])
    live = config["live"]
    minutes = MinutesFeatureManager(paths["minutes"])
    cutoff_utc = session_bounds(day, minutes.cutoff_before_close)[2]
    # Earlier sessions' files do not change; reading them before the stream leaves only
    # today's file to process after the cutoff.
    step("preload earlier sessions for minute features", lambda: minutes.preload(day))
    if not skip_pull:
        if paths["live_source"] == "delayed":
            step("download delayed minute bars up to the cutoff", lambda: delayed_bars(day, cutoff_utc, live, paths["pull"]))
        else:
            step("stream minute bars up to the cutoff", lambda: stream_bars(day, cutoff_utc, live, paths["pull"]))
    step("minute features", lambda: minutes.calculate_features(day, day))
    combined = step("combine with previous-session daily features", lambda: CombinedFeatureManager(paths["combined"]).combine(day, day))
    daily_columns = [column for column in combined.columns if column.startswith("daily_")]
    without_daily = combined[daily_columns].isna().all(axis=1)
    if without_daily.all():
        raise RuntimeError(f"No daily features for {previous_session(day):%Y-%m-%d}: did `nightly` run this morning?")

    model_path = model_path or ROOT_DIR / live["model_path"]
    if not model_path.exists():
        raise RuntimeError(f"No live model at {model_path}; run `python -m src.live retrain --profile {profile}` first")
    model = joblib.load(model_path)
    scores = combined[["date", "symbol"]].copy()
    scores["score"] = model.predict_proba(combined[list(model.feature_names_in_)])[:, list(model.classes_).index(1)]
    scores = scores[scores["symbol"].isin(enough_history(combined, list(model.feature_names_in_), live))]
    scores = scores[scores["symbol"].isin(old_enough(scores["symbol"].tolist(), day, live))]
    excluded = gap_guard(day, combined, float(live.get("gap_guard_max_move", 0.40)))
    scores = scores[~scores["symbol"].isin(excluded["symbol"])]
    from .corporate_action_flow import excluded_symbols

    waiting = excluded_symbols(day)  # pending (or verified but not yet rebuilt) corporate actions
    if waiting:
        print(f"Excluded until the corporate action is decided/applied: {waiting}")
        scores = scores[~scores["symbol"].isin(list(waiting))]
    storage = Storage(config["storage"], ROOT_DIR)
    cutoff_prices = storage.read_parquet(minutes.output_dir() / f"{day:%Y-%m-%d}.parquet")[["symbol", "close_cutoff"]]
    top = select_top(scores.merge(cutoff_prices, on="symbol", how="left"), int(live["top_n"]), live.get("sector_cap"))
    top.insert(1, "rank", top.index + 1)
    output = Path(config["storage"]["output"]["predictions"]) / config["version"]["evaluation"]
    output = (output if kind == "live" else output / kind) / f"{day:%Y-%m-%d}.csv"
    storage.write_csv(top, output)
    meta = json.loads((model_path.with_suffix(".json")).read_text()) if model_path.with_suffix(".json").exists() else {}
    print(
        f"\nTop {len(top)} for {day:%Y-%m-%d} (model trained on {meta.get('train_from', '?')}..{meta.get('train_to', '?')}, "
        f"{len(scores)} stocks scored, {int(without_daily.sum())} without daily features):"
    )
    print(top.to_string(index=False))
    print(f"Saved to {storage.describe(output)}")
    paper.record_buys(top, day, meta.get("train_to", "?"), ledger)
    tops = {profile: top}
    for name in [name for name, other in PROFILES.items() if other.get("shadow_of") == profile]:
        try:  # a shadow model must never stop the main prediction or the orders
            tops[name] = step(f"shadow model {name}", lambda name=name: shadow_predict(name, day, kind, combined, scores, cutoff_prices))
        except Exception as error:
            print(f"Shadow model {name} failed: {error}")
    if trade:
        from .broker import ibkr

        broker = ibkr.load_config()
        target = broker["strategy"]["profile"]
        if target not in tops:
            print(f"IBKR: not trading the {profile} picks (config/broker/ibkr_config.yaml trades {target})")
            if PROFILES.get(target, {}).get("shadow_of") == profile:  # the traded shadow failed above: no orders today
                raise RuntimeError(f"IBKR: the traded model {target} produced no picks, so no buy orders were sent")
        else:
            step("IBKR buy orders", lambda: ibkr.buy(day, tops[target], config=broker))
    if kind == "live":  # compared models' picks in Telegram, after the orders
        try:
            notify_compared_picks(day, tops)
        except Exception as error:  # messages never stop trading
            print(f"Telegram: {error}")
    return top


def notify_compared_picks(day: pd.Timestamp, tops: dict[str, pd.DataFrame]) -> None:
    """Picks message for each model of this run listed under `compare` in config/notify/telegram.yaml
    (the traded model has its own buy message), with the overlap with the traded model's picks."""
    from .broker import ibkr
    from .notify import telegram

    compare = telegram.compared()
    traded = ibkr.load_config()["strategy"]["profile"]
    main = tops.get(traded)
    if main is None:  # the traded model runs in another job; its picks are in its paper ledger
        ledger = paper.load_ledger(f"live{PROFILES[traded]['ledger_suffix']}")
        main = ledger[ledger["trade_date"].astype(str) == f"{day:%Y-%m-%d}"] if not ledger.empty else None
    for name, picks in tops.items():
        label = compare.get(f"live{PROFILES[name]['ledger_suffix']}")
        if name != traded and label:
            telegram.notify_picks(day, picks, label, main=main if main is not None and len(main) else None, sectors=sectors())


def shadow_predict(name: str, day: pd.Timestamp, kind: str, combined: pd.DataFrame, allowed: pd.DataFrame,
                   cutoff_prices: pd.DataFrame) -> pd.DataFrame:
    """Top-N of the shadow profile `name`, scored on the rows its main profile just built (`combined`),
    restricted to the stocks the main profile may pick (`allowed`: after the gap guard and the
    corporate-action exclusions); saved and mock-bought like the main profile's picks."""
    paths = PROFILES[name]
    config = load_runtime_config(paths["train"])
    live = config["live"]
    model_path = ROOT_DIR / live["model_path"]
    model = joblib.load(model_path)
    rows = combined[combined["symbol"].isin(allowed["symbol"])]
    rows = rows[rows["symbol"].isin(enough_history(rows, list(model.feature_names_in_), live))]
    rows = rows[rows["symbol"].isin(old_enough(rows["symbol"].tolist(), day, live))]
    scores = rows[["date", "symbol"]].copy()
    scores["score"] = model.predict_proba(rows[list(model.feature_names_in_)])[:, list(model.classes_).index(1)]
    top = select_top(scores.merge(cutoff_prices, on="symbol", how="left"), int(live["top_n"]), live.get("sector_cap"))
    top.insert(1, "rank", top.index + 1)
    storage = Storage(config["storage"], ROOT_DIR)
    output = Path(config["storage"]["output"]["predictions"]) / config["version"]["evaluation"]
    output = (output if kind == "live" else output / kind) / f"{day:%Y-%m-%d}.csv"
    storage.write_csv(top, output)
    meta = json.loads(model_path.with_suffix(".json").read_text()) if model_path.with_suffix(".json").exists() else {}
    print(f"Shadow {name}: top {len(top)} (model to {meta.get('train_to', '?')}, sector cap {live.get('sector_cap') or '-'}): "
          f"{' '.join(top['symbol'])}")
    paper.record_buys(top, day, meta.get("train_to", "?"), kind + paths["ledger_suffix"])
    return top


def check_corporate_actions(day: pd.Timestamp, session_config: Path | None = None) -> pd.DataFrame:
    """Unexplained overnight jumps into `day` (after recorded splits/spin-offs/renames).

    A split or spin-off missing from config/symbols/corporate_actions.yaml looks like a crash or
    jump to the model and corrupts labels; this writes a warning to logs/corporate_action_alerts.log
    so it can be verified and added (e.g. CTVA/Vylor 2026-10-01 went unnoticed for a weekend)."""
    from .data.raw.corporate_actions import find_candidates
    from .data.raw.session_bars import load_session_bars, session_filename

    config = session_bars.load_config(session_config) if session_config else session_bars.load_config()
    storage = Storage(config["storage"], ROOT_DIR)
    directory = Path(config["storage"]["output"]["session_bars"]) / config["version"]["data"]
    bars = load_session_bars(storage, [directory / session_filename(previous_session(day)), directory / session_filename(day)])
    found = find_candidates(bars)
    found = found[found["date"] == day]
    if not found.empty:
        alert = ROOT_DIR / "logs/corporate_action_alerts.log"
        alert.parent.mkdir(parents=True, exist_ok=True)
        with alert.open("a") as file:
            for _, row in found.iterrows():
                file.write(f"{pd.Timestamp.now(tz=MARKET_TZ):%Y-%m-%d %H:%M} {row['symbol']} {day:%Y-%m-%d}: open/prev close "
                           f"{row['ratio']:.4f} (suggested split {row['suggested_split']}) - verify and add to corporate_actions.yaml\n")
        print(f"WARNING: unexplained overnight jumps on {day:%Y-%m-%d}: {found['symbol'].tolist()} -> logs/corporate_action_alerts.log")
        propose_actions(day, found, "nightly check")
    try:  # smaller drops: spin-off filings at the SEC (never stops the nightly run)
        from .corporate_action_scan import spinoff_gap_check
        from .data.raw.corporate_actions import bar_days

        session = bars.assign(date=bar_days(bars["datetime_utc"])).sort_values(["symbol", "date"])
        session["ratio"] = session["open"] / session.groupby("symbol")["close"].shift(1)
        gaps = session[session["date"] == day].set_index("symbol")["ratio"].dropna()
        spinoff_gap_check(day, gaps, skip=set(found["symbol"]))
    except Exception as error:
        print(f"Spin-off check failed: {error}")
    return found


def nightly(day: pd.Timestamp, skip_pull: bool = False, profile: str = "basic") -> None:
    """Finish session `day` (full-day bars, features), label the session before it, and
    paper-trade `day` late: mock-buy its cutoff picks, mock-sell earlier picks at `day`'s open."""
    if not is_market_session(day):
        print(f"{day:%Y-%m-%d}: not a market session; nothing to do")
        return
    paths = PROFILES[profile]
    before = previous_session(day)
    if paths.get("universe"):
        from .symbols import liquidity_universe

        step("all-market daily bars and the next session's stock lists", lambda: liquidity_universe.update(day))
    if not skip_pull and paths.get("nightly_pull", True):
        step(f"pull full-session minute bars for {day:%Y-%m-%d}", lambda: minutes_pull.main(day, day, config_path=paths["pull"]))
    if paths["builds_daily"]:
        step("session daily bars", lambda: session_bars.main(day, day, config_path=paths.get("session")))
        step("check for unrecorded splits / spin-offs", lambda: check_corporate_actions(day, paths.get("session")))
        step("daily features", lambda: DailyFeatureManager(paths["daily"]).calculate_features(start_date=day, end_date=day))
    if not is_early_close(day):
        step("minute features (full-session file; same values as at the cutoff)",
             lambda: MinutesFeatureManager(paths["minutes"]).calculate_features(day, day))
    if not is_early_close(before):
        step(f"labels for {before:%Y-%m-%d} (needs {day:%Y-%m-%d}'s open)", lambda: LabelManager(paths["labels"]).calculate_labels(before, before))
    full_days = [session for session in (before, day) if not is_early_close(session)]
    if full_days:
        # Half days have no minute-feature rows, so they are left out of the join.
        step("combined features", lambda: CombinedFeatureManager(paths["combined"]).combine(full_days[0], full_days[-1]))
    if not is_early_close(day) and not paths.get("shadow_of"):  # shadows are scored in their main profile's run
        # Paper-trade `day` from its full-session file: features only use bars before the
        # cutoff, so the picks are those a live run would have made (ledger "late").
        step(f"late prediction for {day:%Y-%m-%d} (mock buys, ledger late{paths['ledger_suffix']})",
             lambda: predict(day, skip_pull=True, ledger="late", profile=profile))
    for ledger in ("live", "late"):
        ledger += paths["ledger_suffix"]
        step(f"paper trades ({ledger}): buy prices (cutoff minute) and sell prices (next open)", lambda: paper.settle(ledger))


def retrain(as_of: pd.Timestamp | None = None, model_path: Path | None = None, profile: str = "basic") -> None:
    """Fit the live model like the configured walk-forward variant, on every label known now.

    With `as_of`, only on the labels known before that session (as the walk-forward does),
    e.g. to replay past sessions without look-ahead."""
    config = load_runtime_config(PROFILES[profile]["train"])
    live = config["live"]
    variant = {v["name"]: v for v in config["walk_forward"]["variants"]}[live["variant"]]
    storage = Storage(config["storage"], ROOT_DIR)
    data = load_feature_data(Path(config["storage"]["input"]["features"]) / config["version"]["features"], storage)
    feature_columns = [column for column in data.columns if column not in NON_FEATURE_COLUMNS]
    labeled = label_rows(data, variant_label(config, variant))
    window = variant.get("train_window", "expanding")
    if as_of is not None:
        all_days = sorted(pd.Timestamp(day) for day in data["date"].unique())
        train = training_rows(labeled, all_days, pd.Timestamp(as_of), window)
    else:
        days = sorted(labeled["date"].unique())
        train = labeled[labeled["date"].isin(days[-int(window):] if window != "expanding" else days)]
    train = trainable(train, float(config["model"].get("max_train_move", MAX_TRAIN_MOVE)))
    days = sorted(train["date"].unique())
    model = LGBMClassifier(**config["model"]["params"])
    model.fit(train[feature_columns], train["label"].astype(int))
    live_model = model_path is None  # replay/simulation models are not kept in the history
    model_path = model_path or ROOT_DIR / live["model_path"]
    meta = {
        "variant": live["variant"],
        "features_version": config["version"]["features"],
        "train_from": f"{pd.Timestamp(days[0]):%Y-%m-%d}",
        "train_to": f"{pd.Timestamp(days[-1]):%Y-%m-%d}",
        "train_rows": len(train),
        "features": len(feature_columns),
        "trained_at": f"{pd.Timestamp.now(tz=MARKET_TZ):%Y-%m-%d %H:%M %Z}",
    }
    kept = save_model(model, model_path, meta, keep_history=live_model)
    print(f"Saved {model_path.relative_to(ROOT_DIR)}: {json.dumps(meta)}" + (f"; history {kept.relative_to(ROOT_DIR)}" if kept else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Live daily run")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("predict", help="At the cutoff: pull today's bars and write the top-N list")
    run.add_argument("--date", help="Session to predict (default: today, New York time)")
    run.add_argument("--skip-pull", action="store_true", help="Use the minute file already stored (rerun a past day)")
    run.add_argument("--ledger", default="live", help="Paper ledger for the mock buys (e.g. late for after-the-fact runs)")
    run.add_argument("--trade", action="store_true", help="Also place the IBKR buy orders (config/broker/ibkr_config.yaml)")
    night = commands.add_parser("nightly", help="After midnight ET: finish the previous session")
    night.add_argument("--date", help="Session to finish (default: the last session before today)")
    night.add_argument("--skip-pull", action="store_true", help="Use the minute file already stored (rerun a past day)")
    refit = commands.add_parser("retrain", help="Refit the live model on all labels known now")
    for command in (run, night, refit):
        command.add_argument("--profile", default="basic", choices=sorted(PROFILES),
                             help="Pipeline: basic (XNAS.BASIC minute features) or mini (EQUS.MINI minute features)")
    args = parser.parse_args()

    if args.command == "predict":
        predict(pd.Timestamp(args.date) if args.date else today(), args.skip_pull, ledger=args.ledger, profile=args.profile,
                trade=args.trade)
    elif args.command == "nightly":
        nightly(pd.Timestamp(args.date) if args.date else previous_session(today()), args.skip_pull, profile=args.profile)
    else:
        retrain(profile=args.profile)


if __name__ == "__main__":
    main()
