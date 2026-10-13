from __future__ import annotations

import pandas as pd
import pytest

from src import corporate_action_flow as flow
from src.data.raw import corporate_actions


@pytest.fixture
def actions_file(tmp_path, monkeypatch):
    """A copy of the real actions file (without its pending entries) plus empty workflow state, all under tmp_path."""
    path = tmp_path / "corporate_actions.yaml"
    lines = corporate_actions.ACTIONS_PATH.read_text().splitlines(keepends=True)
    path.write_text("".join(line for line in lines if "verified: false" not in line))
    monkeypatch.setattr(flow, "ACTIONS_PATH", path)
    monkeypatch.setattr(flow, "APPLIED_PATH", tmp_path / "state/applied.json")
    monkeypatch.setattr(flow, "announce", lambda *args, **kwargs: None)  # no Telegram in tests
    flow.applied_state()  # existing verified entries count as applied
    yield path
    corporate_actions.load_actions.cache_clear()


def test_propose_adds_one_pending_entry_that_is_not_applied(actions_file):
    assert flow.propose("ABC", pd.Timestamp("2026-10-07"), 0.5012, "test")
    assert not flow.propose("ABC", pd.Timestamp("2026-10-07"), 0.5012, "test")  # once only
    entry = next(e for e in flow.read_entries() if flow.key(e) == "ABC 2026-10-07")
    assert flow.status(entry) == "pending" and entry["split"] == 2
    applied = corporate_actions.load_actions(actions_file)
    assert not any(a.symbol == "ABC" for a in applied)
    assert flow.excluded_symbols(pd.Timestamp("2026-10-07")) == {"ABC": "pending verification (ABC 2026-10-07)"}
    assert flow.excluded_symbols(pd.Timestamp("2026-10-06")) == {}  # not before its date


def test_unknown_gap_is_pending_and_incomplete(actions_file):
    flow.propose("XYZ", pd.Timestamp("2026-10-07"), 0.52 + 0.3, "test")  # -18%: no split guess
    entry = next(e for e in flow.read_entries() if flow.key(e) == "XYZ 2026-10-07")
    assert entry["type"] == "unknown" and not flow.complete(entry)


def test_verify_then_waiting_for_rebuild_then_applied(actions_file):
    flow.propose("ABC", pd.Timestamp("2026-09-15"), 0.5012, "test")
    assert flow.set_flag("ABC", pd.Timestamp("2026-09-15"), "verify")
    applied = corporate_actions.load_actions(actions_file)
    assert any(a.symbol == "ABC" and a.factor == pytest.approx(0.5) for a in applied)
    assert "waiting for the rebuild" in flow.excluded_symbols(pd.Timestamp("2026-09-16"))["ABC"]
    state = flow.applied_state(); state["ABC 2026-09-15"] = {"status": "verified"}; flow.save_state(flow.APPLIED_PATH, state)
    assert flow.excluded_symbols(pd.Timestamp("2026-09-16")) == {}


def test_verified_future_event_is_not_applied_before_its_ex_date(actions_file):
    flow.propose_known("XYZ", pd.Timestamp("2099-01-05"), {"split": 4}, "test", "test", notify=False)
    assert flow.set_flag("XYZ", pd.Timestamp("2099-01-05"), "verify")
    assert not any(a.symbol == "XYZ" for a in corporate_actions.load_actions(actions_file))
    assert flow.excluded_symbols(pd.Timestamp("2099-01-04")) == {}  # still picked before the ex-date
    assert "XYZ" in flow.excluded_symbols(pd.Timestamp("2099-01-05"))  # excluded from the ex-date


def test_propose_known_spinoff_entry(actions_file):
    assert flow.propose_known("CMCSA", pd.Timestamp("2099-01-05"), {"spinoff": {"child": "vsnt", "ratio": 0.04}}, "ECA test", "test", notify=False)
    entry = next(e for e in flow.read_entries() if flow.key(e) == "CMCSA 2099-01-05")
    assert entry["spinoff"] == {"child": "VSNT", "ratio": 0.04} and flow.status(entry) == "pending" and flow.complete(entry)


def test_reject_keeps_the_entry_and_includes_the_stock(actions_file):
    flow.propose("ABC", pd.Timestamp("2026-10-07"), 0.5012, "test")
    assert flow.set_flag("ABC", pd.Timestamp("2026-10-07"), "reject")
    entry = next(e for e in flow.read_entries() if flow.key(e) == "ABC 2026-10-07")
    assert flow.status(entry) == "rejected"
    assert flow.excluded_symbols(pd.Timestamp("2026-10-08")) == {}
    assert not any(a.symbol == "ABC" for a in corporate_actions.load_actions(actions_file))


def test_next_check_follows_the_cron_schedule():
    tz = "America/New_York"
    assert flow.next_check(pd.Timestamp("2026-10-06 23:52", tz=tz)) == pd.Timestamp("2026-10-07 07:15", tz=tz)  # Tue night -> Wed 07:15
    assert flow.next_check(pd.Timestamp("2026-10-07 10:03", tz=tz)) == pd.Timestamp("2026-10-07 10:15", tz=tz)
    assert flow.next_check(pd.Timestamp("2026-10-07 15:50", tz=tz)) == pd.Timestamp("2026-10-07 16:15", tz=tz)  # not 15:54-16:02
    assert flow.next_check(pd.Timestamp("2026-10-09 18:00", tz=tz)) == pd.Timestamp("2026-10-10 07:15", tz=tz)  # Fri -> Sat 07:15
    assert flow.next_check(pd.Timestamp("2026-10-10 08:00", tz=tz)) == pd.Timestamp("2026-10-12 09:00", tz=tz)  # Sat -> Mon
