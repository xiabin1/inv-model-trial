from __future__ import annotations

import pandas as pd

from src import corporate_action_scan as scan

SPINOFF = ("Company/ Parent: Comcast Corporation Symbol: CMCSA Record Date: December 16, 2025 Payment Date: January 2, 2026 "
           "EX Date: January 5, 2026 Due Bill Redemption Date: January 5, 2026 Ratio of Spin-Off/ Distribution: One share of "
           "Versant Media Group, Inc. Common Stock (VSNT) for every twenty five shares of Comcast Corporation (CMCSA) held. An EX")
REVERSE = ("Request for Information ... Global Engine Group Holding Limited (GLE) will effect a one-for-ten (1-10) reverse split of "
           "its Class A Ordinary Shares. The reverse stock split ... will become effective on Thursday, October 8, 2026.")
FORWARD = "XYZ Corp. (XYZ) will effect a 4-for-1 forward stock split and will begin trading on a split-adjusted basis on Monday, June 15, 2026."


def test_number_words_and_digits():
    assert scan.number("twenty five") == 25 and scan.number("twenty-five") == 25
    assert scan.number("One") == 1 and scan.number("0.5") == 0.5 and scan.number("Request") is None


def test_parse_spinoff_alert():
    assert scan.parse_alert(SPINOFF, "spinoff") == {"date": pd.Timestamp("2026-01-05"),
                                                    "terms": {"spinoff": {"child": "VSNT", "ratio": 0.04}}}


def test_parse_reverse_and_forward_splits():
    assert scan.parse_alert(REVERSE, "reverse_split") == {"date": pd.Timestamp("2026-10-08"), "terms": {"split": 0.1}}
    assert scan.parse_alert(FORWARD, "split") == {"date": pd.Timestamp("2026-06-15"), "terms": {"split": 4.0}}


def test_missing_terms_stay_none():
    assert scan.parse_alert("Spin-Off/Distribution Information for X (X). Ex-date to be determined.", "spinoff") == {"date": None, "terms": None}


def test_spinoff_gap_check_proposes_drops_with_a_spinoff_filing(monkeypatch):
    import src.corporate_action_flow as flow
    import src.corporate_action_scan as scan

    proposed = []
    monkeypatch.setattr(scan, "sec_headers", lambda: {"User-Agent": "test"})
    monkeypatch.setattr(scan, "ticker_ciks", lambda headers: {1: "APTV", 2: "NKE", 3: "XYZ"})
    asked = {}

    def hits(headers, ciks, symbols, start, end, queries):
        asked.update(symbols=symbols, start=start, queries=queries)
        return [{"cik": 1, "date": "2026-03-05", "query": "distribution ratio", "link": "https://sec/aptv"}]

    monkeypatch.setattr(scan, "edgar_hits_by_company", hits)
    monkeypatch.setattr(flow, "propose", lambda symbol, day, ratio, found_by: proposed.append((symbol, round(ratio, 3), found_by)) or True)
    gaps = pd.Series({"APTV": 0.866, "NKE": 0.88, "XYZ": 0.95, "BIG": 0.18})
    result = scan.spinoff_gap_check(pd.Timestamp("2026-04-01"), gaps, skip={"BIG"})

    assert asked["symbols"] == {"APTV", "NKE"}  # -5% is too small; BIG was already flagged by the split rule
    assert asked["start"] == pd.Timestamp("2025-12-02") and '"spin-off"' in asked["queries"]
    assert result == ["APTV"] and proposed[0][:2] == ("APTV", 0.866) and "https://sec/aptv" in proposed[0][2]


def test_spinoff_gap_check_needs_no_sec_call_without_drops(monkeypatch):
    import src.corporate_action_scan as scan

    monkeypatch.setattr(scan, "sec_headers", lambda: (_ for _ in ()).throw(AssertionError("no SEC call expected")))
    assert scan.spinoff_gap_check(pd.Timestamp("2026-04-01"), pd.Series({"AAA": 0.97, "BBB": 1.2})) == []
