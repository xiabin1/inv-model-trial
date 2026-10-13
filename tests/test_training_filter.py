import pandas as pd

from src.walk_forward import trainable


def test_training_leaves_out_moves_beyond_40_percent():
    rows = pd.DataFrame({"symbol": ["CTVA", "AAA", "MRNA", "BBB"], "date": pd.to_datetime(["2026-09-30"] * 4),
                         "future_return": [-0.81, 0.02, 0.84, -0.39], "label": [0, 1, 1, 0]})
    kept = trainable(rows)
    assert kept["symbol"].tolist() == ["AAA", "BBB"]
    assert trainable(rows, max_move=0.9)["symbol"].tolist() == ["CTVA", "AAA", "MRNA", "BBB"]
