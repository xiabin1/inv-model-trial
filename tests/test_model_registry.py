import json

import src.models.model_registry as registry


def card(train_to, trained_at):
    return {"variant": "weekly_plain_label", "features_version": "b1542_v1", "train_from": "2024-10-01", "train_to": train_to,
            "train_rows": 1, "features": 2, "trained_at": trained_at}


def test_save_keeps_previous_and_new_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "HISTORY_DIR", tmp_path / "history")
    model_path = tmp_path / "live_classifier_b1542.joblib"

    first = registry.save({"model": 1}, model_path, card("2026-10-02", "2026-10-06 19:47 EDT"))
    second = registry.save({"model": 2}, model_path, card("2026-10-09", "2026-10-10 09:00 EDT"))

    history = sorted(p.name for p in (tmp_path / "history/live_classifier_b1542").iterdir())
    assert history == [
        "live_classifier_b1542__to-2026-10-02__2026-10-06T1947.joblib",
        "live_classifier_b1542__to-2026-10-02__2026-10-06T1947.json",
        "live_classifier_b1542__to-2026-10-09__2026-10-10T0900.joblib",
        "live_classifier_b1542__to-2026-10-09__2026-10-10T0900.json",
    ]
    assert first.name.startswith("live_classifier_b1542__to-2026-10-02") and second.name.endswith("T0900.joblib")
    assert json.loads(model_path.with_suffix(".json").read_text())["train_to"] == "2026-10-09"
    assert json.loads(second.with_suffix(".json").read_text()) == card("2026-10-09", "2026-10-10 09:00 EDT")


def test_previous_model_without_history_is_archived_before_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "HISTORY_DIR", tmp_path / "history")
    model_path = tmp_path / "live_classifier.joblib"
    model_path.write_bytes(b"old model")
    model_path.with_suffix(".json").write_text(json.dumps(card("2026-09-25", "2026-09-27 10:00 EDT")))

    registry.save({"model": "new"}, model_path, card("2026-10-02", "2026-10-04 10:00 EDT"))

    old = tmp_path / "history/live_classifier/live_classifier__to-2026-09-25__2026-09-27T1000.joblib"
    assert old.read_bytes() == b"old model"


def test_replay_models_are_not_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "HISTORY_DIR", tmp_path / "history")
    model_path = tmp_path / "paper_sim_classifier.joblib"
    assert registry.save({"m": 1}, model_path, card("2026-10-02", "2026-10-06 19:47 EDT"), keep_history=False) is None
    assert model_path.exists() and not (tmp_path / "history").exists()
