import sys
from pathlib import Path

import torch
import pytest

import train
from src.training import read_config


def test_dense_training_entrypoint_and_resume(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    cfg = read_config(root / "tests/fixtures/audit_smoke.yaml", root)
    cfg.update(device="cpu", model_variant="dense")
    monkeypatch.setattr(train, "read_config", lambda *args: dict(cfg))
    output = tmp_path / "dense.pt"
    diagnostics = tmp_path / "dense.jsonl"
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "unused", "--smoke-test",
                                    "--max-steps", "1", "--output", str(output),
                                    "--diagnostics-jsonl", str(diagnostics)])
    train.main()
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 1
    assert checkpoint["training_contract"]["base_lrs"][2] == 0.0001
    assert all(torch.isfinite(value).all() for value in checkpoint["model"].values())
    assert '"event": "before_update"' in diagnostics.read_text()
    checkpoint["init_checkpoint"] = "checkpoints/original_gmm_source.pt"
    torch.save(checkpoint, output)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "unused", "--smoke-test",
                                    "--max-steps", "2", "--resume", str(output), "--output", str(output)])
    train.main()
    resumed = torch.load(output, map_location="cpu", weights_only=False)
    assert resumed["step"] == 2
    assert resumed["init_checkpoint"] == "checkpoints/original_gmm_source.pt"


def test_preparation_checkpoint_is_marked_and_cannot_resume_formally(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    cfg = read_config(root / "tests/fixtures/audit_smoke.yaml", root)
    cfg.update(device="cpu", model_variant="dense")
    monkeypatch.setattr(train, "read_config", lambda *args: dict(cfg))
    output = tmp_path / "preparation.pt"
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "unused", "--smoke-test",
                                    "--max-steps", "1", "--preparation", "--diagnostics-every", "10",
                                    "--stop-loss-above", "10000", "--output", str(output)])
    train.main()
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert checkpoint["preparation_run"] and checkpoint["diagnostic_run"]
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "unused", "--smoke-test",
                                    "--max-steps", "2", "--resume", str(output), "--output", str(output)])
    with pytest.raises(ValueError, match="diagnostic-only"):
        train.main()
    assert torch.load(output, map_location="cpu", weights_only=False)["step"] == 1


def test_warning_snapshot_is_pre_update_and_does_not_change_training(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    cfg = read_config(root / "tests/fixtures/audit_smoke.yaml", root)
    cfg.update(device="cpu", model_variant="dense")
    monkeypatch.setattr(train, "read_config", lambda *args: dict(cfg))
    monkeypatch.setattr(train, "masked_sigmoid_loss", lambda logits, *args: logits.sum()*0 + 101.)
    output, evidence = tmp_path / "out.pt", tmp_path / "warning.pt"
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "unused", "--smoke-test",
                                    "--preparation", "--max-steps", "1", "--output", str(output),
                                    "--warning-evidence", str(evidence)])
    train.main()
    captured = torch.load(evidence, weights_only=False)
    assert captured["step"] == 0 and captured["probe_step"] == 1
    assert len(captured["batches"]) == 1 and captured["diagnostic_run"]
    assert captured["batches"][0]["images"].device.type == "cpu"
    baseline = tmp_path / "baseline.pt"
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "unused", "--smoke-test",
                                    "--preparation", "--max-steps", "1", "--output", str(baseline)])
    train.main()
    a, b = torch.load(output, weights_only=False), torch.load(baseline, weights_only=False)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
