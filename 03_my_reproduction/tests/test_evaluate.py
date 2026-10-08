import numpy as np
import pytest
from pathlib import Path
import torch
import json
import sys
import warnings
from types import SimpleNamespace
from sklearn.exceptions import ConvergenceWarning

from evaluate import normalize_fisher_vectors, voc2007_ap
import evaluate
from evaluate_whole_image import whole_image_score_report, split_interpretation


def test_validation_requires_train_only_pipeline_and_matching_checkpoint():
    assert evaluate.evaluation_splits({}) == ("trainval", "test")
    cfg = {"train_split": "train"}
    assert evaluate.evaluation_splits(cfg, "val", checkpoint_metadata={"train_split": "train"}) == ("train", "val")
    with pytest.raises(ValueError, match="held-out"):
        evaluate.evaluation_splits({}, "val")
    with pytest.raises(ValueError, match="held-out"):
        evaluate.evaluation_splits(cfg, "val", "trainval")
    with pytest.raises(ValueError, match="saw validation"):
        evaluate.evaluation_splits(cfg, "val", checkpoint_metadata={"train_split": "trainval"})


@pytest.mark.parametrize("converged", [True, False])
def test_svm_validation_entrypoint_records_splits_convergence_and_protects_results(tmp_path, monkeypatch, converged):
    cfg = dict(train_split="train", device="cpu", dataset_root="unused",
               evaluation_batch_size=1, num_workers=0, svm_C=1., seed=42)
    monkeypatch.setattr(evaluate, "read_config", lambda *a: cfg)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    metadata = dict(train_split="train", step=100, diagnostic_run=True)
    monkeypatch.setattr(evaluate, "build_model", lambda *a: SimpleNamespace(checkpoint_metadata=metadata))
    splits = []
    def extract_stub(model, root, split, *a):
        splits.append(split)
        return np.array([[2., 1.], [-2., -1.]], dtype=np.float32), [f"{split}_0", f"{split}_1"]
    monkeypatch.setattr(evaluate, "extract_multiscale_fisher_vectors", extract_stub)
    monkeypatch.setattr(evaluate, "load_raw_voc_labels", lambda *a: np.tile([[1], [-1]], (1, 20)))
    class FakeSVM:
        def __init__(self, **kwargs):
            self.max_iter = kwargs["max_iter"]
        def fit(self, *a):
            self.n_iter_ = 2 if converged else self.max_iter
            if not converged:
                warnings.warn("solver did not converge", ConvergenceWarning)
        def decision_function(self, values):
            return values[:, 0]
    monkeypatch.setattr(evaluate, "LinearSVC", FakeSVM)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--config", "unused", "--checkpoint", "unused.pt",
                                     "--split", "val", "--output-dir", str(tmp_path)])
    evaluate.main()
    metrics = json.loads((tmp_path / "voc2007_svm_metrics.json").read_text(encoding="utf-8"))
    assert splits == ["train", "val"]
    assert metrics["evaluation_split"] == "val" and metrics["svm_fit_split"] == "train"
    assert metrics["checkpoint_metadata"]["diagnostic_run"]
    assert metrics["svm_all_converged"] is converged
    assert len(metrics["svm_solver_diagnostics"]) == 20
    with pytest.raises(FileExistsError, match="already exist"):
        evaluate.main()


def test_whole_image_split_labels_do_not_claim_seen_val_is_held_out():
    assert split_interpretation("train", "val") == "held-out validation"
    assert "NOT held-out" in split_interpretation("trainval", "val")
    assert "NOT wholly held-out" in split_interpretation("train", "trainval")


def test_voc2007_ap_perfect_ranking_is_one():
    labels = np.array([1, -1, 0, 1, -1])
    scores = np.array([0.9, 0.4, 1.0, 0.8, 0.1])
    assert voc2007_ap(labels, scores) == 1.0


def test_fisher_normalization_has_unit_norm():
    values = normalize_fisher_vectors(np.array([[4.0, -9.0], [0.0, 0.0]], dtype=np.float32))
    assert np.isclose(np.linalg.norm(values[0]), 1.0)
    assert np.array_equal(values[1], np.zeros(2))


def test_fisher_normalization_does_not_zero_finite_extreme_vectors():
    features = np.full((2, 16384), 1e38, dtype=np.float32)
    features[0, ::2] *= -1
    features[1] = 0
    with np.errstate(over="raise", invalid="raise"):
        normalized = normalize_fisher_vectors(features)
    assert normalized.dtype == np.float32
    assert np.linalg.norm(normalized[0].astype(np.float64)) == pytest.approx(1.)
    assert normalized[0, 0] < 0 and normalized[0, 1] > 0
    assert not normalized[1].any()


def test_voc2007_ap_known_nonperfect_ranking_and_difficult_ignore():
    # Ranking neg,pos,pos -> precision 2/3 at every 11-point threshold.
    assert np.isclose(voc2007_ap(np.array([-1, 1, 1]), np.array([3., 2., 1.])), 2 / 3)
    assert np.isclose(voc2007_ap(np.array([0, -1, 1, 1]), np.array([99., 3., 2., 1.])), 2 / 3)


def test_streamed_scale_fusion_matches_raw_stack_mean(monkeypatch):
    rng = np.random.default_rng(42)
    matrices = [rng.normal(size=(3, 8)).astype(np.float32) for _ in range(5)]
    expected = np.mean(np.stack(matrices), axis=0)
    iterator = iter(matrices)
    monkeypatch.setattr(evaluate, "VOCMultiLabel", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate, "extract", lambda *args: (next(iterator).copy(), ["a", "b", "c"]))
    actual, ids = evaluate.extract_multiscale_fisher_vectors(None, Path("unused"), "test",
                                                           {"image_scales": [480, 576, 688, 864, 1200], "num_workers": 0},
                                                           torch.device("cpu"), 1)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert ids == ["a", "b", "c"]


def test_scale_fusion_rejects_changed_image_order(monkeypatch):
    outputs = iter([(np.ones((2, 2), dtype=np.float32), ["a", "b"]),
                    (np.ones((2, 2), dtype=np.float32), ["b", "a"])])
    monkeypatch.setattr(evaluate, "VOCMultiLabel", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate, "extract", lambda *args: next(outputs))
    with pytest.raises(RuntimeError, match="ordering"):
        evaluate.extract_multiscale_fisher_vectors(None, Path("unused"), "test",
                                                  {"image_scales": [480, 576], "num_workers": 0},
                                                  torch.device("cpu"), 1)


def test_raw_logits_preserve_ranking_lost_to_float32_sigmoid():
    logits = np.array([[17.], [18.]], dtype=np.float32)
    assert torch.sigmoid(torch.from_numpy(logits)).unique().numel() == 1
    labels = np.array([[1], [-1]])
    # Use the 20-class report's expected dimensionality; each class has the same ranking.
    report = whole_image_score_report(np.repeat(logits, 20, axis=1), np.repeat(labels, 20, axis=1))
    assert report["mAP"] == pytest.approx(.5)
    assert report["sigmoid_mAP_diagnostic"] != report["mAP"]
    assert report["sigmoid_saturated_entries"] == 40
    assert np.isfinite(report["mean_class_sum_bce"])


def test_whole_report_excludes_difficult_and_rejects_nonfinite_scores():
    labels = np.tile(np.array([[1], [-1], [0]]), (1, 20))
    logits = np.tile(np.array([[2.], [1.], [1000.]], dtype=np.float32), (1, 20))
    report = whole_image_score_report(logits, labels)
    assert report["mAP"] == 1
    assert report["global_valid_mean_bce"] * (40 / 3) == pytest.approx(report["mean_class_sum_bce"], rel=1e-6)
    logits[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        whole_image_score_report(logits, labels)
