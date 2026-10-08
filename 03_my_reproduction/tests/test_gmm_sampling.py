import json

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from src.gmm_sampling import (sample_patch_indices, save_descriptor_cache, load_descriptor_cache,
                              descriptor_metadata, gmm_diagnostics, save_diagnostics)


def test_budget_covers_all_images_instead_of_first_large_patch_set():
    rng = np.random.default_rng(42)
    selected = [sample_patch_indices(2000, index, 10, 103, rng) for index in range(10)]
    assert sum(len(indices) for indices in selected) == 103
    assert [len(indices) for indices in selected] == [11] * 3 + [10] * 7
    assert all(len(np.unique(indices)) == len(indices) for indices in selected)
    assert any(indices.max() > 1000 for indices in selected)
    again = sample_patch_indices(2000, 0, 10, 103, np.random.default_rng(42))
    np.testing.assert_array_equal(again, selected[0])


def test_short_images_are_not_duplicated_or_over_budget():
    assert len(sample_patch_indices(3, 0, 10, 100, np.random.default_rng(1))) == 3
    assert len(sample_patch_indices(200, 5, 10, 3, np.random.default_rng(1))) == 0
    with pytest.raises(ValueError):
        sample_patch_indices(3, 10, 10, 100, np.random.default_rng(1))


def test_cache_roundtrip_and_config_mismatch(tmp_path):
    samples = np.arange(12, dtype=np.float32).reshape(6, 2)
    meta = {"source_sha256": "abc", "feature_config": {"descriptor_dim": 2}, "seed": 42}
    path = tmp_path / "samples.npz"
    save_descriptor_cache(path, samples, meta, ["a", "b"], [3, 3])
    actual, ids, counts = load_descriptor_cache(path, meta)
    np.testing.assert_array_equal(actual, samples)
    assert ids == ["a", "b"] and counts == [3, 3]
    with pytest.raises(FileExistsError):
        save_descriptor_cache(path, samples, meta, ids, counts)
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_descriptor_cache(path, dict(meta, source_sha256="different"))
    with pytest.raises(ValueError, match="invalid"):
        save_descriptor_cache(tmp_path / "invalid.npz", samples, meta, ids, [2, 3])


def test_cache_identity_tracks_source_but_allows_em_iteration_retry(tmp_path):
    path = tmp_path / "whole.pt"
    path.write_bytes(b"weights_v1")
    cfg = dict(seed=42, descriptor_dim=2, dataset_root="unused", gmm_max_iter=200,
               image_scales=[480], gmm_sampling_mode="uniform_per_image")
    original = descriptor_metadata(path, cfg, 10, 100)
    retry = descriptor_metadata(path, dict(cfg, gmm_max_iter=500), 10, 100)
    assert original == retry
    assert original == descriptor_metadata(path, dict(cfg, train_split="trainval"), 10, 100)
    train_only = descriptor_metadata(path, dict(cfg, train_split="train"), 10, 100)
    assert train_only != original and train_only["training_split"] == "train"
    changed = descriptor_metadata(path, dict(cfg, image_scales=[1200]), 10, 100)
    assert original != changed
    path.write_bytes(b"weights_v2")
    assert original != descriptor_metadata(path, cfg, 10, 100)


def test_gmm_diagnostics_measure_variance_and_component_coverage(tmp_path):
    rng = np.random.default_rng(42)
    samples = np.concatenate([rng.normal(-4, 0.2, (40, 2)), rng.normal(4, 0.3, (40, 2))])
    gmm = GaussianMixture(2, covariance_type="diag", random_state=42).fit(samples)
    report = gmm_diagnostics(gmm, samples, {"mode": "test"}, ["a", "b"], [40, 40])
    assert report["converged"]
    assert sum(report["component_sample_counts"]) == len(samples)
    assert report["image_count"] == 2 and report["sample_count"] == 80
    assert report["variance_quantiles"]["0"] > 0
    assert 0 <= report["fraction_near_variance_floor"] <= 1
    save_diagnostics(tmp_path / "diagnostics.json", report)
    assert json.loads((tmp_path / "diagnostics.json").read_text())["iterations"] == gmm.n_iter_
