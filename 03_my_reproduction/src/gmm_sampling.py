"""Bounded GMM sampling, validated descriptor caches and fit diagnostics."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor_metadata(checkpoint_path, cfg, max_images, max_descriptors):
    keys = ("seed", "descriptor_dim", "patch_scales", "patch_step", "max_patches",
            "patch_coordinate_mode", "roi_coordinate_mapping", "image_scales",
            "fc_dropout", "new_layer_init_std", "backbone", "backbone_model_name", "backbone_config",
            "backbone_backend", "backbone_arch")
    return {
        "schema_version": 1,
        "source_checkpoint": str(Path(checkpoint_path).resolve()),
        "source_sha256": file_sha256(checkpoint_path),
        "extractor_sources": {name: file_sha256(Path(__file__).resolve().parents[1] / name)
                              for name in ("models/fishernet_spp.py", "models/factory.py",
                                           "models/modern_backbone.py", "src/patches.py", "src/voc_dataset.py")},
        "runtime": {name: version(name) for name in ("torch", "torchvision")},
        "feature_config": {key: cfg.get(key) for key in keys},
        "dataset_root": cfg["dataset_root"],
        "sampling_mode": cfg.get("gmm_sampling_mode", "legacy_prefix"),
        "max_images": max_images,
        "max_descriptors": max_descriptors,
        # Preserve legacy trainval cache identities, but never reuse one for
        # a train-only validation trial even when the source filename matches.
        **({"training_split": cfg["train_split"]} if cfg.get("train_split", "trainval") != "trainval" else {}),
    }


def sample_patch_indices(count, image_index, target_images, budget, rng):
    """Divide a fixed budget across images and sample uniformly without replacement.

    image_index is zero-based. Short images are not duplicated to fill a quota.
    Selecting boxes BEFORE RoI/FC avoids computing thousands of unused patches.
    """
    if count < 0 or target_images < 1 or budget < 1 or not 0 <= image_index < target_images:
        raise ValueError("invalid descriptor sampling inputs")
    quota, remainder = divmod(budget, target_images)
    quota += image_index < remainder
    size = min(count, quota)
    return np.sort(rng.choice(count, size=size, replace=False)).astype(np.int64)


def save_descriptor_cache(path, samples, metadata, image_ids, counts):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"descriptor cache already exists: {path}")
    if (samples.ndim != 2 or not np.isfinite(samples).all() or len(samples) != sum(counts)
            or len(image_ids) != len(counts) or any(count < 0 for count in counts)
            or samples.shape[1] != metadata["feature_config"]["descriptor_dim"]):
        raise ValueError("invalid descriptor cache arrays")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, samples=samples, metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                 image_ids=np.asarray(image_ids), counts=np.asarray(counts, dtype=np.int64))
    temporary.replace(path)


def load_descriptor_cache(path, expected_metadata):
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        if metadata != expected_metadata:
            raise ValueError("descriptor cache metadata mismatch; choose a NEW cache path")
        samples = cache["samples"].copy()
        image_ids = cache["image_ids"].tolist()
        counts = cache["counts"].tolist()
    if (samples.ndim != 2 or not np.isfinite(samples).all()
            or len(samples) != sum(counts) or len(image_ids) != len(counts)
            or any(count < 0 for count in counts)
            or samples.shape[1] != metadata["feature_config"]["descriptor_dim"]):
        raise ValueError("invalid descriptor cache arrays")
    return samples, image_ids, counts


def gmm_diagnostics(gmm, samples, metadata, image_ids, counts):
    variances = gmm.covariances_
    levels = [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]
    occupancy = np.bincount(gmm.predict(samples), minlength=gmm.n_components)
    return {
        "metadata": metadata,
        "converged": bool(gmm.converged_),
        "iterations": int(gmm.n_iter_),
        "lower_bound": float(gmm.lower_bound_),
        "reg_covar": gmm.reg_covar,
        "variance_quantiles": dict(zip(map(str, levels), map(float, np.quantile(variances, levels)))),
        "fraction_near_variance_floor": float(np.mean(variances <= gmm.reg_covar * 1.01)),
        "component_weights": gmm.weights_.tolist(),
        "component_sample_counts": occupancy.tolist(),
        "sample_count": len(samples),
        "image_count": len(image_ids),
        "image_ids": image_ids,
        "descriptors_per_image": counts,
    }


def save_diagnostics(path, diagnostics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
