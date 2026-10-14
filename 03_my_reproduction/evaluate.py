"""VOC2007 evaluation for a trained FisherNet checkpoint.

The default ``svm`` protocol follows the paper's final evaluation stage:
extract Fisher vectors, apply signed-square-root + L2 normalization, train one
linear SVM per VOC class on trainval, then report VOC2007 11-point AP on test.
"""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from torch.utils.data import DataLoader

from models.factory import build_model as create_model
from src.voc_dataset import ResizeLongestSide, VOCMultiLabel, VOC_CLASSES
from src.training import read_config, assert_model_compatible, training_split


def voc2007_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute the official VOC2007 11-point interpolated average precision."""
    # In VOC ImageSets/Main files: 1=positive, -1=negative, 0=difficult.
    # Only difficult examples are excluded from AP.
    valid = labels != 0
    labels, scores = labels[valid], scores[valid]
    positives = labels == 1
    if not positives.any():
        return float("nan")
    order = np.argsort(-scores)
    positives = positives[order]
    true_positives = np.cumsum(positives)
    false_positives = np.cumsum(~positives)
    recall = true_positives / positives.sum()
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    return float(np.mean([
        precision[recall >= threshold].max() if np.any(recall >= threshold) else 0.0
        for threshold in np.arange(0.0, 1.1, 0.1)
    ]))


def load_raw_voc_labels(voc_root: Path, split: str, image_ids: list[str]) -> np.ndarray:
    """Load VOC's {-1, 0, 1} labels while preserving difficult examples."""
    tables = {}
    for class_name in VOC_CLASSES:
        path = voc_root / "ImageSets" / "Main" / f"{class_name}_{split}.txt"
        tables[class_name] = {
            fields[0]: int(fields[1])
            for line in path.read_text().splitlines()
            if (fields := line.split())
        }
    return np.asarray(
        [[tables[class_name][image_id] for class_name in VOC_CLASSES] for image_id in image_ids],
        dtype=np.int8,
    )


def normalize_fisher_vectors(features: np.ndarray) -> np.ndarray:
    """Use the standard signed-square-root and L2 FV normalization."""
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Fisher vectors must be a finite [images, dimensions] matrix")
    normalized = np.abs(features)
    np.sqrt(normalized, out=normalized)
    np.copysign(normalized, features, out=normalized)
    # Accumulate in float64 without copying the entire feature matrix. A
    # finite float32 FV can overflow the old squared-norm sum to infinity,
    # silently returning a zero row after division.
    norms = np.sqrt(np.einsum("ij,ij->i", normalized, normalized,
                             dtype=np.float64))[:, None]
    normalized /= np.maximum(norms, 1e-12)
    return normalized


@torch.inference_mode()
def extract(model, dataset, device: torch.device, batch_size: int, num_workers: int, mode: str):
    if mode == "fisher" and batch_size != 1:
        raise ValueError("multi-scale variable-resolution evaluation requires batch_size=1")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    outputs, image_ids = [], []
    model.eval()
    uses_original_coordinates = model.patch_coordinate_mode == "original"
    for batch in loader:
        if uses_original_coordinates:
            images, _, ids, original_sizes = batch
            original_sizes = original_sizes.to(device, non_blocking=True)
        else:
            images, _, ids = batch
            original_sizes = None
        images = images.to(device, non_blocking=True)
        values = (
            model.extract_fisher_vectors(images, original_sizes)
            if mode == "fisher"
            else model(images, original_sizes)
        )
        values = values.cpu().numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError("non-finite evaluation features/scores")
        outputs.append(values)
        image_ids.extend(ids)
    return np.concatenate(outputs, axis=0), image_ids


def extract_multiscale_fisher_vectors(model, dataset_root: Path, split: str, cfg: dict, device: torch.device, batch_size: int):
    """Average raw Fisher vectors from the five paper-specified image scales."""
    if not cfg["image_scales"]:
        raise ValueError("evaluation image_scales must not be empty")
    feature_sum = None
    reference_ids = None
    for longest_side in cfg["image_scales"]:
        dataset = VOCMultiLabel(
            dataset_root,
            year="2007",
            split=split,
            transform=ResizeLongestSide(longest_side),
            return_original_size=cfg.get("patch_coordinate_mode", "resized") == "original",
        )
        features, image_ids = extract(model, dataset, device, batch_size, cfg["num_workers"], "fisher")
        if reference_ids is None:
            reference_ids = image_ids
            feature_sum = features
        elif image_ids != reference_ids:
            raise RuntimeError("image ordering changed between evaluation scales")
        else:
            if features.shape != feature_sum.shape:
                raise RuntimeError("feature shape changed between evaluation scales")
            np.add(feature_sum, features, out=feature_sum)
        del features
        print(f"extracted_scale={longest_side}; split={split}; images={len(image_ids)}", flush=True)
    feature_sum /= len(cfg["image_scales"])
    return feature_sum, reference_ids


def build_model(cfg: dict, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert_model_compatible(checkpoint, cfg)
    restored_cfg = dict(cfg)
    if "backbone_config" in checkpoint["config"]:
        restored_cfg["backbone_config"] = checkpoint["config"]["backbone_config"]
    model = create_model(restored_cfg, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.checkpoint_metadata = {
        "step": checkpoint.get("step"),
        "train_split": checkpoint.get("training_contract", {}).get(
            "train_split", training_split(checkpoint["config"])),
        **{key: bool(checkpoint.get(key, False))
           for key in ("debug_run", "diagnostic_run", "preparation_run")},
    }
    return model


def evaluation_splits(cfg, split="test", fit_split=None, checkpoint_metadata=None):
    if split not in {"val", "test"}:
        raise ValueError("evaluation split must be val or test")
    fit_split = fit_split or training_split(cfg)
    if fit_split not in {"train", "trainval"}:
        raise ValueError("SVM fit split must be train or trainval")
    if split == "val":
        if training_split(cfg) != "train" or fit_split != "train":
            raise ValueError("held-out val requires train-only whole/GMM/Fisher and SVM fitting")
        if checkpoint_metadata is not None and checkpoint_metadata["train_split"] != "train":
            raise ValueError("checkpoint saw validation data; val cannot be claimed held-out")
    return fit_split, split


def report_ap(labels: np.ndarray, scores: np.ndarray) -> tuple[dict[str, float], float]:
    per_class = {
        class_name: voc2007_ap(labels[:, index], scores[:, index])
        for index, class_name in enumerate(VOC_CLASSES)
    }
    return per_class, float(np.nanmean(list(per_class.values())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", choices=("svm", "head"), default="svm")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--fit-split", choices=("train", "trainval"), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default="../05_results/evaluation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    cfg = read_config(args.config, project_root)
    fit_split, evaluation_split = evaluation_splits(cfg, args.split, args.fit_split)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    checkpoint_path = project_root / args.checkpoint
    output_dir = (project_root / args.output_dir).resolve()
    metrics_path = output_dir / f"voc2007_{args.protocol}_metrics.json"
    predictions_path = output_dir / f"voc2007_{args.protocol}_predictions.npz"
    if metrics_path.exists() or predictions_path.exists():
        raise FileExistsError("evaluation results already exist; use a NEW output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size or cfg.get("evaluation_batch_size", 1)
    dataset_root = (project_root / cfg["dataset_root"]).resolve()
    voc_root = dataset_root / "VOC2007"

    model = build_model(cfg, checkpoint_path, device)
    evaluation_splits(cfg, args.split, args.fit_split, model.checkpoint_metadata)
    svm_diagnostics = {}
    if args.protocol == "head":
        test_dataset = VOCMultiLabel(
            dataset_root,
            year="2007",
            split=evaluation_split,
            return_original_size=cfg.get("patch_coordinate_mode", "resized") == "original",
        )
        logits, test_ids = extract(model, test_dataset, device, batch_size, cfg["num_workers"], "head")
        # AP uses ordering; sigmoid saturation can create artificial ties.
        test_scores = logits
    else:
        train_features, train_ids = extract_multiscale_fisher_vectors(
            model, dataset_root, fit_split, cfg, device, batch_size
        )
        test_features, test_ids = extract_multiscale_fisher_vectors(
            model, dataset_root, evaluation_split, cfg, device, batch_size
        )
        train_features = normalize_fisher_vectors(train_features)
        test_features = normalize_fisher_vectors(test_features)
        if set(train_ids) & set(test_ids):
            raise ValueError("SVM fitting/evaluation image IDs overlap")
        train_labels = load_raw_voc_labels(voc_root, fit_split, train_ids)
        test_scores = np.empty((len(test_ids), len(VOC_CLASSES)), dtype=np.float32)
        for index, class_name in enumerate(VOC_CLASSES):
            valid = train_labels[:, index] != 0
            svm = LinearSVC(C=cfg["svm_C"], max_iter=10_000, dual="auto", random_state=cfg["seed"])
            with warnings.catch_warnings(record=True) as fit_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                svm.fit(train_features[valid], train_labels[valid, index])
            convergence_warnings = [str(item.message) for item in fit_warnings
                                    if issubclass(item.category, ConvergenceWarning)]
            for item in fit_warnings:
                if not issubclass(item.category, ConvergenceWarning):
                    warnings.warn(str(item.message), item.category)
            svm_diagnostics[class_name] = {
                "iterations": int(svm.n_iter_), "max_iter": svm.max_iter,
                "converged": not convergence_warnings,
                "convergence_warnings": convergence_warnings,
            }
            test_scores[:, index] = svm.decision_function(test_features)
            print(f"trained_svm={class_name}; iterations={svm.n_iter_}; "
                  f"converged={not convergence_warnings}", flush=True)

    test_labels = load_raw_voc_labels(voc_root, evaluation_split, test_ids)
    per_class, mean_ap = report_ap(test_labels, test_scores)
    np.savez_compressed(
        predictions_path,
        image_ids=np.asarray(test_ids), scores=test_scores, labels=test_labels,
    )
    result = {
        "protocol": args.protocol,
        "ap_protocol": "VOC2007 11-point interpolated AP",
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": model.checkpoint_metadata,
        "evaluation_split": evaluation_split,
        "svm_fit_split": fit_split if args.protocol == "svm" else None,
        "svm_solver_diagnostics": svm_diagnostics,
        "svm_all_converged": all(v["converged"] for v in svm_diagnostics.values())
                              if args.protocol == "svm" else None,
        "interpretation": "held-out validation" if evaluation_split == "val" else "fixed final test protocol",
        "config": cfg,
        "fusion": "mean of RAW scale FVs, then signed sqrt + L2" if args.protocol == "svm" else "single center crop head",
        "mAP": mean_ap,
        "per_class_AP": per_class,
    }
    metrics_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
