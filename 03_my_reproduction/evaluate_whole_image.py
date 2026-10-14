"""Evaluate the whole-image fine-tuning baseline on VOC2007."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.training import read_config, assert_model_compatible, training_split

from evaluate import load_raw_voc_labels, report_ap
from models.factory import build_model
from src.voc_dataset import VOCMultiLabel


def whole_image_score_report(logits, labels):
    """Rank by logits: float32 sigmoid can merge distinct extreme scores."""
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels)
    if logits.shape != labels.shape or logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError("finite logits and matching [images, classes] labels required")
    per_class, mean_ap = report_ap(labels, logits)
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
    _, sigmoid_ap = report_ap(labels, probabilities)
    valid = torch.from_numpy(labels != 0)
    targets = torch.from_numpy((labels == 1).astype(np.float32))
    elements = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits), targets, reduction="none") * valid
    return dict(
        mAP=mean_ap, per_class_AP=per_class, score_type="raw logits",
        sigmoid_mAP_diagnostic=sigmoid_ap,
        sigmoid_saturated_entries=int(((probabilities == 0) | (probabilities == 1)).sum()),
        mean_class_sum_bce=float(elements.sum() / len(logits)),
        global_valid_mean_bce=float(elements.sum() / valid.sum().clamp_min(1)),
        logits_absmax=float(np.max(np.abs(logits))),
    )


def split_interpretation(source_split, evaluation_split):
    if evaluation_split == "test":
        return "fixed-checkpoint attribution evaluation; do not tune on test"
    if source_split == "train" and evaluation_split == "val":
        return "held-out validation"
    if source_split == "train" and evaluation_split == "trainval":
        return "mixed seen-train/unseen-val diagnostic, NOT wholly held-out"
    return "seen-data diagnostic, NOT held-out validation"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="../05_results/whole_image_evaluation")
    parser.add_argument("--split", choices=("train", "val", "trainval", "test"), default="test")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    cfg = read_config(args.config, project_root)
    output_dir = (project_root / args.output_dir).resolve()
    result_path = output_dir / "voc2007_whole_image_metrics.json"
    predictions_path = output_dir / "voc2007_whole_image_predictions.npz"
    if result_path.exists() or predictions_path.exists():
        raise FileExistsError("evaluation results already exist; choose a new output directory")
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    checkpoint_path = project_root / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert_model_compatible(checkpoint, cfg, whole_image=True)
    checkpoint_step = checkpoint["step"]
    source_split = checkpoint.get("training_contract", {}).get(
        "train_split", training_split(checkpoint["config"]))
    provenance = {key: bool(checkpoint.get(key, False)) for key in
                  ("debug_run", "diagnostic_run", "preparation_run")}
    restored_cfg = dict(cfg)
    if "backbone_config" in checkpoint["config"]:
        restored_cfg["backbone_config"] = checkpoint["config"]["backbone_config"]
    model = build_model(restored_cfg, "whole_image", pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    model.eval()

    dataset_root = (project_root / cfg["dataset_root"]).resolve()
    dataset = VOCMultiLabel(dataset_root, year="2007", split=args.split, return_valid_mask=True)
    loader = DataLoader(
        dataset,
        batch_size=cfg["whole_image_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )
    scores, image_ids = [], []
    with torch.inference_mode():
        for images, _, _, ids in loader:
            scores.append(model(images.to(device)).cpu().numpy())
            image_ids.extend(ids)
    labels = load_raw_voc_labels(dataset_root / "VOC2007", args.split, image_ids)
    logits = np.concatenate(scores)
    score_report = whole_image_score_report(logits, labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(predictions_path, image_ids=np.asarray(image_ids), scores=logits, labels=labels)
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "checkpoint_train_split": source_split,
        "checkpoint_provenance": provenance,
        "config": cfg,
        "protocol": "single center-crop whole-image head",
        "ap_protocol": "VOC2007 11-point interpolated AP",
        "split": args.split,
        "interpretation": split_interpretation(source_split, args.split),
        **score_report,
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
