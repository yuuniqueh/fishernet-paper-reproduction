"""Shared loss, optimizer groups and checkpoint contracts for both stages.

Historical checkpoints without a contract used valid_mean and the classifier
rate for the descriptor layer. They must not silently resume a corrected run.
"""

import math
from pathlib import Path

import torch
from torch.nn import functional as F
import yaml


def read_config(path, project_root):
    candidate = Path(path)
    if not candidate.is_absolute() and not candidate.is_file():
        candidate = Path(project_root) / candidate
    config = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    return config


def masked_sigmoid_loss(logits, targets, valid, reduction="batch_sum"):
    """Eq. (11): sum valid classes per image, then average across images.

    valid_mean is retained ONLY for replaying historical experiments. Difficult
    labels are excluded without normalizing by their changing label count.
    BCEWithLogits avoids explicitly taking logs of saturated sigmoid values.
    """
    if logits.ndim != 2 or logits.shape != targets.shape or logits.shape != valid.shape:
        raise ValueError("logits, targets and valid must have identical [B, C] shapes")
    if logits.shape[0] == 0:
        raise ValueError("empty batch")
    elements = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    total = (elements * valid).sum()
    if reduction == "batch_sum":
        return total / logits.shape[0]
    if reduction == "valid_mean":
        return total / valid.sum().clamp_min(1.0)
    raise ValueError(f"unknown loss_reduction: {reduction}")


def stage_loss_reduction(cfg, stage, historical=False):
    """Allow an explicitly declared stage intervention, never an implicit mismatch."""
    reduction = cfg.get("loss_reduction", "valid_mean" if historical else "batch_sum")
    if stage == "whole_image":
        reduction = cfg.get("whole_image_loss_reduction", reduction)
    elif stage != "fisher":
        raise ValueError(f"unknown training stage: {stage}")
    if reduction not in {"batch_sum", "valid_mean"}:
        raise ValueError("loss_reduction must be batch_sum or valid_mean")
    return reduction


def training_split(cfg):
    """Default benchmark protocol stays trainval; validation trials use train."""
    split = cfg.get("train_split", "trainval")
    if split not in {"train", "trainval"}:
        raise ValueError("train_split must be train or trainval; val/test must not enter training")
    return split


def assert_initialization_split_compatible(checkpoint, cfg, gmm=False):
    expected = training_split(cfg)
    if training_split(checkpoint["config"]) != expected:
        raise ValueError("initialization training split mismatch; build a fresh matching source")
    saved = checkpoint.get("training_contract", {}).get("train_split", expected)
    if saved != expected:
        raise ValueError("initialization training split disagrees with saved contract")
    if gmm and training_split(checkpoint.get("source_whole_image_config", checkpoint["config"])) != expected:
        raise ValueError("upstream whole-image training split mismatch; validation data would leak")
    return expected


def assert_whole_source_loss_compatible(checkpoint, cfg, gmm=False):
    """Check the upstream whole stage, not the downstream Fisher experiment loss."""
    if gmm:
        source_cfg = checkpoint.get("source_whole_image_config", checkpoint["config"])
        source_loss = checkpoint.get("source_whole_image_loss_reduction",
                                     stage_loss_reduction(source_cfg, "whole_image", historical=True))
    else:
        source_loss = checkpoint.get("training_contract", {}).get(
            "loss_reduction", stage_loss_reduction(checkpoint["config"], "whole_image", historical=True))
    if source_loss != stage_loss_reduction(cfg, "whole_image"):
        raise ValueError("whole-image loss mismatch; rebuild initialization or explicitly declare its stage loss")
    return source_loss


def training_contract(cfg, stage, historical=False):
    """Record effective semantics, not just YAML fields or current decayed LR."""
    if stage not in {"whole_image", "fisher"}:
        raise ValueError(f"unknown training stage: {stage}")
    contract = {
        "stage": stage,
        "loss_reduction": stage_loss_reduction(cfg, stage, historical),
        "momentum": cfg["momentum"],
        "weight_decay": cfg["weight_decay"],
        "backbone": cfg["backbone"],
        "num_classes": cfg["num_classes"],
        "descriptor_dim": cfg["descriptor_dim"],
        "fc_dropout": cfg.get("fc_dropout", False),
        "new_layer_init_std": cfg.get("new_layer_init_std"),
        "seed": cfg["seed"],
        "dataset_root": cfg["dataset_root"],
        "train_split": training_split(cfg),
    }
    if contract["loss_reduction"] not in {"batch_sum", "valid_mean"}:
        raise ValueError("loss_reduction must be batch_sum or valid_mean")
    if stage == "whole_image":
        contract.update(
            base_lrs=[cfg["whole_image_lr_backbone"], cfg["whole_image_lr_backbone"],
                      cfg["whole_image_lr_new"], cfg["whole_image_lr_new"]],
            milestones=[cfg["whole_image_lr_step"], 2 * cfg["whole_image_lr_step"]],
            gamma=0.1,
            batch_size=cfg["whole_image_batch_size"],
        )
    else:
        old_milestones = [cfg["fisher_steps"] // 2, 3 * cfg["fisher_steps"] // 4]
        contract.update(
            base_lrs=[cfg["lr_backbone"], cfg["lr_backbone"],
                      cfg.get("lr_descriptor_projection", cfg["lr_classifier"] if historical else cfg["lr_backbone"]),
                      cfg["lr_fisher"], cfg["lr_classifier"]],
            milestones=list(cfg.get("fisher_lr_milestones", old_milestones if historical else [])),
            gamma=cfg.get("fisher_lr_gamma", 0.1),
            micro_batch=cfg.get("fisher_micro_batch_size", 1),
            accumulation=cfg.get("fisher_gradient_accumulation_steps", 2),
            model_variant=cfg.get("model_variant", "dense"),
            num_components=cfg["num_components"],
            patch_coordinate_mode=cfg.get("patch_coordinate_mode", "resized"),
            roi_coordinate_mapping=cfg.get("roi_coordinate_mapping", "legacy_ratio"),
            patch_scales=list(cfg["patch_scales"]),
            patch_step=cfg["patch_step"],
            max_patches=cfg.get("max_patches"),
            image_scales=list(cfg["image_scales"]),
        )
    if any(not math.isfinite(lr) or lr <= 0 for lr in contract["base_lrs"]):
        raise ValueError("learning rates must be positive and finite")
    return contract


def build_optimizer(model, cfg, stage):
    contract = training_contract(cfg, stage)
    patch_params = []
    if hasattr(model, "fc6"):
        patch_params = list(model.fc6.parameters()) + list(model.fc7.parameters())
    modules = [list(model.backbone.parameters()), patch_params,
               list(model.descriptor_projection.parameters())]
    names = ["backbone", "fc6_fc7", "descriptor_projection"]
    if stage == "fisher":
        modules.append(list(model.fisher.parameters()))
        names.append("fisher")
    modules.append(list(model.classifier.parameters()))
    names.append("classifier")
    seen = [id(parameter) for parameters in modules for parameter in parameters]
    expected = {id(parameter) for parameter in model.parameters()}
    if len(set(seen)) != len(seen) or set(seen) != expected:
        raise ValueError("optimizer groups must cover every model parameter exactly once")
    groups = [dict(params=parameters, lr=lr, name=name)
              for name, parameters, lr in zip(names, modules, contract["base_lrs"])]
    optimizer = torch.optim.SGD(groups, momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=contract["milestones"], gamma=contract["gamma"]
    )
    return optimizer, scheduler, contract


def assert_resume_compatible(checkpoint, cfg, stage):
    current = training_contract(cfg, stage)
    saved = checkpoint.get("training_contract")
    if saved is None:
        if "config" not in checkpoint:
            raise ValueError("resume checkpoint lacks config/contract; cannot verify optimizer semantics")
        saved = training_contract(checkpoint["config"], stage, historical=True)
    else:
        saved = dict(saved)
        # Contracts written before the initialization audit omitted this field.
        saved.setdefault("new_layer_init_std", checkpoint.get("config", {}).get("new_layer_init_std"))
        saved.setdefault("train_split", training_split(checkpoint.get("config", {})))
    differences = [key for key in current if current[key] != saved.get(key)]
    if differences:
        raise ValueError("incompatible resume settings: " + ", ".join(differences)
                         + "; start a NEW experiment instead of resuming old optimizer state")
    state_groups = checkpoint["optimizer"]["param_groups"]
    saved_lrs = checkpoint["scheduler"].get("base_lrs")
    if saved_lrs != current["base_lrs"] or len(state_groups) != len(current["base_lrs"]):
        raise ValueError("saved optimizer/scheduler LR groups disagree with configuration")
    if any(group["momentum"] != cfg["momentum"] or group["weight_decay"] != cfg["weight_decay"]
           for group in state_groups):
        raise ValueError("saved optimizer momentum/weight decay disagree with configuration")
    scheduler_state = checkpoint["scheduler"]
    saved_milestones = sorted(scheduler_state["milestones"].elements())
    if saved_milestones != sorted(current["milestones"]) or scheduler_state["gamma"] != current["gamma"]:
        raise ValueError("saved scheduler policy disagrees with configuration")


def assert_model_compatible(checkpoint, cfg, whole_image=False):
    """Don't accidentally assess or initialize a checkpoint with different RoIs."""
    if "config" not in checkpoint:
        raise ValueError("checkpoint lacks model configuration")
    saved = checkpoint["config"]
    keys = {"backbone": "alexnet", "num_classes": 20, "descriptor_dim": 256,
            "fc_dropout": False, "new_layer_init_std": None}
    if not whole_image:
        keys.update(model_variant="dense", num_components=32, patch_coordinate_mode="resized",
                    roi_coordinate_mapping="legacy_ratio", max_patches=None,
                    patch_scales=[64, 96, 128, 160, 192, 224, 256], patch_step=32)
    differences = [key for key, default in keys.items()
                   if saved.get(key, default) != cfg.get(key, default)]
    if differences:
        raise ValueError("checkpoint/model configuration mismatch: " + ", ".join(differences))


def protect_output(output, resume=None):
    """Existing experiment files are never overwritten by a fresh run."""
    output = Path(output)
    if output.exists() and (resume is None or output.resolve() != Path(resume).resolve()):
        raise FileExistsError(f"output already exists: {output}; choose a new experiment filename")
    output.parent.mkdir(parents=True, exist_ok=True)


def checked_loss_value(loss):
    value = loss.detach().item()
    if not math.isfinite(value):
        raise FloatingPointError("non-finite loss; stopping before optimizer update; checkpoint NOT overwritten")
    return value


def reject_diagnostic_checkpoint(checkpoint, diagnostic=False):
    """Local intervention evidence must not silently become formal training."""
    if checkpoint.get("diagnostic_run", False) and not diagnostic:
        raise ValueError("diagnostic-only checkpoint cannot initialize/resume formal training")


def ensure_finite_gradients(model):
    """Check the infinity norm before SGD, WITHOUT clipping finite gradients.

    A finite loss alone does not guarantee finite backpropagated derivatives.
    foreach kernels keep this a single grouped check instead of per-layer syncs.
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"),
                                         norm_type=float("inf"), error_if_nonfinite=True)


def ensure_finite_checkpoint_state(model, optimizer):
    """Never replace a checkpoint with non-finite parameters or SGD history.

    Finite gradients do not guarantee finite updated weights/momentum. Check at
    save time only, so normal training does not pay this cost on every batch.
    This is a persistence guard, not a solution to finite-gradient explosions.
    """
    tensors = list(model.state_dict().values())
    tensors += [value for state in optimizer.state.values() for value in state.values()
                if isinstance(value, torch.Tensor)]
    checks = [torch.isfinite(t).all() for t in tensors if torch.is_floating_point(t)]
    if checks and not bool(torch.stack(checks).all()):
        raise FloatingPointError("non-finite checkpoint state; checkpoint NOT overwritten")
