"""Matched short diagnostic interventions, NOT a new paper/mAP experiment.

Every arm reloads the same finite checkpoint, including SGD momentum, then
uses the saved image/scale/flip plan. Geometry interventions deliberately use
the source GMM/weights unchanged to isolate their immediate effect; a real
geometry experiment must refit GMM and train from that initialization.
"""

import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import threading
import time
import traceback

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.fishernet_spp import FisherNetSPPAlexNet
from src.training import (read_config, build_optimizer, assert_resume_compatible,
                          checked_loss_value, ensure_finite_gradients,
                          ensure_finite_checkpoint_state, masked_sigmoid_loss, training_split)
from src.training_diagnostics import TrainingDiagnostics
from src.voc_dataset import VOCMultiLabel, ResizeLongestSide


ARM_OVERRIDES = {
    "control": {},
    "historical_loss": {"loss_reduction": "valid_mean"},
    "late_decay": {"diagnostic_lr_multiplier": 0.01},
    "late_decay_repeat": {"diagnostic_lr_multiplier": 0.01},
    "historical_projection_lr": {"diagnostic_projection_lr": 0.001},
    "boundary": {"roi_coordinate_mapping": "half_open_ratio"},
    "original_patches": {"roi_coordinate_mapping": "half_open_ratio",
                         "patch_coordinate_mode": "original"},
}


def apply_optimizer_intervention(optimizer, scheduler, overrides):
    """Explicit local LR interventions; never change source momentum or weights."""
    factor = overrides.get("diagnostic_lr_multiplier", 1.0)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("diagnostic LR multiplier must be positive and finite")
    for group in optimizer.param_groups:
        group["lr"] *= factor
        if "initial_lr" in group:
            group["initial_lr"] *= factor
    scheduler.base_lrs = [lr * factor for lr in scheduler.base_lrs]
    if "diagnostic_projection_lr" in overrides:
        value = overrides["diagnostic_projection_lr"]
        if not math.isfinite(value) or value <= 0:
            raise ValueError("diagnostic projection LR must be positive and finite")
        index = 2  # Both historical/current five-group Fisher contracts use this position.
        optimizer.param_groups[index]["lr"] = value
        optimizer.param_groups[index]["initial_lr"] = value
        scheduler.base_lrs[index] = value
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
    return list(scheduler._last_lr)


def make_batch_plan(image_ids, seed, steps, accumulation, scales):
    if steps <= 0 or accumulation <= 0 or not image_ids or not scales:
        raise ValueError("positive steps/accumulation and nonempty dataset/scales required")
    rng = random.Random(seed)
    order, records = [], []
    for ordinal in range(steps * accumulation):
        if not order:
            order = list(range(len(image_ids)))
            rng.shuffle(order)
        index = order.pop()
        records.append(dict(ordinal=ordinal, dataset_index=index, image_id=image_ids[index],
                            scale=rng.choice(scales), flip=rng.random() < .5))
    return records


class PlannedVOC(Dataset):
    def __init__(self, base, plan):
        self.base, self.plan = base, plan

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        record = self.plan[index]
        # Workers get independent dataset objects. No process-global RNG is
        # consumed by augmentation, regardless of prefetch/worker scheduling.
        self.base.transform = ResizeLongestSide(record["scale"], record["flip"])
        images, targets, valid, image_id, original_size = self.base[record["dataset_index"]]
        if image_id != record["image_id"]:
            raise ValueError("dataset identity differs from batch plan")
        return images, targets, valid, image_id, original_size, record["ordinal"]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batch_fingerprint(images, targets, valid, original_sizes):
    digest = hashlib.sha256()
    for tensor in (images, targets, valid, original_sizes):
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def paired_inputs_equal(arms):
    successful_prefix = None
    compared = 0
    for arm in arms.values():
        fingerprints = arm["batch_fingerprints"]
        if successful_prefix is not None:
            prefix = min(len(fingerprints), len(successful_prefix))
            if fingerprints[:prefix] != successful_prefix[:prefix]:
                return False, prefix
            compared = max(compared, prefix)
        else:
            successful_prefix = fingerprints
    return compared > 0, compared


def validate_resource_status(gpu_free_mib, ram_free_bytes, commit_free_bytes,
                             reserve_new_process=True, memory_calibration=False,
                             gpu_required_mib=None):
    """Admission only; never change training size to fit a busy machine.

    Windows event 2004 recorded a 13.57 GB main process plus loader workers
    in the previous failure. Physical RAM alone misses that commit limit.
    Reserve 16 GiB before this diagnostic's checkpoint/worker allocation.
    """
    if gpu_required_mib is None:
        # Opt-in diagnostic: the prior identical worst-plan calibration used
        # 4018 MiB reserved. 5632 leaves >1.5 GiB for context/extra headroom;
        # recalibrate in this process before proceeding. Default stays 6144.
        gpu_required_mib = 5632 if memory_calibration else 6144
    if gpu_required_mib <= 0:
        raise ValueError("positive GPU reservation required")
    if gpu_free_mib < gpu_required_mib:
        raise RuntimeError(f"need >={gpu_required_mib} MiB GPU free; got {gpu_free_mib}")
    if ram_free_bytes < 2 * 1024 ** 3:
        raise RuntimeError("need >=2 GiB physical RAM free")
    # Between arms this process reuses allocations; demanding another full
    # process budget would count its retained allocator memory twice.
    # Explicit user-authorised bounded calibration, NOT the old full-run gate.
    # mmap/packed labels remove needless allocations; the 8 GiB provisional
    # start budget must then pass two worst-plan updates and a measured 2 GiB
    # remaining commit floor before any 500-step arm is allowed to proceed.
    minimum_commit_gib = (8 if memory_calibration else 16) if reserve_new_process else 2
    if commit_free_bytes < minimum_commit_gib * 1024 ** 3:
        raise RuntimeError(f"need >={minimum_commit_gib} GiB Windows commit headroom for checkpoint/loader allocation")
    return dict(gpu_free_mib=gpu_free_mib, ram_free_bytes=ram_free_bytes,
                commit_free_bytes=commit_free_bytes)


def windows_memory_status():
    """Global Windows headroom; includes other apps and spawned loader workers."""
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total_physical", "available_physical", "total_page", "available_page",
             "total_virtual", "available_virtual", "extended")]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("cannot verify physical RAM/commit")
    return dict(ram_free_bytes=status.available_physical,
                commit_free_bytes=status.available_page)


class MemoryMonitor:
    """Sample system headroom throughout a bounded GPU diagnostic, not an allocator tweak."""
    def __init__(self, enabled, reclaim=None, ram_reclaim_below_bytes=0,
                 commit_reclaim_below_bytes=4 * 1024 ** 3):
        self.enabled = enabled
        self.stop = threading.Event()
        self.thread = None
        self.initial = None
        self.minimum = None
        self.reclaim = reclaim
        # Opt-in for qualified fixed-shape whole training. Existing diagnostic
        # cache policy stays unchanged unless this threshold is supplied.
        self.ram_reclaim_below_bytes = ram_reclaim_below_bytes
        self.commit_reclaim_below_bytes = commit_reclaim_below_bytes
        self.cache_releases = []

    def sample(self):
        if self.enabled:
            value = windows_memory_status()
            if self.initial is None:
                self.initial = value
                self.minimum = dict(value)
            else:
                for key in value:
                    self.minimum[key] = min(self.minimum[key], value[key])
            return value

    def guard(self):
        value = self.sample()
        # Dynamic input shapes can leave many unused CUDA allocator segments,
        # whose driver allocations also consume Windows commit. Reclaim only
        # unused segments under pressure; never gradients, parameters or data.
        pressure = value and (value["commit_free_bytes"] < self.commit_reclaim_below_bytes
                              or value["ram_free_bytes"] < self.ram_reclaim_below_bytes)
        if pressure and self.reclaim:
            released = self.reclaim()
            if released:
                after = self.sample()
                report = dict(before=value, after=after,
                              pressure_reason=("commit" if value["commit_free_bytes"] < self.commit_reclaim_below_bytes else "physical_ram"),
                              **released)
                self.cache_releases.append(report)
                print(f"resource_cache_release={report}", flush=True)
                value = after
        if value and (value["commit_free_bytes"] < 2 * 1024 ** 3
                      or value["ram_free_bytes"] < 256 * 1024 ** 2):
            raise RuntimeError(f"controlled diagnostic memory floor reached: {value}; no further update")
        return value

    def start(self):
        if self.enabled:
            self.sample()
            def poll():
                while not self.stop.wait(.02):
                    self.sample()
            self.thread = threading.Thread(target=poll, daemon=True)
            self.thread.start()

    def finish(self):
        if self.thread is not None:
            self.stop.set()
            self.thread.join()
        self.sample()
        return dict(initial=self.initial, minimum=self.minimum,
                    peak_added_commit_bytes=(self.initial["commit_free_bytes"] - self.minimum["commit_free_bytes"])
                    if self.initial else None, cache_releases=self.cache_releases)


def reclaim_unused_cuda_cache():
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    if reserved - allocated < 64 * 1024 ** 2:
        return None
    gc.collect()
    torch.cuda.empty_cache()
    return dict(cuda_allocated_before=allocated, cuda_reserved_before=reserved,
                cuda_allocated_after=torch.cuda.memory_allocated(),
                cuda_reserved_after=torch.cuda.memory_reserved())


def gpu_reservation_after_calibration(calibration):
    """Same live process: context already exists; reserve measured allocator peak + 512 MiB."""
    peak = calibration.get("cuda_peak_reserved_bytes")
    if (calibration.get("status") != "completed" or calibration.get("updates_completed") != 2
            or not isinstance(peak, int) or peak <= 0):
        raise ValueError("successful CUDA memory calibration required for in-process GPU reservation")
    return math.ceil(peak / 1024 ** 2) + 512


def resource_check(reserve_new_process=True, memory_calibration=False,
                   gpu_required_mib=None):
    """Local admission check; no parallel high-memory GPU work is launched."""
    import subprocess
    if gpu_required_mib is None:
        gpu_required_mib = 5632 if memory_calibration else 6144
    row = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"
    ], text=True).strip().splitlines()
    if len(row) != 1 or int(row[0]) < gpu_required_mib:
        raise RuntimeError(f"need a single GPU with >={gpu_required_mib} MiB free; got {row}")
    status = windows_memory_status()
    return validate_resource_status(int(row[0]), status["ram_free_bytes"],
                                    status["commit_free_bytes"], reserve_new_process,
                                    memory_calibration, gpu_required_mib)


def memory_calibration_plan(plan, image_sizes, cfg):
    """Repeat the two largest actual plan geometries; no patch/scale reduction."""
    def score(record):
        width, height = image_sizes[record["image_id"]]
        scale = record["scale"] / max(width, height)
        resized_height, resized_width = max(1, round(height * scale)), max(1, round(width * scale))
        if cfg.get("patch_coordinate_mode", "resized") == "original":
            patch_height, patch_width = height, width
        else:
            patch_height, patch_width = resized_height, resized_width
        count = sum(max(0, (patch_height - size) // cfg["patch_step"] + 1)
                    * max(0, (patch_width - size) // cfg["patch_step"] + 1)
                    for size in cfg["patch_scales"])
        if cfg.get("max_patches") is not None:
            count = min(count, cfg["max_patches"])
        return count, resized_height * resized_width
    largest = sorted(plan, key=score, reverse=True)[:cfg.get("fisher_gradient_accumulation_steps", 2)]
    return [dict(record, ordinal=index) for index, record in enumerate(largest * 2)]


@torch.no_grad()
def predicted_fisher_relative_update(model, optimizer):
    """Observe the configured SGD displacement before update; never alter gradients."""
    predictions = {}
    parameters = {id(p): name for name, p in model.fisher.named_parameters()}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) not in parameters or p.grad is None:
                continue
            if group.get("nesterov") or group.get("dampening", 0) != 0:
                raise ValueError("displacement observer supports this project's plain SGD only")
            direction = p.grad.detach() + group["weight_decay"] * p.detach()
            previous = optimizer.state[p].get("momentum_buffer")
            if previous is not None:
                direction = direction + group["momentum"] * previous
            norm = float(p.detach().norm(dtype=torch.float64))
            predictions[parameters[id(p)]] = float(direction.norm(dtype=torch.float64)) * group["lr"] / norm if norm > 0 else None
    return predictions


def validate_source(checkpoint, allow_preparation=False):
    if checkpoint.get("debug_run", False):
        raise ValueError("debug source is not permitted")
    if checkpoint.get("diagnostic_run", False) and not (
        allow_preparation and checkpoint.get("preparation_run", False)
    ):
        raise ValueError("diagnostic source requires explicit preparation-source permission")


def run_arm(name, cfg, source, directory, base_dataset, plan, args, device):
    efficient = getattr(args, "memory_efficient", False)
    memory = MemoryMonitor(efficient and device.type == "cuda", reclaim_unused_cuda_cache)
    memory.start()
    if memory.enabled:
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    checkpoint = torch.load(source, map_location="cpu", weights_only=False, mmap=efficient)
    assert_resume_compatible(checkpoint, cfg, "fisher")
    validate_source(checkpoint, getattr(args, "allow_preparation_source", False))
    source_step = checkpoint["step"]
    changed_cfg = {**cfg, **ARM_OVERRIDES[name]}
    model = FisherNetSPPAlexNet(
        num_classes=cfg["num_classes"], descriptor_dim=cfg["descriptor_dim"],
        num_components=cfg["num_components"], patch_sizes=tuple(cfg["patch_scales"]),
        patch_step=cfg["patch_step"], max_patches=cfg.get("max_patches"),
        patch_coordinate_mode=changed_cfg.get("patch_coordinate_mode", "resized"),
        roi_coordinate_mapping=changed_cfg.get("roi_coordinate_mapping", "legacy_ratio"),
        pretrained_backbone=False, fc_dropout=cfg.get("fc_dropout", False),
        new_layer_init_std=cfg.get("new_layer_init_std"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer, scheduler, contract = build_optimizer(model, cfg, "fisher")
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    effective_lrs = apply_optimizer_intervention(optimizer, scheduler, ARM_OVERRIDES[name])
    effective_reduction = changed_cfg.get("loss_reduction", contract["loss_reduction"])
    ensure_finite_checkpoint_state(model, optimizer)
    del checkpoint
    gc.collect()
    memory.guard()
    options = dict(batch_size=1, shuffle=False, num_workers=cfg["num_workers"],
                   generator=torch.Generator().manual_seed(cfg["seed"]))
    if cfg["num_workers"] > 0:
        options["prefetch_factor"] = cfg.get("fisher_prefetch_factor", 1)
    loader = DataLoader(PlannedVOC(base_dataset, plan), **options)
    iterator = iter(loader)
    diagnostics = TrainingDiagnostics(directory / f"{name}.jsonl", model)
    diagnostics.write(event="start", diagnostic_run=True, source=str(source), source_step=source_step,
                      overrides=ARM_OVERRIDES[name], source_contract=contract,
                      effective_loss_reduction=effective_reduction,
                      effective_lrs=effective_lrs,
                      note="local intervention; unchanged source GMM; NOT full geometry retraining")
    model.train()
    records, losses, common_losses, fingerprints = [], [], [], []
    status, error, completed = "completed", None, 0
    start = time.perf_counter()
    warning_saved = False
    current_batches = []
    rng_before_update = None

    def save_evidence(reason, step):
        nonlocal warning_saved
        if warning_saved:
            return
        evidence = directory / f"{name}_first_warning.pt"
        # Snapshot the exact *pre-update* state and current transformed tensors.
        # No claim of exact replay of the original 9274-step failure.
        ensure_finite_checkpoint_state(model, optimizer)
        temporary = evidence.with_suffix(".pt.tmp")
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), source=str(source), source_step=source_step,
                        probe_step=step, config=changed_cfg, diagnostic_run=True, reason=reason,
                        batches=current_batches, rng_before_update=rng_before_update), temporary)
        temporary.replace(evidence)
        warning_saved = True
        diagnostics.write(event="first_warning_saved", step=step, reason=reason, path=str(evidence))

    try:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            rng_before_update = dict(python=random.getstate(), numpy=np.random.get_state(),
                                     torch=torch.get_rng_state(),
                                     cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
            loss_values, common_values, records, current_batches = [], [], [], []
            for micro in range(cfg.get("fisher_gradient_accumulation_steps", 2)):
                memory.guard()
                images, targets, valid, ids, sizes, ordinal = next(iterator)
                fingerprint = batch_fingerprint(images, targets, valid, sizes)
                fingerprints.append(fingerprint)
                current_batches.append(dict(images=images, targets=targets, valid=valid,
                                            original_sizes=sizes, image_ids=list(ids), ordinal=int(ordinal[0])))
                images_gpu, targets_gpu, valid_gpu = images.to(device), targets.to(device), valid.to(device)
                logits = model(images_gpu, sizes.to(device) if changed_cfg.get("patch_coordinate_mode") == "original" else None)
                loss = masked_sigmoid_loss(
                    logits, targets_gpu, valid_gpu, effective_reduction)
                common_value = float(masked_sigmoid_loss(
                    logits.detach(), targets_gpu, valid_gpu, "batch_sum"))
                record = dict(image_ids=list(ids), ordinal=int(ordinal[0]), fingerprint=fingerprint,
                              input_shape=list(images.shape), loss=float(loss.detach()),
                              common_class_sum_loss=common_value,
                              activations=dict(diagnostics.activations))
                records.append(record)
                diagnostics.write(event="forward", step=step, source_step=source_step,
                                  micro_batch=micro + 1, **record)
                value = checked_loss_value(loss)
                if value > args.loss_limit:
                    save_evidence(f"loss exceeded diagnostic stop limit {args.loss_limit}", step)
                    raise FloatingPointError("diagnostic loss limit exceeded; no update applied")
                loss_values.append(value)
                common_values.append(common_value)
                (loss / cfg.get("fisher_gradient_accumulation_steps", 2)).backward()
                # Keep only CPU evidence for this batch; no lingering graph or
                # last GPU input reference while allocating the next microbatch.
                del images_gpu, targets_gpu, valid_gpu, logits, loss
                memory.guard()
            grad_max = float(ensure_finite_gradients(model))
            ratio_limit = getattr(args, "warning_update_ratio", None)
            if ratio_limit is not None:
                predictions = predicted_fisher_relative_update(model, optimizer)
                if any(value is not None and value > ratio_limit for value in predictions.values()):
                    diagnostics.write(event="large_update_prediction", step=step, predictions=predictions)
                    save_evidence(f"predicted Fisher relative update exceeded {ratio_limit}: {predictions}", step)
            if step == getattr(args, "evidence_step", None):
                save_evidence("explicit pre-update capture for same-batch replay", step)
            if max(loss_values) > args.warning_loss or grad_max > args.warning_grad:
                save_evidence(f"loss={max(loss_values)}; gradient_absmax={grad_max}", step)
            if warning_saved and getattr(args, "capture_only", False):
                status = "captured_before_update"
                diagnostics.write(event="capture_only_stop", step=step, no_update=True)
                break
            if step == 1 or step % args.group_every == 0 or warning_saved:
                diagnostics.write(event="before_update", step=step, **diagnostics.snapshot(model, optimizer))
            diagnostics.capture_before_update(model)
            optimizer.step()
            updates = diagnostics.update_snapshot(model)
            diagnostics.write(event="after_update", step=step, gradient_absmax=grad_max, updates=updates,
                              memory=memory.sample() if memory.enabled else None,
                              cuda_allocated_bytes=torch.cuda.memory_allocated(device) if memory.enabled else None,
                              cuda_reserved_bytes=torch.cuda.memory_reserved(device) if memory.enabled else None)
            scheduler.step()
            ensure_finite_checkpoint_state(model, optimizer) if warning_saved else None
            completed = step
            losses.append(sum(loss_values) / len(loss_values))
            common_losses.append(sum(common_values) / len(common_values))
            print(f"arm={name} update={step}/{args.steps} source_step={source_step} "
                  f"loss={losses[-1]:.6g} gradient_absmax={grad_max:.6g}", flush=True)
    except (FloatingPointError, RuntimeError) as exception:
        # OOM aborts the entire run. Do not auto-restart another memory-heavy arm.
        if "out of memory" in str(exception).lower():
            raise
        if isinstance(exception, RuntimeError) and "non-finite" not in str(exception):
            raise
        status, error = "failed_or_stopped", str(exception)
        diagnostics.write(event="failure", step=completed + 1, error=error,
                          traceback=traceback.format_exc(), batches=records)
        if not warning_saved:
            save_evidence(error, completed + 1)
        print(f"arm={name} stopped: {error}", flush=True)
    finally:
        diagnostics.close()
        memory_statistics = memory.finish()
        exception = sys.exc_info()[1]
        if exception is not None and memory.enabled:
            # Preserve measured resource evidence even if an unexpected error
            # prevents the normal arm summary. Never write a usable model here.
            with (directory / f"{name}_partial_memory.json").open("x", encoding="utf-8") as stream:
                json.dump(dict(status="aborted", error=str(exception), updates_completed=completed,
                               memory_statistics=memory_statistics,
                               cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                               cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                               diagnostic_run=True), stream, indent=2)
    ensure_finite_checkpoint_state(model, optimizer)
    elapsed = time.perf_counter() - start
    result = dict(status=status, error=error, updates_completed=completed, source_step=source_step,
                  overrides=ARM_OVERRIDES[name], elapsed_seconds=elapsed,
                  seconds_per_completed_update=elapsed / max(completed, 1),
                  first_20_mean=float(np.mean(losses[:20])) if losses else None,
                  last_20_mean=float(np.mean(losses[-20:])) if losses else None,
                  max_loss=max(losses, default=None), batch_fingerprints=fingerprints,
                  effective_lrs=effective_lrs,
                  common_class_sum_first_20_mean=float(np.mean(common_losses[:20])) if common_losses else None,
                  common_class_sum_last_20_mean=float(np.mean(common_losses[-20:])) if common_losses else None,
                  common_class_sum_max=max(common_losses, default=None),
                  evidence_saved=warning_saved, diagnostic_run=True,
                  memory_statistics=memory_statistics,
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if memory.enabled else None,
                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if memory.enabled else None,
                  memory_efficient=efficient)
    # Normal completion writes metrics only, not a formal/usable checkpoint.
    del model, optimizer, scheduler, iterator, loader, diagnostics, current_batches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--allow-preparation-source", action="store_true",
                        help="explicit diagnostic-only contrast from a finite preparation checkpoint; never a formal resume")
    parser.add_argument("--capture-only", action="store_true",
                        help="single-arm bounded capture; stop before applying the first warning update")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--memory-efficient", action="store_true",
                        help="mmap source + two worst-plan calibration updates with measured memory floors before the matched arms; no algorithm/worker changes")
    parser.add_argument("--arms", nargs="+", choices=ARM_OVERRIDES, default=["control", "boundary"])
    parser.add_argument("--plan-seed", type=int, default=20261003)
    parser.add_argument("--group-every", type=int, default=10)
    parser.add_argument("--evidence-step", type=int, default=None,
                        help="capture exact transformed batches and pre-update state at a selected probe")
    parser.add_argument("--warning-loss", type=float, default=100.)
    parser.add_argument("--warning-grad", type=float, default=10000.)
    parser.add_argument("--warning-update-ratio", type=float, default=None,
                        help="capture first predicted large Fisher SGD displacement before it happens; diagnostic only")
    parser.add_argument("--loss-limit", type=float, default=10000.)
    args = parser.parse_args()
    if args.steps <= 0 or args.group_every <= 0 or len(set(args.arms)) != len(args.arms):
        raise ValueError("positive steps/cadence and unique arms required")
    if len(args.arms) < 2 and not args.capture_only:
        raise ValueError("a matched contrast requires at least two arms")
    if args.evidence_step is not None and not 1 <= args.evidence_step <= args.steps:
        raise ValueError("evidence-step must be within the planned updates")
    if args.warning_update_ratio is not None and (not math.isfinite(args.warning_update_ratio) or args.warning_update_ratio <= 0):
        raise ValueError("warning-update-ratio must be finite and positive")
    if any(not math.isfinite(v) or v <= 0 for v in (args.warning_loss, args.warning_grad, args.loss_limit)):
        raise ValueError("diagnostic limits must be finite and positive")
    root = Path(__file__).resolve().parent
    cfg = read_config(args.config, root)
    if cfg.get("model_variant") != "spp" or cfg.get("fisher_micro_batch_size", 1) != 1:
        raise ValueError("diagnostic runner currently supports SPP microbatch=1 only")
    source = (root / args.checkpoint).resolve()
    directory = (root / args.output_dir).resolve()
    device = torch.device(cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    # A temporary resource wait must not create an empty "failed experiment"
    # directory which then prevents the authorised first launch.
    if device.type == "cuda":
        admission = resource_check(memory_calibration=True) if args.memory_efficient else resource_check()
        print(f"initial_resource_check={admission}", flush=True)
    directory.mkdir(parents=True, exist_ok=False)
    base = VOCMultiLabel((root / cfg["dataset_root"]).resolve(), return_valid_mask=True,
                         split=training_split(cfg),
                         return_original_size=True, transform=ResizeLongestSide(cfg["image_scales"][0]))
    plan = make_batch_plan([item[0] for item in base.items], args.plan_seed, args.steps,
                           cfg.get("fisher_gradient_accumulation_steps", 2), cfg["image_scales"])
    with (directory / "batch_plan.json").open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2)
    summary = dict(diagnostic_run=True, source=str(source), source_sha256=file_sha256(source),
                   config=cfg, planned_updates=args.steps, plan_sha256=file_sha256(directory / "batch_plan.json"),
                   pytorch_version=torch.__version__, seed=args.plan_seed, arms={},
                   interpretation="short local numerical intervention; no mAP or paper equivalence conclusion")
    if args.memory_efficient:
        sizes = {}
        for record in plan:
            if record["image_id"] not in sizes:
                with Image.open(base.base / "JPEGImages" / f"{record['image_id']}.jpg") as image:
                    sizes[record["image_id"]] = image.size
        calibration_plan = memory_calibration_plan(plan, sizes, cfg)
        calibration_directory = directory / "memory_calibration"
        calibration_directory.mkdir()
        with (calibration_directory / "batch_plan.json").open("x", encoding="utf-8") as stream:
            json.dump(calibration_plan, stream, indent=2)
        calibration_args = copy.copy(args)
        calibration_args.steps = 2
        calibration_args.evidence_step = None
        calibration_args.capture_only = False
        print("memory_calibration_start: two worst-plan updates; weights will be discarded", flush=True)
        calibration = run_arm("control", cfg, source, calibration_directory, base,
                              calibration_plan, calibration_args, device)
        with (calibration_directory / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(calibration, stream, indent=2)
        minimum = calibration["memory_statistics"]["minimum"]
        if (calibration["updates_completed"] != 2 or calibration["status"] != "completed"
                or (minimum and (minimum["commit_free_bytes"] < 2 * 1024 ** 3
                                 or minimum["ram_free_bytes"] < 256 * 1024 ** 2))):
            raise RuntimeError("bounded memory calibration did not pass; no matched arm launched")
        summary["memory_calibration"] = calibration
        print(f"memory_calibration_passed={calibration['memory_statistics']}; peak_cuda_reserved={calibration['cuda_peak_reserved_bytes']}", flush=True)
    for arm in args.arms:
        if device.type == "cuda":
            if args.memory_efficient:
                # Calibration and arms share this live CUDA context. Requiring
                # the original 6144 MiB free again counts that context twice.
                required = gpu_reservation_after_calibration(summary["memory_calibration"])
                admission = resource_check(reserve_new_process=False, gpu_required_mib=required)
            else:
                admission = resource_check(reserve_new_process=False)
            print(f"resource_check={admission}", flush=True)
        summary["arms"][arm] = run_arm(arm, cfg, source, directory, base, plan, args, device)
        with (directory / f"{arm}_summary.json").open("x", encoding="utf-8") as stream:
            json.dump(summary["arms"][arm], stream, indent=2)
    matched, count = paired_inputs_equal(summary["arms"])
    summary.update(matched_inputs=matched, compared_micro_batches=count)
    summary["source_sha256_after"] = file_sha256(source)
    if summary["source_sha256_after"] != summary["source_sha256"]:
        raise RuntimeError("source checkpoint changed during diagnostic")
    if not matched and not (args.capture_only and len(args.arms) == 1):
        raise RuntimeError("input fingerprints disagree: contrast is invalid")
    with (directory / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"saved={directory / 'summary.json'}; matched_inputs={matched}; micro_batches={count}", flush=True)


if __name__ == "__main__":
    main()
