"""Training entry point for the first runnable FisherNet model."""

import argparse
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.training import (read_config, masked_sigmoid_loss, build_optimizer,
                          assert_resume_compatible, assert_model_compatible,
                          protect_output, checked_loss_value, ensure_finite_gradients,
                          ensure_finite_checkpoint_state, reject_diagnostic_checkpoint,
                          assert_whole_source_loss_compatible, training_split,
                          assert_initialization_split_compatible)
from src.voc_dataset import RandomFisherNetTransform, VOCMultiLabel
from models.fishernet import FisherNetAlexNet
from models.fishernet_spp import FisherNetSPPAlexNet
from src.training_diagnostics import TrainingDiagnostics
from src.guarded_fisher import FisherMemory, maximum_geometry_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="train on random images without VOC")
    parser.add_argument("--max-patches", type=int, default=None, help="limit patches per image for a fast smoke test")
    parser.add_argument("--init-checkpoint", default=None, help="GMM-initialized FisherNet checkpoint")
    parser.add_argument("--resume", default=None, help="training checkpoint to resume")
    parser.add_argument("--output", default="checkpoints/fishernet_last.pt")
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--diagnostics-jsonl", default=None, help="new opt-in numerical diagnostic file")
    parser.add_argument("--diagnostics-every", type=int, default=1, help="sample observer/update diagnostics every N steps; first step always observed")
    parser.add_argument("--preparation", action="store_true", help="mark this bounded qualification run diagnostic-only; never resume it as formal training")
    parser.add_argument("--stop-loss-above", type=float, default=None, help="optional finite-loss safety screen, only for --preparation")
    parser.add_argument("--warning-evidence", default=None, help="new diagnostic-only pre-update snapshot of the first numerical warning")
    parser.add_argument("--warning-update-ratio", type=float, default=0.1)
    parser.add_argument("--replay-data-to-step", action="store_true", help="reconstruct seeded loader prefix for a diagnostic resume; not a guarantee of bitwise replay")
    parser.add_argument("--memory-efficient", action="store_true")
    parser.add_argument("--memory-probe", action="store_true", help="two discarded maximum-geometry updates; no training checkpoint saved")
    parser.add_argument("--memory-summary", default=None)
    parser.add_argument("--memory-qualification", default=None)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    cfg = read_config(args.config, project_root)
    if args.memory_probe:
        if (not args.memory_efficient or not args.init_checkpoint or args.resume or args.debug or args.smoke_test
                or args.max_patches is not None or args.max_steps not in (None, 2)
                or args.warning_evidence):
            raise ValueError("memory-probe requires a fresh full GMM, two discarded updates and no diagnostic weights")
        args.max_steps = 2
        args.preparation = True
        # Qualify the observer overhead used by preflight and periodic formal
        # diagnostics too; JSON-only evidence, never probe model weights.
        if not args.diagnostics_jsonl and args.memory_summary:
            args.diagnostics_jsonl = str(Path(args.memory_summary).with_suffix('.jsonl'))
        args.diagnostics_every = 1
    if args.memory_efficient and (args.resume or args.debug or args.smoke_test or args.max_patches is not None):
        raise ValueError("guarded Fisher currently qualifies fresh full-data/all-patch training only")
    summary = project_root / args.memory_summary if args.memory_summary else None
    qualification = project_root / args.memory_qualification if args.memory_qualification else None
    source = project_root / args.init_checkpoint if args.init_checkpoint else None
    with FisherMemory(args.memory_efficient, cfg, source, summary, args.max_steps or cfg['fisher_steps'],
                      args.memory_probe, qualification) as memory:
        run_training(args, project_root, cfg, memory)


def run_training(args, project_root, cfg, memory):
    if args.warning_evidence and not args.preparation:
        raise ValueError("warning-evidence requires --preparation")
    if not math.isfinite(args.warning_update_ratio) or args.warning_update_ratio <= 0:
        raise ValueError("warning-update-ratio must be positive and finite")
    evidence_path = (project_root / args.warning_evidence).resolve() if args.warning_evidence else None
    if evidence_path:
        protect_output(evidence_path)
    if args.init_checkpoint and args.resume:
        raise ValueError("use either --init-checkpoint or --resume, not both")
    if args.replay_data_to_step and (not args.resume or args.smoke_test):
        raise ValueError("data-prefix replay requires a real-data --resume checkpoint")
    if args.max_patches is not None:
        cfg["max_patches"] = args.max_patches
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    if args.diagnostics_every <= 0:
        raise ValueError("diagnostics-every must be positive")
    if args.stop_loss_above is not None:
        if not args.preparation or not math.isfinite(args.stop_loss_above) or args.stop_loss_above <= 0:
            raise ValueError("stop-loss-above requires --preparation and a positive finite limit")
    if args.save_every is not None and args.save_every <= 0:
        raise ValueError("save-every must be positive")
    output = (project_root / args.output).resolve()
    if not args.memory_probe:
        protect_output(output, project_root / args.resume if args.resume else None)
    device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    print(f"device={device}; config={args.config}")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    if cfg["backbone"] != "alexnet":
        raise NotImplementedError("The first runnable stage currently implements AlexNet only.")
    variant = cfg.get("model_variant", "dense")
    if variant == "dense":
        model = FisherNetAlexNet(
            num_classes=cfg["num_classes"],
            descriptor_dim=cfg["descriptor_dim"],
            num_components=cfg["num_components"],
            pretrained_backbone=cfg.get("pretrained_backbone", False) and not (args.init_checkpoint or args.resume),
        ).to(device)
    elif variant == "spp":
        max_patches = args.max_patches if args.max_patches is not None else cfg.get("max_patches")
        model = FisherNetSPPAlexNet(
            num_classes=cfg["num_classes"],
            descriptor_dim=cfg["descriptor_dim"],
            num_components=cfg["num_components"],
            patch_sizes=tuple(cfg["patch_scales"]),
            patch_step=cfg["patch_step"],
            max_patches=max_patches,
            patch_coordinate_mode=cfg.get("patch_coordinate_mode", "resized"),
            roi_coordinate_mapping=cfg.get("roi_coordinate_mapping", "legacy_ratio"),
            pretrained_backbone=cfg.get("pretrained_backbone", False) and not (args.init_checkpoint or args.resume),
            fc_dropout=cfg.get("fc_dropout", False),
            new_layer_init_std=cfg.get("new_layer_init_std"),
        ).to(device)
    else:
        raise ValueError(f"unknown model_variant: {variant}")
    print(f"model={variant}; max_patches={getattr(model, 'max_patches', 'n/a')}")
    opt, scheduler, contract = build_optimizer(model, cfg, "fisher")
    print(f"loss_reduction={contract['loss_reduction']}; lr_groups="
          f"{[(group['name'], group['lr']) for group in opt.param_groups]}; "
          f"lr_milestones={contract['milestones']}", flush=True)
    step = 0
    initialization_source = args.init_checkpoint
    if args.init_checkpoint:
        checkpoint = torch.load(project_root / args.init_checkpoint, map_location="cpu", weights_only=False, mmap=args.memory_efficient)
        reject_diagnostic_checkpoint(checkpoint)
        if not bool(checkpoint.get("gmm_converged", False)):
            raise ValueError("GMM checkpoint is unverified or NOT converged; training was not started")
        if checkpoint.get("debug_run", False) and not (args.debug or args.smoke_test):
            raise ValueError("debug-only GMM checkpoint cannot initialize a full experiment")
        assert_model_compatible(checkpoint, cfg)
        assert_initialization_split_compatible(checkpoint, cfg, gmm=True)
        assert_whole_source_loss_compatible(checkpoint, cfg, gmm=True)
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        print(f"initialized={project_root / args.init_checkpoint}")
    if args.resume:
        checkpoint = torch.load(project_root / args.resume, map_location="cpu", weights_only=False)
        reject_diagnostic_checkpoint(checkpoint, diagnostic=args.replay_data_to_step)
        if bool(checkpoint.get("debug_run", False)) != bool(args.debug or args.smoke_test):
            raise ValueError("cannot mix debug/smoke and full-data runs via --resume")
        assert_resume_compatible(checkpoint, cfg, "fisher")
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = checkpoint["step"]
        initialization_source = checkpoint.get("init_checkpoint")
        del checkpoint
        print(f"resumed={project_root / args.resume}; step={step}")

    def save_checkpoint():
        memory.guard()
        ensure_finite_checkpoint_state(model, opt)
        if args.memory_probe:
            return
        temporary = output.with_suffix(f"{output.suffix}.tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "config": cfg,
                "training_contract": contract,
                "init_checkpoint": initialization_source,
                "diagnostic_run": bool(args.replay_data_to_step or args.preparation),
                "preparation_run": bool(args.preparation),
                "debug_run": bool(args.debug or args.smoke_test),
            },
            temporary,
        )
        temporary.replace(output)

    if args.smoke_test:
        micro_batch_size = cfg.get("fisher_micro_batch_size", 1)
        accumulation_steps = cfg.get("fisher_gradient_accumulation_steps", 2)
        loader = None
    else:
        dataset_root = (project_root / cfg["dataset_root"]).resolve()
        ds = VOCMultiLabel(
            dataset_root,
            year="2007",
            split=training_split(cfg),
            transform=RandomFisherNetTransform(cfg["image_scales"], horizontal_flip=True),
            return_valid_mask=True,
            return_original_size=cfg.get("patch_coordinate_mode", "resized") == "original",
        )
        if args.debug:
            ds.items = ds.items[: min(8, len(ds))]
        if args.memory_probe:
            ds, memory.probe_image_ids = maximum_geometry_dataset(ds, cfg)
            print(f'fisher_probe_image_ids={memory.probe_image_ids}; maximum_scale={max(cfg["image_scales"])}; all_patches=true', flush=True)
        # Original images have different aspect ratios. Batch size 1 preserves each
        # image's geometry; gradients are accumulated to the paper's effective batch 2.
        micro_batch_size = cfg.get("fisher_micro_batch_size", 1)
        accumulation_steps = cfg.get("fisher_gradient_accumulation_steps", 2)
        if micro_batch_size != 1:
            raise ValueError("variable-resolution images require fisher_micro_batch_size=1")
        if accumulation_steps < 1:
            raise ValueError("gradient accumulation must be positive")
        loader_options = {
            "batch_size": micro_batch_size,
            "shuffle": True,
            "num_workers": cfg["num_workers"],
        }
        # On Windows, high-resolution multi-scale images can exhaust the
        # shared-memory mapping used by several prefetched worker batches.
        # Keep background loading enabled, but allow the experiment config to
        # bound the number of images resident in that queue.
        if cfg["num_workers"] > 0:
            loader_options["prefetch_factor"] = cfg.get("fisher_prefetch_factor", 2)
        loader = DataLoader(ds, **loader_options)
        data_iterator = iter(loader)

    def next_batch():
        nonlocal data_iterator
        if args.smoke_test:
            images = torch.randn(micro_batch_size, 3, 224, 224)
            targets = torch.randint(0, 2, (micro_batch_size, cfg["num_classes"]), dtype=torch.float32)
            valid = torch.ones_like(targets)
            original_sizes = torch.tensor([[224, 224]] * micro_batch_size) if cfg.get("patch_coordinate_mode") == "original" else None
            return images, targets, valid, original_sizes, ["synthetic"]
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            batch = next(data_iterator)
        if cfg.get("patch_coordinate_mode", "resized") == "original":
            images, targets, valid, image_ids, original_sizes = batch
        else:
            images, targets, valid, image_ids = batch
            original_sizes = None
        return images, targets, valid, original_sizes, list(image_ids)

    if args.replay_data_to_step:
        skipped_batches = step * accumulation_steps
        print(f"replaying_data_prefix={skipped_batches} micro-batches; seeded reconstruction, not verified exact replay", flush=True)
        for skipped in range(skipped_batches):
            next_batch()
            if (skipped + 1) % 2000 == 0:
                print(f"replayed_micro_batches={skipped + 1}/{skipped_batches}", flush=True)

    diagnostics = TrainingDiagnostics(project_root / args.diagnostics_jsonl, model) if args.diagnostics_jsonl else None
    if diagnostics:
        diagnostics.write(event="start", resume=args.resume, init=args.init_checkpoint,
                          start_step=step, contract=contract, replay_data_prefix=args.replay_data_to_step,
                          preparation=args.preparation, diagnostics_every=args.diagnostics_every)

    model.train()
    target_steps = args.max_steps or cfg["fisher_steps"]
    save_every = args.save_every or cfg["fisher_save_every"]
    print(f"micro_batch={micro_batch_size}; gradient_accumulation={accumulation_steps}")
    evidence_saved = False
    current_batches = []
    rng_before_update = None

    def save_warning(reason, probe_step):
        nonlocal evidence_saved
        if not evidence_path or evidence_saved:
            return
        # Evidence may contain nonfinite statistics: preserve the scene, not a usable model.
        temporary = evidence_path.with_suffix(".pt.tmp")
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                        scheduler=scheduler.state_dict(), config=cfg, step=step,
                        probe_step=probe_step, diagnostic_run=True, preparation_run=True,
                        debug_run=bool(args.debug or args.smoke_test), reason=reason,
                        batches=current_batches, rng_before_update=rng_before_update,
                        init_checkpoint=initialization_source), temporary)
        temporary.replace(evidence_path)
        evidence_saved = True
        if diagnostics:
            diagnostics.write(event="first_warning_saved", step=probe_step, reason=reason, path=str(evidence_path))
        print(f"warning_evidence={evidence_path}; probe_step={probe_step}; reason={reason}", flush=True)

    while step < target_steps:
        memory.guard()
        current_batches = []
        if evidence_path and not evidence_saved:
            rng_before_update = dict(python=random.getstate(), numpy=np.random.get_state(),
                                     torch=torch.get_rng_state(),
                                     cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
        if diagnostics:
            diagnostics.active = step == 0 or (step + 1) % args.diagnostics_every == 0
            diagnostics.activations.clear()
        opt.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        logits_shape = None
        batch_records = []
        for _ in range(accumulation_steps):
            memory.guard()
            images, targets, valid, original_sizes, image_ids = next_batch()
            if evidence_path and not evidence_saved:
                current_batches.append(dict(images=images.clone(), targets=targets.clone(), valid=valid.clone(),
                                            original_sizes=original_sizes.clone() if original_sizes is not None else None,
                                            image_ids=image_ids))
            images, targets, valid = images.to(device), targets.to(device), valid.to(device)
            if original_sizes is not None:
                original_sizes = original_sizes.to(device)
            logits = model(images, original_sizes) if variant == "spp" else model(images)
            loss = masked_sigmoid_loss(logits, targets, valid, contract["loss_reduction"])
            if diagnostics and diagnostics.active:
                batch_records.append(dict(image_ids=image_ids, input_shape=list(images.shape),
                                          loss=float(loss.detach()), activations=dict(diagnostics.activations)))
                diagnostics.write(event="forward", step=step + 1, micro_batch=len(batch_records), **batch_records[-1])
            try:
                loss_value = checked_loss_value(loss)
            except FloatingPointError:
                save_warning("nonfinite loss before optimizer update", step + 1)
                raise
            if loss_value > 100:
                save_warning(f"single-image loss={loss_value} exceeded warning 100", step + 1)
            if args.stop_loss_above is not None and loss_value > args.stop_loss_above:
                if diagnostics:
                    diagnostics.write(event="preparation_stop", step=step + 1,
                                      image_ids=image_ids, loss=loss_value,
                                      reason="finite loss exceeded preparation screen; no update")
                raise FloatingPointError("finite loss exceeded preparation screen; no update; do not automatically restart")
            (loss / accumulation_steps).backward()
            accumulated_loss += loss_value
            logits_shape = tuple(logits.shape)
            if args.memory_efficient:
                del loss, logits, images, targets, valid, original_sizes
                memory.guard()
        step += 1
        try:
            gradient_max = float(ensure_finite_gradients(model))
        except RuntimeError:
            save_warning("nonfinite gradient before optimizer update", step)
            raise
        if evidence_path and not evidence_saved:
            from short_contrast import predicted_fisher_relative_update
            predicted = predicted_fisher_relative_update(model, opt)
            if gradient_max > 10000 or any(value is not None and value > args.warning_update_ratio for value in predicted.values()):
                save_warning(f"gradient_absmax={gradient_max}; predicted_relative_update={predicted}", step)
        observe_update = diagnostics and (diagnostics.active or accumulated_loss / accumulation_steps > 100)
        if observe_update:
            diagnostics.write(event="before_update", step=step, batches=batch_records,
                              **diagnostics.snapshot(model, opt))
            diagnostics.capture_before_update(model)
        opt.step()
        if observe_update:
            diagnostics.write(event="after_update", step=step, updates=diagnostics.update_snapshot(model), parameters={name: {
                "finite": bool(torch.isfinite(param).all()), "absmax": float(param.detach().abs().max())}
                for name, param in model.named_parameters() if name.startswith("fisher.") or name == "classifier.weight"})
        scheduler.step()
        memory.after_update(step)
        print(f"step={step}/{target_steps} loss={accumulated_loss / accumulation_steps:.4f} logits={logits_shape}", flush=True)
        if step % save_every == 0:
            save_checkpoint()
            print(f"saved={output}; resume_step={step}")
    save_checkpoint()
    if not args.memory_probe:
        print(f"saved={output}")
    else:
        print("memory_probe_complete; two updates discarded; no training checkpoint saved", flush=True)
    if diagnostics:
        diagnostics.close()


if __name__ == "__main__":
    main()
