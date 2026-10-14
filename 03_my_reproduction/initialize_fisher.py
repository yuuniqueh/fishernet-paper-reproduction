"""Stage 2: fit a GMM to fine-tuned patch descriptors and initialize FisherNet."""

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from src.training import (read_config, assert_model_compatible, protect_output,
                          assert_whole_source_loss_compatible, training_split,
                          assert_initialization_split_compatible)

from models.factory import build_model
from src.gmm_init import initialize_fisher_from_gmm
from src.gmm_sampling import (descriptor_metadata, sample_patch_indices,
                              save_descriptor_cache, load_descriptor_cache,
                              gmm_diagnostics, save_diagnostics)
from src.voc_dataset import RandomFisherNetTransform, VOCMultiLabel
from src.guarded_gmm import GMMMemory, GuardedGaussianMixture, DescriptorBuffer, run_gmm_memory_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--whole-image-checkpoint", default="checkpoints/whole_image_last.pt")
    parser.add_argument("--output", default="checkpoints/fishernet_gmm_init.pt")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-descriptors", type=int, default=None)
    parser.add_argument("--gmm-max-iter", type=int, default=None, help="retry EM on a matching descriptor cache")
    parser.add_argument("--allow-debug-source", action="store_true", help="only for explicitly marked smoke experiments")
    parser.add_argument("--memory-efficient", action="store_true")
    parser.add_argument("--memory-probe", action="store_true", help="independent memory qualification only; no GMM/cache checkpoint")
    parser.add_argument("--memory-summary", default=None)
    parser.add_argument("--memory-qualification", default=None)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    cfg = read_config(args.config, project_root)
    if args.gmm_max_iter is not None:
        if args.gmm_max_iter < 1:
            raise ValueError("gmm-max-iter must be positive")
        cfg["gmm_max_iter"] = args.gmm_max_iter
    if args.memory_probe and not args.memory_efficient:
        raise ValueError("memory-probe requires memory-efficient")
    if args.memory_efficient and (args.max_images or args.max_descriptors or args.allow_debug_source):
        raise ValueError("guarded GMM uses the unchanged configured descriptor budget/nondebug source")
    summary = project_root / args.memory_summary if args.memory_summary else None
    qualification = project_root / args.memory_qualification if args.memory_qualification else None
    with GMMMemory(args.memory_efficient, cfg, project_root / args.whole_image_checkpoint,
                   summary, args.memory_probe, qualification) as memory:
        run_initialization(args, project_root, cfg, memory)


def run_initialization(args, project_root, cfg, memory):
    output = (project_root / args.output).resolve()
    if not args.memory_probe:
        protect_output(output)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    checkpoint_path = project_root / args.whole_image_checkpoint
    memory.guard("load_source")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=args.memory_efficient)
    assert_model_compatible(checkpoint, cfg, whole_image=True)
    assert_initialization_split_compatible(checkpoint, cfg)
    debug_source = bool(checkpoint.get("debug_run", False))
    if debug_source and not args.allow_debug_source:
        raise ValueError("debug-only whole-image checkpoint cannot initialize a full experiment")

    cfg["backbone_config"] = checkpoint["config"].get("backbone_config", cfg.get("backbone_config"))
    whole_image = build_model(cfg, "whole_image", pretrained=False)
    whole_image.load_state_dict(checkpoint["model"])
    source_step = checkpoint.get("step")
    source_config = checkpoint["config"]
    if source_step is None or source_step < cfg["whole_image_steps"]:
        raise ValueError("whole-image fine-tuning is incomplete; do not initialize GMM yet")
    source_loss = assert_whole_source_loss_compatible(checkpoint, cfg)
    del checkpoint
    model = build_model(cfg, pretrained=False).to(device)
    model.backbone.load_state_dict(whole_image.backbone.state_dict(), strict=True)
    if hasattr(model, "fc6"):
        model.fc6.load_state_dict(whole_image.fc6.state_dict())
        model.fc7.load_state_dict(whole_image.fc7.state_dict())
    model.descriptor_projection.load_state_dict(whole_image.descriptor_projection.state_dict())
    del whole_image
    model.eval()

    dataset_root = (project_root / cfg["dataset_root"]).resolve()
    dataset = VOCMultiLabel(
        dataset_root,
        year="2007",
        split=training_split(cfg),
        transform=RandomFisherNetTransform(cfg["image_scales"], horizontal_flip=False),
        return_original_size=cfg.get("patch_coordinate_mode", "resized") == "original",
    )
    if args.memory_probe:
        run_gmm_memory_probe(model, dataset, cfg, memory, device)
        return
    max_images = args.max_images or cfg["gmm_max_images"]
    max_descriptors = args.max_descriptors or cfg["gmm_max_descriptors"]
    if max_images < 1 or max_descriptors < cfg["num_components"]:
        raise ValueError("GMM requires positive max_images and at least K descriptors")
    sampling_mode = cfg.get("gmm_sampling_mode", "legacy_prefix")
    if sampling_mode not in {"legacy_prefix", "uniform_per_image"}:
        raise ValueError("gmm_sampling_mode must be legacy_prefix or uniform_per_image")
    target_images = min(max_images, len(dataset))
    metadata = descriptor_metadata(checkpoint_path, cfg, target_images, max_descriptors)
    cache_path = cfg.get("gmm_descriptor_cache")
    cache_path = (project_root / cache_path).resolve() if cache_path else None
    if cache_path is not None and cache_path.exists():
        samples, sampled_ids, counts = load_descriptor_cache(cache_path, metadata)
        print(f"reused_descriptor_cache={cache_path}", flush=True)
    else:
        loader_options = dict(batch_size=1, shuffle=True, num_workers=cfg["num_workers"])
        if cfg["num_workers"] > 0:
            loader_options["prefetch_factor"] = cfg.get("fisher_prefetch_factor", 2)
        loader = DataLoader(dataset, **loader_options)
        collected, sampled_ids, counts = [], [], []
        buffer = DescriptorBuffer(max_descriptors, cfg["descriptor_dim"]) if args.memory_efficient else None
        total = 0
        rng = np.random.default_rng(cfg["seed"])
        with torch.inference_mode():
            for image_index, batch in enumerate(loader):
                memory.guard("sampling")
                if cfg.get("patch_coordinate_mode", "resized") == "original":
                    images, _, ids, original_sizes = batch
                else:
                    images, _, ids = batch
                    original_sizes = None
                patch_indices = None
                if sampling_mode == "uniform_per_image":
                    height, width = images.shape[-2:]
                    patch_count = (model.descriptor_count(height, width) if hasattr(model, "descriptor_count")
                                   else len(model._image_boxes(height, width, original_sizes, images.device)))
                    selected = sample_patch_indices(patch_count, image_index, target_images, max_descriptors, rng)
                    if len(selected) == 0:
                        sampled_ids.extend(ids)
                        counts.append(0)
                        if image_index + 1 >= target_images:
                            break
                        continue
                    patch_indices = torch.from_numpy(selected).to(device)
                if original_sizes is not None:
                    original_sizes = original_sizes.to(device)
                descriptors = model.extract_descriptors(
                    images.to(device), original_sizes, patch_indices=patch_indices
                ).squeeze(0).cpu().numpy()
                chosen = descriptors[:max_descriptors - total]
                if buffer is not None:
                    buffer.append(chosen)
                else:
                    collected.append(chosen)
                sampled_ids.extend(ids)
                counts.append(len(chosen))
                total += len(chosen)
                if args.memory_efficient:
                    del chosen, descriptors, images, original_sizes, patch_indices, batch
                    memory.guard()
                if (image_index + 1) % 50 == 0:
                    print(f"sampling_images={image_index + 1}/{target_images}; descriptors={total}", flush=True)
                if image_index + 1 >= target_images or total >= max_descriptors:
                    break
        if not collected and not (buffer is not None and buffer.size):
            raise ValueError("no descriptors collected from configured patches")
        samples = buffer.array() if buffer is not None else np.concatenate(collected, axis=0)
        if args.memory_efficient:
            del loader
            memory.guard("cache_write")
        if cache_path is not None:
            save_descriptor_cache(cache_path, samples, metadata, sampled_ids, counts)
            print(f"saved_descriptor_cache={cache_path}", flush=True)
    if not np.isfinite(samples).all() or len(samples) < cfg["num_components"]:
        raise ValueError("GMM input must be finite and contain at least K descriptors")
    print(f"GMM samples={samples.shape[0]} descriptor_dim={samples.shape[1]}; images={len(sampled_ids)}; mode={sampling_mode}", flush=True)
    diagnostics_path = output.with_suffix(".gmm_diagnostics.json")
    if diagnostics_path.exists():
        suffix = 2
        while output.with_suffix(f".gmm_diagnostics_{suffix}.json").exists():
            suffix += 1
        diagnostics_path = output.with_suffix(f".gmm_diagnostics_{suffix}.json")
    gmm_type = GuardedGaussianMixture if args.memory_efficient else GaussianMixture
    gmm = gmm_type(
        n_components=cfg["num_components"],
        covariance_type="diag",
        max_iter=cfg["gmm_max_iter"],
        reg_covar=1e-6,
        random_state=cfg["seed"],
        verbose=cfg.get("gmm_verbose", 0),
        verbose_interval=10,
    )
    if args.memory_efficient:
        gmm.memory_guard = memory.guard
    try:
        gmm.fit(samples)
        diagnostics = gmm_diagnostics(gmm, samples, metadata, sampled_ids, counts)
    except Exception as error:
        save_diagnostics(diagnostics_path, dict(metadata=metadata, converged=False, error=str(error),
                                               sample_count=len(samples), image_count=len(sampled_ids)))
        raise
    save_diagnostics(diagnostics_path, diagnostics)
    print(f"gmm_diagnostics={diagnostics_path}; converged={gmm.converged_}; iterations={gmm.n_iter_}", flush=True)
    if not gmm.converged_:
        raise RuntimeError(f"GMM did not converge in {gmm.n_iter_} iterations (lower_bound={gmm.lower_bound_}); "
                           "no usable initialization checkpoint saved; do not start training")
    initialize_fisher_from_gmm(model.fisher, gmm.means_, gmm.covariances_)
    memory.guard("save_checkpoint")
    if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise FloatingPointError("nonfinite initialized model; no usable GMM checkpoint saved")
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "gmm_means": gmm.means_,
            "gmm_variances": gmm.covariances_,
            "gmm_converged": gmm.converged_,
            "gmm_iterations": gmm.n_iter_,
            "gmm_lower_bound": gmm.lower_bound_,
            "source_whole_image_checkpoint": str(checkpoint_path),
            "source_whole_image_step": source_step,
            "source_whole_image_config": source_config,
            "source_whole_image_loss_reduction": source_loss,
            "gmm_sample_count": len(samples),
            "gmm_image_count": len(sampled_ids),
            "gmm_sampling_mode": sampling_mode,
            "gmm_diagnostics": str(diagnostics_path),
            "descriptor_cache": str(cache_path) if cache_path else None,
            "debug_run": debug_source,
            "config": cfg,
        },
        temporary,
    )
    temporary.replace(output)
    memory.completed = True
    print(f"saved={output}")


if __name__ == "__main__":
    main()
