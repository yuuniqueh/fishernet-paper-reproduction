"""Stage 1: whole-image fine-tuning before FisherNet/GMM initialization."""

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.training import (read_config, masked_sigmoid_loss, build_optimizer,
                          assert_resume_compatible, protect_output, checked_loss_value, ensure_finite_gradients,
                          ensure_finite_checkpoint_state, training_split)

from models.whole_image import WholeImageAlexNet
from src.voc_dataset import VOCMultiLabel, whole_image_finetune_transform
from src.guarded_memory import WholeImageMemory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", default="checkpoints/whole_image_last.pt")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--memory-efficient", action="store_true", help="opt-in guarded fixed-224 train-only validation; unchanged batch/loss/LR")
    parser.add_argument("--memory-summary", default=None, help="new resource evidence file required with --memory-efficient")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    cfg = read_config(args.config, project_root)
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    if args.save_every is not None and args.save_every <= 0:
        raise ValueError("save-every must be positive")
    if args.memory_summary and not args.memory_efficient:
        raise ValueError("memory-summary requires --memory-efficient")
    summary = (project_root / args.memory_summary).resolve() if args.memory_summary else None
    with WholeImageMemory(args.memory_efficient, cfg, summary,
                          args.max_steps or cfg["whole_image_steps"]) as memory:
        run_training(args, cfg, project_root, memory)


def run_training(args, cfg, project_root, memory):
    output = (project_root / args.output).resolve()
    protect_output(output, project_root / args.resume if args.resume else None)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    dataset_root = (project_root / cfg["dataset_root"]).resolve()
    dataset = VOCMultiLabel(
        dataset_root,
        year="2007",
        split=training_split(cfg),
        transform=whole_image_finetune_transform(),
        return_valid_mask=True,
    )
    if args.debug:
        dataset.items = dataset.items[: min(32, len(dataset))]
    loader = DataLoader(
        dataset,
        batch_size=cfg["whole_image_batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )
    model = WholeImageAlexNet(
        num_classes=cfg["num_classes"],
        descriptor_dim=cfg["descriptor_dim"],
        pretrained=cfg["pretrained_backbone"] and not args.resume,
        fc_dropout=cfg.get("fc_dropout", False),
        new_layer_init_std=cfg.get("new_layer_init_std"),
    ).to(device)
    optimizer, scheduler, contract = build_optimizer(model, cfg, "whole_image")
    print(f"loss_reduction={contract['loss_reduction']}; lr_groups="
          f"{[(group['name'], group['lr']) for group in optimizer.param_groups]}", flush=True)
    step = 0
    resume_provenance = None
    if args.resume:
        resume_path = project_root / args.resume
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False,
                                mmap=args.memory_efficient)
        if bool(checkpoint.get("debug_run", False)) != bool(args.debug):
            raise ValueError("cannot mix debug and full-data runs via --resume")
        assert_resume_compatible(checkpoint, cfg, "whole_image")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = checkpoint["step"]
        resume_provenance = dict(checkpoint=str(resume_path.resolve()), step=step,
                                 data_rng_restored=False,
                                 data_resume_mode="legacy_reseeded_epoch")
        del checkpoint
        print(f"resumed={resume_path}; step={step}")

    memory.begin_training(step)

    def save_checkpoint(path: Path):
        ensure_finite_checkpoint_state(model, optimizer)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "config": cfg,
                "training_contract": contract,
                "debug_run": bool(args.debug),
                "memory_efficient": args.memory_efficient,
                "resume_provenance": resume_provenance,
            },
            temporary,
        )
        temporary.replace(path)

    model.train()
    target_steps = args.max_steps or cfg["whole_image_steps"]
    save_every = args.save_every or cfg["whole_image_save_every"]
    while step < target_steps:
        for images, targets, valid, _ in loader:
            memory.guard()
            step += 1
            logits = model(images.to(device))
            targets, valid = targets.to(device), valid.to(device)
            loss = masked_sigmoid_loss(logits, targets, valid, contract["loss_reduction"])
            loss_value = checked_loss_value(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            ensure_finite_gradients(model)
            memory.guard()
            optimizer.step()
            scheduler.step()
            print(f"step={step}/{target_steps} loss={loss_value:.4f} logits={tuple(logits.shape)}", flush=True)
            if args.memory_efficient:
                del images, targets, valid, logits, loss
            memory.after_update(step)
            if step % save_every == 0:
                memory.guard()
                save_checkpoint(output)
                print(f"saved={output}; resume_step={step}")
            if step >= target_steps:
                break
    memory.guard()
    save_checkpoint(output)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
