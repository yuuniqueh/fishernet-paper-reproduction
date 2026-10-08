"""Opt-in memory protection for the fixed-shape train-only whole stage.

Reuse the pressure-reclamation implementation tested by short_contrast, not
its two-worst-patch calibration or its diagnostic model/optimizer semantics.
The first two real updates are monitored warmup, not discarded training.
"""

import json
from pathlib import Path

import torch

from short_contrast import MemoryMonitor, reclaim_unused_cuda_cache, resource_check
from src.training import protect_output, training_split


class WholeImageMemory:
    def __init__(self, enabled, cfg, summary_path, target_steps):
        self.enabled = enabled
        self.cfg = cfg
        self.path = Path(summary_path) if summary_path else None
        self.target_steps = target_steps
        self.completed = 0
        self.start_step = 0
        self.updates_this_run = 0
        self.monitor = None
        self.admission = None

    def __enter__(self):
        if not self.enabled:
            return self
        if training_split(self.cfg) != "train":
            raise ValueError("memory-efficient whole stage is currently qualified only for train-only validation")
        if self.path is None or self.target_steps < 2:
            raise ValueError("memory-efficient whole stage needs a new memory-summary and at least two updates")
        if self.cfg.get("device") != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("memory-efficient whole stage requires verified CUDA, not a CPU fallback")
        # Explicit new fixed-224 validation path: retain 6 GiB GPU/2 GiB RAM,
        # provisionally reserve 8 GiB commit, then observe every real update.
        # This does NOT change admission for the existing Fisher entrypoint.
        self.admission = resource_check(memory_calibration=True, gpu_required_mib=6144)
        protect_output(self.path)
        self.monitor = MemoryMonitor(True, reclaim_unused_cuda_cache,
                                     ram_reclaim_below_bytes=512 * 1024 ** 2)
        self.monitor.start()
        torch.cuda.reset_peak_memory_stats()
        print(f"whole_memory_admission={self.admission}; fixed_input=224; "
              "first_two_real_updates_are_monitored_warmup", flush=True)
        return self

    def guard(self):
        if self.monitor is not None:
            return self.monitor.guard()

    def begin_training(self, step):
        """Absolute checkpoint step and this process's warmup are distinct."""
        if step < 0 or step >= self.target_steps:
            raise ValueError("whole training start step must precede its target")
        if self.enabled and self.target_steps - step < 2:
            raise ValueError("guarded whole continuation requires at least two monitored real updates")
        self.start_step = self.completed = step
        self.updates_this_run = 0

    def after_update(self, step):
        if step != self.completed + 1:
            raise ValueError("whole memory updates must advance exactly one step")
        self.completed = step
        self.updates_this_run += 1
        self.guard()
        if self.monitor is not None and self.updates_this_run == 2:
            minimum = self.monitor.minimum
            if (minimum["commit_free_bytes"] < 2 * 1024 ** 3
                    or minimum["ram_free_bytes"] < 256 * 1024 ** 2):
                raise RuntimeError("whole-stage monitored warmup exceeded memory floor; do not automatically restart")
            print(f"whole_memory_warmup_passed={minimum}; start_step={self.start_step}; "
                  f"updates_this_run={self.updates_this_run}; "
                  f"cuda_peak_reserved={torch.cuda.max_memory_reserved()}; "
                  "continued_guarding_required", flush=True)

    def __exit__(self, error_type, error, tb):
        if self.monitor is None:
            return False
        report = dict(stage="whole_image", train_split=training_split(self.cfg),
                      memory_efficient=True, fixed_crop_size=224,
                      target_steps=self.target_steps, updates_completed=self.completed,
                      start_step=self.start_step, updates_this_run=self.updates_this_run,
                      status="completed" if error is None and self.completed == self.target_steps else "stopped",
                      error=str(error) if error is not None else None,
                      admission=self.admission, memory_statistics=self.monitor.finish(),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      note="monitored actual training; no guarantee of future memory safety or bitwise CUDA repeatability")
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(self.path)
        print(f"whole_memory_summary={self.path}; status={report['status']}", flush=True)
        return False
