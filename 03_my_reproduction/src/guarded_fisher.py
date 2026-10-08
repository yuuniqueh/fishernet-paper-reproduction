"""Opt-in all-patch Fisher memory qualification and continuous pressure guard.

The probe performs two discarded updates at the largest actual train geometry.
It never qualifies a changed batch, patch, scale, loss or optimizer algorithm.
"""

import json
import math
from pathlib import Path
import subprocess

from PIL import Image
import torch
from torch.utils.data import Subset

from short_contrast import MemoryMonitor, reclaim_unused_cuda_cache, windows_memory_status
from src.gmm_sampling import file_sha256
from src.training import protect_output, training_contract, training_split
from src.voc_dataset import ResizeLongestSide

GIB = 1024 ** 3


def fisher_memory_identity(source, cfg):
    root = Path(__file__).resolve().parents[1]
    return dict(source=str(Path(source).resolve()), source_sha256=file_sha256(source),
                contract=training_contract(cfg, 'fisher'), num_workers=cfg['num_workers'],
                torch_version=torch.__version__,
                prefetch=cfg.get('fisher_prefetch_factor', 2),
                implementation={name: file_sha256(root / name) for name in
                    ('train.py', 'src/guarded_fisher.py', 'src/voc_dataset.py', 'src/training.py', 'short_contrast.py',
                     'src/training_diagnostics.py',
                     'models/fishernet_spp.py', 'src/fisher_layer.py', 'src/patches.py')})


def fisher_commit_budget(report, identity):
    if (report.get('status') != 'completed' or not report.get('probe_only')
            or report.get('updates_completed') != 2 or report.get('identity') != identity
            or len(report.get('probe_image_ids', [])) != 2):
        raise ValueError('matching completed independent all-patch Fisher memory probe required')
    stats = report['memory_statistics']
    if (stats['minimum']['commit_free_bytes'] < 2 * GIB
            or stats['minimum']['ram_free_bytes'] < 256 * 1024 ** 2):
        raise ValueError('Fisher memory probe sampled below safety floor')
    return max(8, math.ceil((stats['peak_added_commit_bytes'] + 2 * GIB + 512 * 1024 ** 2) / GIB))


def fisher_resource_check(commit_gib):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                  '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    status = windows_memory_status()
    if len(rows) != 1 or int(rows[0]) < 6144:
        raise RuntimeError(f'Fisher needs single GPU >=6144 MiB free; got {rows}')
    if status['ram_free_bytes'] < 2 * GIB or status['commit_free_bytes'] < commit_gib * GIB:
        raise RuntimeError(f'Fisher needs RAM >=2 GiB and commit >={commit_gib} GiB; got {status}')
    return dict(gpu_free_mib=int(rows[0]), **status)


def maximum_geometry_dataset(dataset, cfg):
    """Same train images/normalization and ALL patches; only probe has fixed scale."""
    scale = max(cfg['image_scales'])
    scored = []
    for index, (image_id, _, _) in enumerate(dataset.items):
        with Image.open(dataset.base / 'JPEGImages' / f'{image_id}.jpg') as image:
            w, h = image.size
        ratio = scale / max(w, h)
        rw, rh = max(1, round(w * ratio)), max(1, round(h * ratio))
        patches = sum(max(0, (rw - size) // cfg['patch_step'] + 1)
                      * max(0, (rh - size) // cfg['patch_step'] + 1) for size in cfg['patch_scales'])
        scored.append((patches, rw * rh, index))
    largest = sorted(scored, reverse=True)[:2]
    if len(largest) != 2:
        raise ValueError('two actual train geometries required for Fisher qualification')
    dataset.transform = ResizeLongestSide(scale, horizontal_flip=False)
    indices = [index for _, _, index in largest]
    return Subset(dataset, indices), [dataset.items[index][0] for index in indices]


class FisherMemory:
    def __init__(self, enabled, cfg, source, summary, target_steps, probe=False, qualification=None):
        self.enabled, self.cfg, self.source, self.target_steps = enabled, cfg, source, target_steps
        self.path = Path(summary) if summary else None
        self.probe, self.qualification = probe, qualification
        self.monitor = None
        self.completed = 0
        self.probe_image_ids = []

    def __enter__(self):
        if not self.enabled:
            return self
        if (training_split(self.cfg) != 'train' or self.cfg.get('model_variant') != 'spp'
                or self.cfg.get('patch_coordinate_mode') != 'resized' or self.cfg.get('max_patches') is not None
                or self.cfg.get('fisher_micro_batch_size') != 1
                or self.cfg.get('fisher_gradient_accumulation_steps') != 2
                or self.cfg.get('device') != 'cuda' or not torch.cuda.is_available()
                or self.path is None or not self.source):
            raise ValueError('guarded Fisher requires fresh train-only all-patch SPP, batch 1x2, CUDA and new summary')
        self.identity = fisher_memory_identity(self.source, self.cfg)
        if self.probe:
            budget = 12  # own bounded measurement, not a whole/GMM gate
            self.reclaim_commit_below = 4 * GIB
        else:
            if not self.qualification:
                raise ValueError('guarded Fisher requires independent --memory-qualification')
            report = json.loads(Path(self.qualification).read_text(encoding='utf-8'))
            budget = fisher_commit_budget(report, self.identity)
            # Mixed resolutions retained far more idle CUDA segments than the
            # square probe (actual 100-step evidence). Keep a conservative
            # measured worst-step commit reserve before subsequent forwards;
            # only unused allocator blocks can be reclaimed, not model state.
            self.reclaim_commit_below = max(4 * GIB,
                report['memory_statistics']['peak_added_commit_bytes'] + 2 * GIB)
        self.admission = fisher_resource_check(budget)
        protect_output(self.path)
        self.monitor = MemoryMonitor(True, reclaim_unused_cuda_cache, ram_reclaim_below_bytes=512 * 1024 ** 2,
                                     commit_reclaim_below_bytes=self.reclaim_commit_below)
        self.monitor.start()
        torch.cuda.reset_peak_memory_stats()
        print(f'fisher_memory_admission={self.admission}; commit_budget_gib={budget}; '
              f'cache_reclaim_commit_below_bytes={self.reclaim_commit_below}; probe_only={self.probe}', flush=True)
        return self

    def guard(self):
        if self.monitor is not None:
            self.monitor.guard()

    def after_update(self, step):
        if not self.enabled:
            return
        if step != self.completed + 1:
            raise ValueError('fresh Fisher updates must advance exactly one step')
        self.completed = step
        self.guard()
        if self.probe and step == 2:
            stats = self.monitor.minimum
            if stats['commit_free_bytes'] < 2 * GIB or stats['ram_free_bytes'] < 256 * 1024 ** 2:
                raise RuntimeError('Fisher all-patch probe sampled below safety floor')

    def __exit__(self, error_type, error, tb):
        if self.monitor is None:
            return False
        report = dict(stage='fisher', probe_only=self.probe, identity=self.identity,
                      updates_completed=self.completed, target_steps=self.target_steps,
                      probe_image_ids=self.probe_image_ids,
                      status='completed' if error is None and self.completed == self.target_steps else 'stopped',
                      error=str(error) if error else None, admission=self.admission,
                      cache_reclaim_commit_below_bytes=self.reclaim_commit_below,
                      memory_statistics=self.monitor.finish(),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      note='all patches and optimizer unchanged; sampled qualification does not guarantee 40000-step safety')
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(self.path)
        print(f'fisher_memory_summary={self.path}; status={report["status"]}', flush=True)
        return False
