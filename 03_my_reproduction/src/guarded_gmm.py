"""Independent, opt-in memory qualification for descriptor sampling and CPU EM.

No EM equations, sampling quotas or random seeds are changed. A separate probe
has no initialization/cache output: two largest train geometries and synthetic
full-shape CPU EM (two iterations) only qualify a monitored resource budget.
"""

import gc
import json
import math
from pathlib import Path
import subprocess
import warnings

import numpy as np
from PIL import Image
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
import torch
from torch.utils.data import DataLoader, Subset

from short_contrast import MemoryMonitor, reclaim_unused_cuda_cache, windows_memory_status
from src.gmm_sampling import descriptor_metadata, file_sha256, sample_patch_indices
from src.training import protect_output, training_split
from src.voc_dataset import ResizeLongestSide

GIB = 1024 ** 3


def gmm_memory_identity(source, cfg):
    root = Path(__file__).resolve().parents[1]
    return dict(descriptors=descriptor_metadata(source, cfg, cfg['gmm_max_images'], cfg['gmm_max_descriptors']),
                components=cfg['num_components'], max_iter=cfg['gmm_max_iter'], reg_covar=1e-6,
                num_workers=cfg['num_workers'], prefetch=cfg.get('fisher_prefetch_factor', 2),
                sklearn=sklearn.__version__,
                implementation={name: file_sha256(root / name) for name in
                                ('initialize_fisher.py', 'src/guarded_gmm.py', 'src/gmm_sampling.py')})


def qualified_commit_budget(report, identity):
    if (report.get('status') != 'completed' or not report.get('probe_only')
            or report.get('identity') != identity or report.get('probe_images_completed') != 2
            or report.get('probe_em_iterations') != 2 or not report.get('probe_em_finite')):
        raise ValueError('matching completed independent GMM memory probe required')
    stats = report['memory_statistics']
    if (stats['minimum']['commit_free_bytes'] < 2 * GIB
            or stats['minimum']['ram_free_bytes'] < 256 * 1024 ** 2):
        raise ValueError('GMM memory probe sampled below safety floor')
    # The probe starts before checkpoint/model/worker allocations. Include
    # sampled added commit, a 2 GiB floor and 512 MiB additional margin.
    return max(4, math.ceil((stats['peak_added_commit_bytes'] + 2 * GIB + 512 * 1024 ** 2) / GIB))


def gmm_resource_check(commit_gib):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                    '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    value = windows_memory_status()
    if len(rows) != 1 or int(rows[0]) < 6144:
        raise RuntimeError(f'GMM needs single GPU >=6144 MiB free; got {rows}')
    if value['ram_free_bytes'] < 2 * GIB or value['commit_free_bytes'] < commit_gib * GIB:
        raise RuntimeError(f'GMM needs RAM >=2 GiB and commit >={commit_gib} GiB; got {value}')
    return dict(gpu_free_mib=int(rows[0]), **value)


class GMMMemory:
    def __init__(self, enabled, cfg, source, summary, probe=False, qualification=None):
        self.enabled, self.cfg, self.source = enabled, cfg, source
        self.path = Path(summary) if summary else None
        self.probe, self.qualification = probe, qualification
        self.monitor = None
        self.identity = None
        self.completed = False
        self.phase = 'admission'
        self.images = self.em_iterations = 0
        self.em_finite = False

    def __enter__(self):
        if not self.enabled:
            return self
        if (training_split(self.cfg) != 'train' or self.cfg.get('gmm_sampling_mode') != 'uniform_per_image'
                or self.path is None or self.cfg.get('device') != 'cuda' or not torch.cuda.is_available()):
            raise ValueError('guarded GMM requires train-only uniform sampling, CUDA and a new summary')
        self.identity = gmm_memory_identity(self.source, self.cfg)
        if self.probe:
            # Independent bounded measurement, not borrowed from whole/Fisher.
            # mmap source + ~100 MB descriptors + CPU EM workspace; admission
            # conservatively reserves 12 GiB, followed by sampled floor checks.
            budget = 12
        else:
            if not self.qualification:
                raise ValueError('formal guarded GMM needs --memory-qualification')
            report = json.loads(Path(self.qualification).read_text(encoding='utf-8'))
            budget = qualified_commit_budget(report, self.identity)
        self.admission = gmm_resource_check(budget)
        protect_output(self.path)
        self.monitor = MemoryMonitor(True, reclaim_unused_cuda_cache, ram_reclaim_below_bytes=512 * 1024 ** 2)
        self.monitor.start()
        torch.cuda.reset_peak_memory_stats()
        print(f'gmm_memory_admission={self.admission}; commit_budget_gib={budget}; probe_only={self.probe}', flush=True)
        return self

    def guard(self, phase=None):
        if phase:
            self.phase = phase
        if self.monitor is not None:
            self.monitor.guard()

    def __exit__(self, error_type, error, tb):
        if self.monitor is None:
            return False
        stats = self.monitor.finish()
        report = dict(stage='gmm', probe_only=self.probe, identity=self.identity,
                      status='completed' if self.completed and error is None else 'stopped',
                      error=str(error) if error else None, last_phase=self.phase,
                      admission=self.admission, memory_statistics=stats,
                      probe_images_completed=self.images, probe_em_iterations=self.em_iterations,
                      probe_em_finite=self.em_finite,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      note='sampled qualification, not a guarantee against all transient or later memory pressure')
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(self.path)
        print(f'gmm_memory_summary={self.path}; status={report["status"]}', flush=True)
        return False


class GuardedGaussianMixture(GaussianMixture):
    """Superclass math/parameters unchanged; guard between initialization/EM phases."""
    def _memory_guard(self, phase):
        guard = getattr(self, 'memory_guard', None)
        if guard is not None:
            guard(phase)

    def _initialize_parameters(self, *args, **kwargs):
        self._memory_guard('cpu_initialization_before')
        result = super()._initialize_parameters(*args, **kwargs)
        self._memory_guard('cpu_initialization_after')
        return result

    def _e_step(self, *args, **kwargs):
        self._memory_guard('cpu_e_step_before')
        result = super()._e_step(*args, **kwargs)
        self._memory_guard('cpu_e_step_after')
        return result

    def _m_step(self, *args, **kwargs):
        self._memory_guard('cpu_m_step_before')
        result = super()._m_step(*args, **kwargs)
        self._memory_guard('cpu_m_step_after')
        return result


def run_gmm_memory_probe(model, dataset, cfg, memory, device):
    """Separate process/RNG stream; no descriptor cache or usable GMM saved."""
    memory.guard('find_largest_train_geometries')
    scale = max(cfg['image_scales'])
    scored = []
    for index, (image_id, _, _) in enumerate(dataset.items):
        with Image.open(dataset.base / 'JPEGImages' / f'{image_id}.jpg') as image:
            w, h = image.size
        factor = scale / max(w, h)
        scored.append((round(w * factor) * round(h * factor), index))
    indices = [index for _, index in sorted(scored, reverse=True)[:2]]
    if len(indices) != 2:
        raise ValueError('two actual train images required for GMM probe')
    dataset.transform = ResizeLongestSide(scale, horizontal_flip=False)
    options = dict(batch_size=1, num_workers=cfg['num_workers'], shuffle=False)
    if cfg['num_workers']:
        options['prefetch_factor'] = cfg.get('fisher_prefetch_factor', 2)
    loader = DataLoader(Subset(dataset, indices), **options)
    rng = np.random.default_rng(cfg['seed'])
    with torch.inference_mode():
        for ordinal, batch in enumerate(loader):
            memory.guard('probe_maximum_geometry')
            if cfg.get('patch_coordinate_mode', 'resized') == 'original':
                images, _, ids, original_sizes = batch
            else:
                images, _, ids = batch
                original_sizes = None
            count = len(model._image_boxes(*images.shape[-2:], original_sizes, images.device))
            selected = sample_patch_indices(count, ordinal, cfg['gmm_max_images'], cfg['gmm_max_descriptors'], rng)
            descriptors = model.extract_descriptors(images.to(device),
                            original_sizes.to(device) if original_sizes is not None else None,
                            patch_indices=torch.from_numpy(selected).to(device))
            if not bool(torch.isfinite(descriptors).all()):
                raise FloatingPointError('nonfinite descriptor in GMM memory probe')
            torch.cuda.synchronize()
            print(f'gmm_probe_image={ids}; shape={tuple(images.shape)}; descriptors={len(selected)}', flush=True)
            memory.images += 1
            del descriptors, images, original_sizes, batch
            memory.guard()
    del loader
    gc.collect()
    reclaim_unused_cuda_cache()
    memory.guard('probe_full_shape_cpu_em')
    # float32 matches the real CPU descriptor array; KMeans + two full EM steps.
    samples = np.random.default_rng(cfg['seed']).standard_normal(
        (cfg['gmm_max_descriptors'], cfg['descriptor_dim']), dtype=np.float32)
    gmm = GuardedGaussianMixture(n_components=cfg['num_components'], covariance_type='diag',
                                max_iter=2, tol=0, reg_covar=1e-6, random_state=cfg['seed'], verbose=2)
    gmm.memory_guard = memory.guard
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        gmm.fit(samples)
    memory.em_iterations = int(gmm.n_iter_)
    memory.em_finite = bool(np.isfinite(gmm.means_).all() and np.isfinite(gmm.covariances_).all()
                            and np.isfinite(gmm.lower_bound_))
    if memory.em_iterations != 2 or not memory.em_finite:
        raise FloatingPointError('CPU EM memory probe incomplete or nonfinite')
    memory.guard('probe_complete')
    if (memory.monitor.minimum['commit_free_bytes'] < 2 * GIB
            or memory.monitor.minimum['ram_free_bytes'] < 256 * 1024 ** 2):
        raise RuntimeError('independent GMM memory probe sampled below safety floor')
    memory.completed = True


class DescriptorBuffer:
    """Preallocate exactly the requested descriptor budget; preserve row order/dtype."""
    def __init__(self, capacity, dimension):
        self.capacity, self.dimension, self.size = capacity, dimension, 0
        self.data = None

    def append(self, rows):
        if rows.ndim != 2 or rows.shape[1] != self.dimension or self.size + len(rows) > self.capacity:
            raise ValueError('invalid descriptor buffer append')
        if self.data is None:
            self.data = np.empty((self.capacity, self.dimension), dtype=rows.dtype)
        if self.data.dtype != rows.dtype:
            raise ValueError('descriptor dtype changed')
        self.data[self.size:self.size + len(rows)] = rows
        self.size += len(rows)

    def array(self):
        if self.data is None or not self.size:
            raise ValueError('no descriptors collected')
        return self.data[:self.size]
