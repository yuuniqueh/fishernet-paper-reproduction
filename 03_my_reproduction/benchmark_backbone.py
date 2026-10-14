"""Frozen-backbone trial: train-only PCA/GMM/Fisher and global linear SVMs.

This bounded experiment tests pretrained features without whole-image or
end-to-end fine-tuning. It is deliberately reported separately from FisherNet
training. Token caches live on disk; only one batch is resident on the GPU.
"""

import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from evaluate import load_raw_voc_labels, normalize_fisher_vectors, report_ap
from models.modern_backbone import DenseBackbone, MODEL_IDS
from src.fisher_layer import FisherLayer
from src.gmm_init import initialize_fisher_from_gmm
from src.gmm_sampling import file_sha256
from src.training import read_config
from src.voc_dataset import VOCMultiLabel, VOC_CLASSES


def probe_transform(size):
    # Retain the entire image at a fixed size for efficient batched probing.
    return transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


@torch.inference_mode()
def extract_cache(backbone, dataset, directory, split, identity, batch_size, workers, device):
    ids = [item[0] for item in dataset.items]
    metadata = dict(identity, split=split, image_ids=ids)
    metadata_path = directory / f"{split}_cache.json"
    token_path, global_path = (directory / f"{split}_{kind}.npy" for kind in ("tokens", "global"))
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("token cache identity mismatch; choose a new output directory")
        tokens = np.load(token_path, mmap_mode="r")
        global_features = np.load(global_path, mmap_mode="r")
        if len(tokens) != len(ids) or len(global_features) != len(ids):
            raise ValueError("truncated token cache")
        print(f"reused_cache={split}; images={len(ids)}", flush=True)
        return tokens, global_features, ids
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False)
    tokens = global_features = None
    offset = 0
    started = time.perf_counter()
    for images, _, batch_ids in loader:
        patches, cls = backbone.features(images.to(device))
        pooled = torch.cat((cls, patches.mean(1)), dim=1) if backbone.kind != "resnet50" else cls
        if tokens is None:
            tokens = np.lib.format.open_memmap(str(token_path) + ".partial", mode="w+", dtype=np.float16,
                                              shape=(len(ids), *patches.shape[1:]))
            global_features = np.lib.format.open_memmap(str(global_path) + ".partial", mode="w+", dtype=np.float32,
                                                       shape=(len(ids), pooled.shape[1]))
        count = len(batch_ids)
        if ids[offset:offset + count] != list(batch_ids):
            raise RuntimeError("image ordering changed during extraction")
        tokens[offset:offset + count] = patches.cpu().numpy()
        global_features[offset:offset + count] = pooled.cpu().numpy()
        offset += count
        if offset % (batch_size * 10) == 0 or offset == len(ids):
            print(f"extract={split}; images={offset}/{len(ids)}; seconds={time.perf_counter()-started:.1f}", flush=True)
    tokens.flush()
    global_features.flush()
    del tokens, global_features
    Path(str(token_path) + ".partial").replace(token_path)
    Path(str(global_path) + ".partial").replace(global_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return np.load(token_path, mmap_mode="r"), np.load(global_path, mmap_mode="r"), ids


def sample_train_tokens(tokens, budget, seed):
    # Distribute the budget across ALL fitting images; never sample val/test.
    rng = np.random.default_rng(seed)
    chunks = []
    quota, extra = divmod(budget, len(tokens))
    for index, image_tokens in enumerate(tokens):
        count = min(len(image_tokens), quota + (index < extra))
        if count:
            selected = rng.choice(len(image_tokens), count, replace=False)
            chunks.append(np.asarray(image_tokens[selected], dtype=np.float32))
    if not chunks:
        raise ValueError("empty training descriptor sample")
    return np.concatenate(chunks)


def pca_projection(pca, device):
    layer = torch.nn.Linear(pca.n_features_in_, pca.n_components_).to(device)
    weight = pca.components_ / np.sqrt(pca.explained_variance_)[:, None]
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(weight, dtype=torch.float32, device=device))
        layer.bias.copy_(torch.as_tensor(-weight @ pca.mean_, dtype=torch.float32, device=device))
    return layer


@torch.inference_mode()
def encode_tokens(tokens, projection, fisher, batch_size, device):
    features = []
    for start in range(0, len(tokens), batch_size):
        batch = torch.from_numpy(np.array(tokens[start:start + batch_size], dtype=np.float32)).to(device)
        features.append(fisher(projection(batch)).cpu().numpy())
    return normalize_fisher_vectors(np.concatenate(features))


def fit_svms(train_features, eval_features, train_labels, eval_labels, seed, svm_c):
    scores = np.empty((len(eval_features), len(VOC_CLASSES)), dtype=np.float32)
    diagnostics, coefficients, intercepts = {}, [], []
    for index, name in enumerate(VOC_CLASSES):
        valid = train_labels[:, index] != 0
        if len(np.unique(train_labels[valid, index])) != 2:
            raise ValueError(f"SVM fitting split needs positives and negatives for {name}")
        svm = LinearSVC(C=svm_c, dual="auto", max_iter=10000, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            svm.fit(train_features[valid], train_labels[valid, index])
        messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        diagnostics[name] = dict(iterations=int(svm.n_iter_), converged=not messages, warnings=messages)
        scores[:, index] = svm.decision_function(eval_features)
        coefficients.append(svm.coef_[0])
        intercepts.append(svm.intercept_[0])
    per_class, mean_ap = report_ap(eval_labels, scores)
    return dict(mAP=mean_ap, AP=per_class, svm_diagnostics=diagnostics,
                svm_all_converged=all(item["converged"] for item in diagnostics.values())), scores, np.stack(coefficients), np.asarray(intercepts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-split", choices=("train", "trainval"), default="train")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--sample-descriptors", type=int, default=50000)
    args = parser.parse_args()
    if args.split == "val" and args.fit_split != "train":
        raise ValueError("held-out val requires fitting on train only")
    if min(args.image_size, args.batch_size, args.pca_dim, args.components, args.sample_descriptors) < 1:
        raise ValueError("sizes and budgets must be positive")
    root = Path(__file__).resolve().parent
    cfg = read_config(args.config, root)
    if cfg["backbone"] == "alexnet":
        raise ValueError("use a DINOv2/DINOv3/ResNet50 backbone configuration")
    directory = (root / args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "metrics.json"
    if result_path.exists():
        raise FileExistsError("completed experiment already exists; choose a new output directory")
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    cfg = dict(cfg, freeze_backbone=True)
    backbone = DenseBackbone(cfg, pretrained=True)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    backbone.to(device).eval()
    identity = dict(backbone=cfg["backbone"], model_name=cfg.get("backbone_model_name", MODEL_IDS.get(cfg["backbone"])),
                    revision=getattr(getattr(backbone.network, "config", None), "_commit_hash", None),
                    image_size=args.image_size, preprocessing="resize_square_imagenet_normalization",
                    token_dtype="float16", torch_version=torch.__version__)
    dataset_root = (root / cfg["dataset_root"]).resolve()
    identity["dataset_root"] = str(dataset_root)
    if cfg.get("backbone_backend") == "official":
        identity.update(backend="official", architecture=cfg["backbone_arch"],
                        weights_sha256=file_sha256(cfg["backbone_weights"]),
                        implementation_sha256={name: file_sha256(Path(cfg["backbone_repo"]) / name)
                            for name in ("hubconf.py", "dinov3/models/vision_transformer.py",
                                         "dinov3/hub/backbones.py")})
    identity["labels_sha256"] = {
        f"{name}_{split}": file_sha256(dataset_root / "VOC2007" / "ImageSets" / "Main" / f"{name}_{split}.txt")
        for split in (args.fit_split, args.split) for name in VOC_CLASSES
    }
    transform = probe_transform(args.image_size)
    train_ds = VOCMultiLabel(dataset_root, split=args.fit_split, transform=transform)
    eval_ds = VOCMultiLabel(dataset_root, split=args.split, transform=transform)
    if {item[0] for item in train_ds.items} & {item[0] for item in eval_ds.items}:
        raise ValueError("fitting and evaluation images overlap")
    started = time.perf_counter()
    train_tokens, train_global, train_ids = extract_cache(backbone, train_ds, directory, args.fit_split,
        identity, args.batch_size, cfg["num_workers"], device)
    eval_tokens, eval_global, eval_ids = extract_cache(backbone, eval_ds, directory, args.split,
        identity, args.batch_size, cfg["num_workers"], device)
    samples = sample_train_tokens(train_tokens, args.sample_descriptors, cfg["seed"])
    if args.pca_dim > min(samples.shape):
        raise ValueError("PCA dimension exceeds available fitting descriptors")
    with threadpool_limits(limits=4):
        pca = PCA(args.pca_dim, whiten=True, svd_solver="randomized", random_state=cfg["seed"])
        reduced = pca.fit_transform(samples)
        print(f"PCA fitted on {len(samples)} train descriptors; fitting GMM", flush=True)
        gmm = GaussianMixture(args.components, covariance_type="diag", max_iter=cfg.get("gmm_max_iter", 200),
                              reg_covar=1e-4, random_state=cfg["seed"])
        gmm.fit(reduced)
    if not gmm.converged_:
        raise RuntimeError("GMM did not converge; no result claimed")
    projection = pca_projection(pca, device)
    fisher = FisherLayer(args.pca_dim, args.components).to(device)
    initialize_fisher_from_gmm(fisher, gmm.means_, gmm.covariances_)
    train_fisher = encode_tokens(train_tokens, projection, fisher, args.batch_size, device)
    eval_fisher = encode_tokens(eval_tokens, projection, fisher, args.batch_size, device)
    labels = load_raw_voc_labels(dataset_root / "VOC2007", args.fit_split, train_ids)
    eval_labels = load_raw_voc_labels(dataset_root / "VOC2007", args.split, eval_ids)
    results = {}
    svm_state = {}
    with threadpool_limits(limits=4):
        for name, train_features, eval_features in (
            ("global_svm", normalize(np.array(train_global)), normalize(np.array(eval_global))),
            ("fisher_svm", train_fisher, eval_fisher),
        ):
            result, scores, coefficients, intercepts = fit_svms(train_features, eval_features, labels, eval_labels,
                                                                cfg["seed"], cfg["svm_C"])
            results[name] = result
            svm_state[name] = dict(coefficients=coefficients, intercepts=intercepts)
            np.savez_compressed(directory / f"{name}_predictions.npz", image_ids=eval_ids, scores=scores, labels=eval_labels)
            print(f"{name}: mAP={result['mAP']:.6f}; converged={result['svm_all_converged']}", flush=True)
    # The probe uses PCA rather than a VOC-trained descriptor layer. Preserve
    # fitted parameters and full provenance for independent replay.
    torch.save(dict(projection=projection.cpu().state_dict(), fisher=fisher.cpu().state_dict(),
                    config=cfg, identity=identity, svm=svm_state, fit_image_ids=train_ids,
                    evaluation_image_ids=eval_ids, protocol="frozen_backbone_probe"), directory / "probe_encoder.pt")
    report = dict(protocol="frozen_backbone_probe", ap_protocol="VOC2007 11-point interpolated AP",
                  fit_split=args.fit_split, evaluation_split=args.split, identity=identity,
                  train_images=len(train_ids), evaluation_images=len(eval_ids), seed=cfg["seed"],
                  image_size=args.image_size, pca_dim=args.pca_dim, num_components=args.components,
                  sampled_train_descriptors=len(samples), gmm_converged=bool(gmm.converged_),
                  gmm_iterations=int(gmm.n_iter_), svm_C=cfg["svm_C"],
                  normalization="global: L2; Fisher: signed sqrt + L2", backbone_finetuned=False,
                  global_features="CLS + mean spatial tokens (CNN: mean spatial features)",
                  elapsed_seconds=time.perf_counter() - started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                  results=results)
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={result_path}", flush=True)


if __name__ == "__main__":
    main()
