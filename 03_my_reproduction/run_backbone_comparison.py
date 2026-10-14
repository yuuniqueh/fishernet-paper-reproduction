"""Fixed-budget DINOv2 validation and DINOv2/DINOv3 trainval/test experiments.

The existing DINOv3 validation configuration defines the common hyperparameters.
Test is evaluated only once per backbone, using the final 40000-step checkpoint.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "4"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = os.pathsep.join((str(ROOT / ".deps"), str(ROOT), os.environ.get("PYTHONPATH", "")))
SOURCE_REPORT = ROOT / "results/dinov3_40000_20261011.json"
BACKBONE_KEYS = {"backbone", "backbone_model_name", "backbone_backend", "backbone_repo",
                 "backbone_weights", "backbone_arch", "backbone_local_files_only", "backbone_config"}
OPERATIONAL_KEYS = {"gmm_descriptor_cache", "backbone_cache_dir", "train_split"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def common_parameters(config):
    return {key: value for key, value in config.items() if key not in BACKBONE_KEYS | OPERATIONAL_KEYS}


def prepare(args, output):
    import yaml
    original = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))["provenance"]["config"]
    base = {key: value for key, value in original.items() if key not in BACKBONE_KEYS | OPERATIONAL_KEYS}
    base.update(backbone_cache_dir=".model-cache", freeze_backbone=False)
    source = ROOT / ".model-cache/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056"
    assert (source / "model.safetensors").is_file() and (source / "config.json").is_file()
    weights = Path(args.dinov3_weights).resolve()
    repository = Path(args.dinov3_repo).resolve()
    assert weights.is_file() and (repository / "hubconf.py").is_file()
    jobs = []
    for name, kind, split, evaluation in [
        ("dinov2_validation", "dinov2", "train", "val"),
        ("dinov2_test", "dinov2", "trainval", "test"),
        ("dinov3_test", "dinov3", "trainval", "test"),
    ]:
        directory = output / name
        directory.mkdir(exist_ok=True)
        config = dict(base, backbone=kind, train_split=split,
                      backbone_model_name=f"facebook/{'dinov2-small' if kind == 'dinov2' else 'dinov3-vits16-pretrain-lvd1689m'}")
        portable = dict(config)
        if kind == "dinov2":
            config.update(backbone_backend="transformers", backbone_local_files_only=True)
        else:
            config.update(backbone_backend="official", backbone_repo=str(repository),
                          backbone_weights=str(weights), backbone_arch="dinov3_vits16")
        config["gmm_descriptor_cache"] = str(directory / "descriptors.npz")
        assert common_parameters(config) == common_parameters(original)
        config_path = directory / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        filename = f"voc2007_{kind}_40000{'_trainval' if split == 'trainval' else ''}.yaml"
        portable_path = ROOT / "configs" / filename
        if portable_path.exists():
            assert yaml.safe_load(portable_path.read_text(encoding="utf-8")) == portable
        else:
            portable_path.write_text(yaml.safe_dump(portable, sort_keys=False), encoding="utf-8")
        jobs.append(dict(name=name, backbone=kind, train_split=split, evaluation_split=evaluation,
                         directory=str(directory), config=str(config_path), config_sha256=sha(config_path)))
    voc = (ROOT / base["dataset_root"] / "VOC2007").resolve()
    ids = {split: (voc / f"ImageSets/Main/{split}.txt").read_text().split()
           for split in ("train", "val", "trainval", "test")}
    assert {key: len(value) for key, value in ids.items()} == dict(train=2501, val=2510, trainval=5011, test=4952)
    assert not set(ids["train"]) & set(ids["val"])
    assert set(ids["trainval"]) == set(ids["train"]) | set(ids["val"])
    assert not set(ids["trainval"]) & set(ids["test"])
    assert all((voc / f"JPEGImages/{item}.jpg").is_file() for item in ids["trainval"] + ids["test"])
    plan = dict(created_at=now(), jobs=jobs, source_report=str(SOURCE_REPORT),
                source_report_sha256=sha(SOURCE_REPORT), common_parameters=common_parameters(original),
                source_code_sha256={str(path.relative_to(ROOT)): sha(path) for path in ROOT.rglob("*.py")
                                    if not any(part.startswith(".") for part in path.relative_to(ROOT).parts)},
                pretrained_sources={"dinov2": {"revision": source.name, "weights_sha256": sha(source / "model.safetensors")},
                                    "dinov3": {"weights_sha256": sha(weights), "repository": str(repository)}},
                split_counts={key: len(value) for key, value in ids.items()},
                split_sha256={split: sha(voc / f"ImageSets/Main/{split}.txt") for split in ids},
                test_policy="Fixed 9000/40000 steps, identical common parameters, final checkpoint only; no test-based selection or retuning.",
                comparison_limit="This matches the existing DINOv3 extension configuration, not all original-paper settings. Backbone family, patch geometry, pretrained weights and backend differ.")
    write_json(output / "experiment_plan.json", plan)
    return plan


class Experiment:
    def __init__(self, output, plan):
        self.output, self.plan = output, plan
        self.state = dict(status="running", started_at=now(), runner_pid=os.getpid(), stages=[], completed_jobs=[])

    def update(self, **values):
        self.state.update(values, updated_at=now())
        write_json(self.output / "status.json", self.state)

    def run(self, job, stage, arguments):
        directory = Path(job["directory"])
        logfile = directory / f"{stage}.log"
        command = [sys.executable, "-u", "-B", *map(str, arguments)]
        record = dict(job=job["name"], stage=stage, command=command, started_at=now(), logfile=str(logfile))
        self.state["stages"].append(record)
        self.update(job=job["name"], stage=stage, step=0)
        print(f"{now()} START {job['name']}/{stage}", flush=True)
        with logfile.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
            self.update(child_pid=process.pid)
            last_step = 0
            for line in process.stdout:
                log.write(line)
                log.flush()
                match = re.match(r"step=(\d+)/(\d+) loss=([\d.eE+-]+)", line)
                if match:
                    last_step = int(match[1])
                    if last_step % 500 == 0:
                        self.update(step=last_step, target_steps=int(match[2]), last_loss=float(match[3]))
                        print(f"{now()} {job['name']}/{stage} {line.strip()}", flush=True)
                if stage == "fisher_train" and job["evaluation_split"] == "val" and line.startswith("saved="):
                    if last_step in {5000, 10000, 20000, 30000, 40000}:
                        target = directory / f"fisher_step_{last_step:05d}.pt"
                        if not target.exists():
                            shutil.copyfile(directory / "fisher_trained.pt", target)
            returncode = process.wait()
        record.update(finished_at=now(), returncode=returncode)
        self.update(child_pid=None)
        if returncode:
            raise RuntimeError(f"{job['name']}/{stage} exit {returncode}; see {logfile}")
        print(f"{now()} DONE {job['name']}/{stage}", flush=True)

    def evaluate(self, job, name, checkpoint, expected_step):
        import numpy as np
        import yaml
        from evaluate import load_raw_voc_labels, report_ap
        directory = Path(job["directory"]) / name
        self.run(job, name, ["evaluate.py", "--config", job["config"], "--checkpoint", checkpoint,
                            "--protocol", "svm", "--split", job["evaluation_split"],
                            "--fit-split", job["train_split"], "--output-dir", directory])
        metrics = json.loads((directory / "voc2007_svm_metrics.json").read_text(encoding="utf-8"))
        cfg = yaml.safe_load(Path(job["config"]).read_text(encoding="utf-8"))
        voc = (ROOT / cfg["dataset_root"] / "VOC2007").resolve()
        expected = (voc / f"ImageSets/Main/{job['evaluation_split']}.txt").read_text().split()
        with np.load(directory / "voc2007_svm_predictions.npz") as saved:
            ids, labels, scores = saved["image_ids"].tolist(), saved["labels"], saved["scores"]
            assert ids == expected and len(ids) == len(set(ids))
            assert labels.shape == scores.shape == (len(expected), 20)
            assert np.array_equal(labels, load_raw_voc_labels(voc, job["evaluation_split"], ids))
            assert np.isfinite(scores).all()
            _, recomputed = report_ap(labels, scores)
        assert abs(recomputed - metrics["mAP"]) < 1e-12
        assert metrics["svm_all_converged"] and metrics["svm_fit_split"] == job["train_split"]
        metadata = metrics["checkpoint_metadata"]
        assert metadata["train_split"] == job["train_split"]
        assert not any(metadata.get(flag, False) for flag in ("debug_run", "diagnostic_run", "preparation_run"))
        if expected_step:
            assert metadata["step"] == expected_step
        audit = dict(images=len(ids), exact_ids=True, exact_labels=True, all_finite=True,
                     recomputed_mAP=recomputed, prediction_sha256=sha(directory / "voc2007_svm_predictions.npz"))
        write_json(directory / "prediction_audit.json", audit)
        return dict(metrics=metrics, audit=audit, predictions=str(directory / "voc2007_svm_predictions.npz"))

    def audit_training(self, job):
        import torch
        import yaml
        from models.modern_backbone import DenseBackbone
        directory = Path(job["directory"])
        whole = torch.load(directory / "whole_image.pt", map_location="cpu", weights_only=False, mmap=True)
        initialized = torch.load(directory / "fisher_gmm.pt", map_location="cpu", weights_only=False, mmap=True)
        trained = torch.load(directory / "fisher_trained.pt", map_location="cpu", weights_only=False, mmap=True)
        assert whole["step"] == 9000 and trained["step"] == 40000
        assert whole["scheduler"]["last_epoch"] == 9000 and trained["scheduler"]["last_epoch"] == 40000
        assert initialized["gmm_converged"] and initialized["source_whole_image_step"] == 9000
        for checkpoint in (whole, initialized, trained):
            assert checkpoint["config"]["train_split"] == job["train_split"]
            assert checkpoint["config"]["freeze_backbone"] is False
            assert not any(checkpoint.get(flag, False) for flag in ("debug_run", "diagnostic_run", "preparation_run"))
            assert all(torch.isfinite(value).all().item() for value in checkpoint["model"].values())
        cfg = yaml.safe_load(Path(job["config"]).read_text(encoding="utf-8"))
        initial_backbone = DenseBackbone(cfg, pretrained=True).state_dict()
        whole_changes = sum(not torch.equal(value, whole["model"]["backbone." + name])
                            for name, value in initial_backbone.items() if value.is_floating_point())
        assert whole_changes > 0
        updates = {}
        for prefix in ("backbone.", "descriptor_projection.", "fisher.", "classifier."):
            names = [name for name, value in initialized["model"].items()
                     if name.startswith(prefix) and value.is_floating_point()]
            changed = sum(not torch.equal(initialized["model"][name], trained["model"][name]) for name in names)
            assert changed > 0, prefix
            updates[prefix.rstrip(".")] = dict(tensors=len(names), changed_tensors=changed)
        audit = dict(whole_steps=9000, fisher_steps=40000, whole_changed_backbone_tensors=whole_changes,
                     fisher_parameter_updates=updates, all_checkpoints_finite=True, gmm_converged=True,
                     gmm_iterations=initialized["gmm_iterations"],
                     checkpoint_sha256={name: sha(directory / name) for name in ("whole_image.pt", "fisher_gmm.pt", "fisher_trained.pt")})
        write_json(directory / "training_audit.json", audit)
        return audit

    def job(self, job):
        directory, config = Path(job["directory"]), job["config"]
        assert sha(config) == job["config_sha256"]
        self.run(job, "whole_train", ["train_whole_image.py", "--config", config, "--output", directory / "whole_image.pt"])
        self.run(job, "gmm", ["initialize_fisher.py", "--config", config, "--whole-image-checkpoint", directory / "whole_image.pt", "--output", directory / "fisher_gmm.pt"])
        self.run(job, "fisher_train", ["train.py", "--config", config, "--init-checkpoint", directory / "fisher_gmm.pt", "--output", directory / "fisher_trained.pt", "--diagnostics-jsonl", directory / "fisher_diagnostics.jsonl", "--diagnostics-every", "500"])
        audit = self.audit_training(job)
        evaluations = {"40000": self.evaluate(job, "evaluation_40000", directory / "fisher_trained.pt", 40000)}
        if job["evaluation_split"] == "val":
            evaluations["0"] = self.evaluate(job, "evaluation_initialized", directory / "fisher_gmm.pt", 0)
            for step in (5000, 10000, 20000, 30000):
                evaluations[str(step)] = self.evaluate(job, f"evaluation_{step:05d}", directory / f"fisher_step_{step:05d}.pt", step)
        summary = dict(status="completed", completed_at=now(), job=job, training_audit=audit, evaluations=evaluations)
        write_json(directory / "summary.json", summary)
        self.state["completed_jobs"].append(dict(name=job["name"], final_mAP=evaluations["40000"]["metrics"]["mAP"]))
        self.update()
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dinov3-repo", required=True)
    parser.add_argument("--dinov3-weights", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "status.json").exists():
        raise FileExistsError("An execution already exists; use a fresh output directory.")
    plan = prepare(args, output)
    if args.prepare_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    import torch
    assert torch.cuda.is_available()
    experiment = Experiment(output, plan)
    experiment.update(gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__)
    try:
        summaries = {}
        for job in plan["jobs"]:
            summaries[job["name"]] = experiment.job(job)
        write_json(output / "summary.json", dict(status="completed", completed_at=now(), plan=plan, experiments=summaries))
        experiment.update(status="completed", job=None, stage="completed", child_pid=None, completed_at=now())
        print(json.dumps(experiment.state["completed_jobs"], indent=2), flush=True)
    except BaseException as error:
        experiment.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
