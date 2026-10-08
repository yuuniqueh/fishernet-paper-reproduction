from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.training import (masked_sigmoid_loss, build_optimizer, assert_resume_compatible,
                          assert_model_compatible, protect_output, checked_loss_value, read_config,
                          ensure_finite_gradients, ensure_finite_checkpoint_state)
from src.training import stage_loss_reduction, assert_whole_source_loss_compatible, training_contract
from src.training import training_split, assert_initialization_split_compatible


class TinyModel(nn.Module):
    def __init__(self, dense=False):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        if not dense:
            self.fc6 = nn.Linear(2, 2)
            self.fc7 = nn.Linear(2, 2)
        self.descriptor_projection = nn.Linear(2, 2)
        self.fisher = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, 2)


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    return read_config(root / "configs/voc2007_alexnet.yaml", root)


def test_loss_matches_class_sum_and_extreme_logits_are_stable():
    logits = torch.tensor([[0.0, 1000.0, -1000.0], [1.0, -2.0, 3.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    valid = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    loss = masked_sigmoid_loss(logits, targets, valid)
    reference = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert torch.allclose(loss, (reference * valid).sum(dim=1).mean())
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[valid == 0], torch.zeros(2))


def test_twenty_class_loss_and_gradients_not_divided_by_twenty():
    logits = torch.zeros(2, 20, requires_grad=True)
    targets = torch.zeros_like(logits)
    valid = torch.ones_like(logits)
    new = masked_sigmoid_loss(logits, targets, valid, "batch_sum")
    old = masked_sigmoid_loss(logits, targets, valid, "valid_mean")
    new_grad, = torch.autograd.grad(new, logits)
    old_grad, = torch.autograd.grad(old, logits)
    assert torch.allclose(new, old * 20)
    assert torch.allclose(new_grad, old_grad * 20)


def test_explicit_stage_loss_separates_fisher_intervention_without_upstream_retraining(config):
    cfg = dict(config, loss_reduction="valid_mean", whole_image_loss_reduction="batch_sum")
    assert training_contract(cfg, "whole_image")["loss_reduction"] == "batch_sum"
    assert training_contract(cfg, "fisher")["loss_reduction"] == "valid_mean"
    source = dict(config=dict(config, loss_reduction="batch_sum"),
                  training_contract=dict(loss_reduction="batch_sum"))
    assert assert_whole_source_loss_compatible(source, cfg) == "batch_sum"
    gmm = dict(config=source["config"], source_whole_image_config=source["config"])
    assert assert_whole_source_loss_compatible(gmm, cfg, gmm=True) == "batch_sum"
    undeclared = dict(config, loss_reduction="valid_mean")
    with pytest.raises(ValueError, match="mismatch"):
        assert_whole_source_loss_compatible(source, undeclared)
    with pytest.raises(ValueError, match="mismatch"):
        assert_whole_source_loss_compatible(gmm, undeclared, gmm=True)
    with pytest.raises(ValueError):
        stage_loss_reduction(dict(config, whole_image_loss_reduction="unknown"), "whole_image")


def test_training_split_is_explicit_and_upstream_validation_leak_rejected(config):
    assert training_split(config) == "trainval"
    candidate = dict(config, train_split="train")
    for split in ("val", "test"):
        with pytest.raises(ValueError, match="must not enter training"):
            training_split(dict(config, train_split=split))
    with pytest.raises(ValueError, match="split mismatch"):
        assert_initialization_split_compatible(dict(config=config), candidate)
    gmm = dict(config=candidate, source_whole_image_config=config)
    with pytest.raises(ValueError, match="upstream"):
        assert_initialization_split_compatible(gmm, candidate, gmm=True)
    gmm["source_whole_image_config"] = candidate
    assert assert_initialization_split_compatible(gmm, candidate, gmm=True) == "train"


def test_legacy_resume_defaults_to_trainval_but_cannot_switch_split(config):
    opt, sched, contract = build_optimizer(TinyModel(), config, "fisher")
    contract.pop("train_split")
    checkpoint = dict(config=config, training_contract=contract,
                      optimizer=opt.state_dict(), scheduler=sched.state_dict())
    assert_resume_compatible(checkpoint, config, "fisher")
    with pytest.raises(ValueError, match="train_split"):
        assert_resume_compatible(checkpoint, dict(config, train_split="train"), "fisher")


def test_accumulation_matches_full_batch_with_different_mask_counts():
    logits = torch.randn(2, 20, dtype=torch.double, requires_grad=True)
    targets = torch.randint(0, 2, (2, 20)).double()
    valid = torch.ones_like(targets)
    valid[0, :7] = 0
    full_grad, = torch.autograd.grad(masked_sigmoid_loss(logits, targets, valid), logits)
    accumulated = sum(masked_sigmoid_loss(logits[i:i + 1], targets[i:i + 1], valid[i:i + 1]) / 2
                      for i in range(2))
    micro_grad, = torch.autograd.grad(accumulated, logits)
    assert torch.allclose(full_grad, micro_grad, atol=1e-12)


def test_all_difficult_is_connected_zero_and_invalid_inputs_rejected():
    logits = torch.randn(1, 20, requires_grad=True)
    loss = masked_sigmoid_loss(logits, torch.zeros_like(logits), torch.zeros_like(logits))
    loss.backward()
    assert loss.item() == 0 and torch.equal(logits.grad, torch.zeros_like(logits))
    with pytest.raises(ValueError):
        masked_sigmoid_loss(logits, logits[:, :2], logits)
    with pytest.raises(ValueError):
        masked_sigmoid_loss(logits, logits, logits, "typo")
    with pytest.raises(FloatingPointError):
        checked_loss_value(torch.tensor(float("nan")))


@pytest.mark.parametrize("dense", [False, True])
def test_fisher_groups_cover_parameters_and_update_with_correct_rates(config, dense):
    model = TinyModel(dense=dense)
    config.update(momentum=0.0, weight_decay=0.0)
    optimizer, _, contract = build_optimizer(model, config, "fisher")
    assert contract["base_lrs"] == [1e-4, 1e-4, 0.001, 0.1, 0.001]
    before = {id(p): p.detach().clone() for p in model.parameters()}
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    for group in optimizer.param_groups:
        for p in group["params"]:
            assert torch.allclose(before[id(p)] - p, torch.full_like(p, group["lr"]), atol=4e-8)


def test_whole_image_groups_and_uncovered_parameter_detection(config):
    model = TinyModel()
    del model.fisher
    _, _, contract = build_optimizer(model, config, "whole_image")
    assert contract["base_lrs"] == [0.001, 0.001, 0.01, 0.01]
    model.untracked = nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="exactly once"):
        build_optimizer(model, config, "whole_image")


def test_resume_contract_and_saved_optimizer_checked(config):
    model = TinyModel()
    optimizer, scheduler, contract = build_optimizer(model, config, "fisher")
    checkpoint = dict(config=config, optimizer=optimizer.state_dict(),
                      scheduler=scheduler.state_dict(), training_contract=contract)
    assert_resume_compatible(checkpoint, config, "fisher")
    workers_only = dict(config, num_workers=0)
    assert_resume_compatible(checkpoint, workers_only, "fisher")
    for key, value in [("loss_reduction", "batch_sum"), ("lr_descriptor_projection", 0.0001),
                       ("fisher_lr_milestones", []), ("roi_coordinate_mapping", "backbone_stride"),
                       ("new_layer_init_std", 0.01)]:
        changed = dict(config, **{key: value})
        with pytest.raises(ValueError, match="incompatible resume"):
            assert_resume_compatible(checkpoint, changed, "fisher")
    broken = deepcopy(checkpoint)
    broken["scheduler"]["base_lrs"][2] = 0.0001
    with pytest.raises(ValueError, match="LR groups"):
        assert_resume_compatible(broken, config, "fisher")


def test_legacy_checkpoint_can_resume_only_legacy_settings(config):
    original = dict(config)
    for key in ("loss_reduction", "lr_descriptor_projection", "fisher_lr_milestones", "fisher_lr_gamma"):
        del original[key]
    legacy = dict(original, loss_reduction="valid_mean", lr_descriptor_projection=0.001,
                  fisher_lr_milestones=[20000, 30000])
    opt, sched, _ = build_optimizer(TinyModel(), legacy, "fisher")
    checkpoint = dict(config=original, optimizer=opt.state_dict(), scheduler=sched.state_dict())
    assert_resume_compatible(checkpoint, legacy, "fisher")
    modern = dict(config, loss_reduction="batch_sum", lr_descriptor_projection=0.0001,
                  fisher_lr_milestones=[])
    with pytest.raises(ValueError, match="incompatible resume"):
        assert_resume_compatible(checkpoint, modern, "fisher")


def test_new_default_schedule_does_not_invent_decay(config):
    for key in ("fisher_lr_milestones", "lr_descriptor_projection"):
        del config[key]
    _, _, contract = build_optimizer(TinyModel(), config, "fisher")
    assert contract["milestones"] == [] and contract["base_lrs"][2] == config["lr_backbone"]


def test_model_mapping_check_and_protect_existing_files(config, tmp_path):
    checkpoint = {"config": config}
    assert_model_compatible(checkpoint, config)
    with pytest.raises(ValueError, match="roi_coordinate_mapping"):
        assert_model_compatible(checkpoint, dict(config, roi_coordinate_mapping="backbone_stride"))
    existing = tmp_path / "weights.pt"
    existing.touch()
    with pytest.raises(FileExistsError):
        protect_output(existing)
    protect_output(existing, resume=existing)


def test_finite_gradient_guard_does_not_clip_and_rejects_nan():
    model = nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 1000.)
    ensure_finite_gradients(model)
    assert all(torch.equal(p.grad, torch.full_like(p, 1000.)) for p in model.parameters())
    model.weight.grad[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        ensure_finite_gradients(model)


def test_checkpoint_guard_rejects_nonfinite_weights_and_momentum():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    ensure_finite_checkpoint_state(model, optimizer)
    with torch.no_grad():
        model.weight[0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="checkpoint NOT overwritten"):
        ensure_finite_checkpoint_state(model, optimizer)
    with torch.no_grad():
        model.weight[0, 0] = 1.
    optimizer.state[model.weight]["momentum_buffer"][0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="checkpoint NOT overwritten"):
        ensure_finite_checkpoint_state(model, optimizer)
def test_diagnostic_evidence_cannot_become_formal_training():
    from src.training import reject_diagnostic_checkpoint
    reject_diagnostic_checkpoint({"diagnostic_run": False})
    reject_diagnostic_checkpoint({})
    with pytest.raises(ValueError, match="diagnostic-only"):
        reject_diagnostic_checkpoint({"diagnostic_run": True})
    reject_diagnostic_checkpoint({"diagnostic_run": True}, diagnostic=True)
