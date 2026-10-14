"""Offline checks for token geometry, frozen state and checkpoint continuity."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from models.factory import build_model
from models.modern_backbone import DenseBackbone
from src.training import assert_model_compatible, read_config
import train
import models.modern_backbone as modern
from benchmark_backbone import sample_train_tokens, pca_projection
import numpy as np
from sklearn.decomposition import PCA


def tiny_config(kind="dinov3", frozen=False):
    return dict(backbone=kind, model_variant="dense", descriptor_dim=8, num_components=2,
                num_classes=20, freeze_backbone=frozen,
                backbone_config=dict(hidden_size=24, num_hidden_layers=1,
                                     num_attention_heads=3, intermediate_size=48,
                                     patch_size=16, image_size=32, num_register_tokens=4 if kind == "dinov3" else 0))


@pytest.mark.parametrize("kind", ["dinov2", "dinov3"])
def test_spatial_tokens_exclude_cls_and_registers_and_retain_borders(kind):
    pytest.importorskip("transformers")
    backbone = DenseBackbone(tiny_config(kind))
    backbone.eval()
    aligned = torch.randn(2, 3, 32, 48)
    with torch.no_grad():
        expected = backbone.network(pixel_values=aligned).last_hidden_state
        patches, cls = backbone.features(aligned)
    start = 5 if kind == "dinov3" else 1
    assert torch.equal(patches, expected[:, start:])
    assert torch.equal(cls, expected[:, 0])
    assert backbone(torch.randn(1, 3, 33, 49)).shape == (1, 12, 24)
    assert backbone.descriptor_count(33, 49) == 12


def test_frozen_resnet_keeps_batchnorm_and_backbone_unchanged():
    cfg = dict(tiny_config(), backbone="resnet50", freeze_backbone=True)
    model = build_model(cfg)
    model.train()
    assert not model.backbone.training
    assert not any(module.training for module in model.backbone.modules())
    before = model.backbone.network[1].running_mean.clone()
    model(torch.randn(2, 3, 64, 96)).sum().backward()
    assert torch.equal(model.backbone.network[1].running_mean, before)
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert model.descriptor_projection.weight.grad is not None


@pytest.mark.parametrize("frozen", [True, False])
def test_whole_to_fisher_transfer_and_gradient_path(frozen):
    pytest.importorskip("transformers")
    cfg = tiny_config(frozen=frozen)
    whole = build_model(cfg, "whole_image")
    model = build_model(cfg)
    model.backbone.load_state_dict(whole.backbone.state_dict())
    model.descriptor_projection.load_state_dict(whole.descriptor_projection.state_dict())
    images = torch.randn(2, 3, 32, 48)
    whole.eval()
    model.eval()
    torch.testing.assert_close(model.extract_descriptors(images).mean(1), whole.extract_descriptor(images))
    model.train()
    loss = model(images).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.descriptor_projection.weight.grad.isfinite().all()
    assert model.fisher.w.grad.isfinite().all()
    backbone_grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert bool(backbone_grads) != frozen
    assert all(g.isfinite().all() for g in backbone_grads)


def test_checkpoint_rejects_wrong_backbone_and_architecture():
    cfg = tiny_config()
    checkpoint = {"config": cfg}
    with pytest.raises(ValueError, match="backbone"):
        assert_model_compatible(checkpoint, dict(cfg, backbone="resnet50"))
    with pytest.raises(ValueError, match="hidden_size"):
        assert_model_compatible(checkpoint, dict(cfg, backbone_config={"hidden_size": 48}))
    with pytest.raises(ValueError, match="dense spatial"):
        build_model(dict(cfg, model_variant="spp"))


def test_dino_training_entrypoint_resume_restores_offline_architecture(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    root = Path(__file__).resolve().parents[1]
    cfg = read_config(root / "tests/fixtures/audit_smoke.yaml", root)
    cfg.update(tiny_config(frozen=True), device="cpu", pretrained_backbone=False)
    monkeypatch.setattr(train, "read_config", lambda *a: dict(cfg))
    output = tmp_path / "dino.pt"
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "unused", "--smoke-test",
                                    "--max-steps", "1", "--output", str(output)])
    train.main()
    first = torch.load(output, weights_only=False)
    assert first["step"] == 1
    # Restore architecture from checkpoint rather than asking the network.
    cfg.pop("backbone_config")
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "unused", "--smoke-test",
                                    "--max-steps", "2", "--resume", str(output), "--output", str(output)])
    train.main()
    resumed = torch.load(output, weights_only=False)
    assert resumed["step"] == 2
    assert resumed["config"]["backbone_config"]["hidden_size"] == 24


def test_official_local_weights_load_strictly_without_network(tmp_path, monkeypatch):
    (tmp_path / "hubconf.py").write_text("")
    class FakeDino(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim, self.patch_size = 24, 16
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(24, 24)])
            self.embed = torch.nn.Conv2d(3, 24, 16, stride=16)
        def forward_features(self, images):
            tokens = self.embed(images).flatten(2).transpose(1, 2)
            return dict(x_norm_patchtokens=tokens, x_norm_clstoken=tokens.mean(1))
    reference = FakeDino()
    weights = tmp_path / "local.pth"
    torch.save(reference.state_dict(), weights)
    constructors = SimpleNamespace(__file__=str(tmp_path / "dinov3/hub/backbones.py"),
                                   dinov3_vits16=lambda **kwargs: FakeDino())
    monkeypatch.setattr(modern.importlib, "import_module", lambda name: constructors)
    cfg = dict(tiny_config(), backbone_backend="official", backbone_repo=str(tmp_path),
               backbone_arch="dinov3_vits16", backbone_weights=str(weights))
    loaded = DenseBackbone(cfg, pretrained=True)
    image = torch.randn(1, 3, 32, 48)
    torch.testing.assert_close(loaded(image), reference.forward_features(image)["x_norm_patchtokens"])
    assert cfg["backbone_config"]["hidden_size"] == 24
    torch.save({"wrong_key": torch.ones(1)}, weights)
    with pytest.raises(RuntimeError, match="state_dict"):
        DenseBackbone(cfg, pretrained=True)


def test_probe_sampling_covers_train_images_and_pca_layer_matches_sklearn():
    tokens = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    sampled = sample_train_tokens(tokens, 10, 42)
    assert len(sampled) == 10
    assert set((sampled[:, 0] // 30).astype(int)) == {0, 1, 2, 3}
    rng = np.random.default_rng(42)
    values = rng.normal(size=(100, 6)).astype(np.float32)
    pca = PCA(3, whiten=True).fit(values)
    layer = pca_projection(pca, "cpu")
    np.testing.assert_allclose(layer(torch.from_numpy(values)).detach().numpy(), pca.transform(values), atol=1e-6)
