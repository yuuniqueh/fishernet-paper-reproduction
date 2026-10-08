"""Opt-in diagnostics. Observes training without changing its objective/update."""

import json
from pathlib import Path

import torch


def tensor_summary(value):
    value = value.detach()
    return {"shape": list(value.shape), "finite": bool(torch.isfinite(value).all()),
            "absmax": float(value.abs().max()), "norm": float(value.norm(dtype=torch.float64))}


class TrainingDiagnostics:
    def __init__(self, path, model):
        self.path = Path(path)
        # A new process must never truncate an existing diagnostic record.
        self.stream = self.path.open("x", encoding="utf-8")
        self.activations = {}
        self.active = True
        self.handles = []
        self.fisher_module = model.fisher
        self.previous_observer = getattr(model.fisher, "diagnostic_observer", None)
        if hasattr(model.fisher, "num_components"):
            model.fisher.diagnostic_observer = self._fisher_observer
        self.before_parameters = {}
        for name in ("descriptor_projection", "fisher", "classifier"):
            module = getattr(model, name)
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def hook(module, inputs, output):
            if self.active:
                self.activations[name] = tensor_summary(output)
        return hook

    @torch.no_grad()
    def _fisher_observer(self, x, z, gamma, mu, sigma):
        if not self.active:
            return
        probabilities = gamma.detach()
        winners = probabilities.argmax(-1)
        counts = torch.bincount(winners.flatten(), minlength=self.fisher_module.num_components)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
        self.activations["fisher_posterior"] = {
            "patch_count": x.shape[1], "hard_counts": counts.cpu().tolist(),
            "mean_occupancy": probabilities.mean((0, 1)).cpu().tolist(),
            "mean_max_probability": float(probabilities.max(-1).values.mean()),
            "mean_entropy": float(entropy.mean()),
            "z_absmax": float(z.detach().abs().max()),
            "first_order": tensor_summary(mu), "second_order": tensor_summary(sigma),
        }

    def write(self, **record):
        # Python JSON represents non-finite values explicitly for failure reports.
        self.stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.stream.flush()

    def snapshot(self, model, optimizer):
        groups = {}
        for group in optimizer.param_groups:
            grads = [p.grad.detach() for p in group["params"] if p.grad is not None]
            buffers = [optimizer.state[p]["momentum_buffer"].detach()
                       for p in group["params"] if "momentum_buffer" in optimizer.state[p]]
            groups[group["name"]] = {
                "lr": group["lr"],
                "grad_absmax": max((float(g.abs().max()) for g in grads), default=0.),
                "grad_norm": float(torch.stack([g.norm(dtype=torch.float64) for g in grads]).norm()) if grads else 0.,
                "momentum_norm": float(torch.stack([b.norm(dtype=torch.float64) for b in buffers]).norm()) if buffers else 0.,
            }
        params = {name: tensor_summary(p) for name, p in model.named_parameters()
                  if name.startswith("fisher.") or name == "classifier.weight"}
        details = {}
        for name, p in model.fisher.named_parameters():
            details[name] = {"parameter": tensor_summary(p)}
            if p.grad is not None:
                details[name]["gradient"] = tensor_summary(p.grad)
                flat_index = int(p.grad.detach().abs().argmax())
                details[name]["largest_gradient_index"] = list(torch.unravel_index(
                    torch.tensor(flat_index), p.shape))
                details[name]["largest_gradient_index"] = [int(i) for i in details[name]["largest_gradient_index"]]
            if "momentum_buffer" in optimizer.state[p]:
                details[name]["momentum"] = tensor_summary(optimizer.state[p]["momentum_buffer"])
        return {"groups": groups, "parameters": params, "fisher_details": details}

    @torch.no_grad()
    def capture_before_update(self, model):
        # All Fisher/head parameters plus explicitly named CNN/projection probes.
        # No full fc6/fc7 copies: keep observation memory bounded.
        probes = {"backbone.0.weight", "backbone.10.weight", "fc6.bias", "fc7.bias",
                  "descriptor_projection.weight", "descriptor_projection.bias"}
        self.before_parameters = {
            name: p.detach().clone() for name, p in model.named_parameters()
            if name.startswith(("fisher.", "classifier.")) or name in probes
        }

    @torch.no_grad()
    def update_snapshot(self, model):
        changes = {}
        for name, p in model.named_parameters():
            if name not in self.before_parameters:
                continue
            previous = self.before_parameters[name]
            delta = p.detach() - previous
            before_norm = float(previous.norm(dtype=torch.float64))
            summary = tensor_summary(delta)
            changes[name] = {**summary, "parameter_norm_before": before_norm,
                             "relative_update": summary["norm"] / before_norm if before_norm > 0 else None}
        self.before_parameters.clear()
        return changes

    def close(self):
        for handle in self.handles:
            handle.remove()
        if hasattr(self.fisher_module, "num_components"):
            self.fisher_module.diagnostic_observer = self.previous_observer
        self.stream.close()
