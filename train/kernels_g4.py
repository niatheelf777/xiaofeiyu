#!/usr/bin/env python3
"""Unified Gemma 4 (`12B`, `26B-A4B`, `E4B`) BF16 Non-QKV LoRA & DPO Kernels.

Implements:
1. Upper-layer non-QKV target selection (`{o_proj, gate_proj, up_proj, down_proj}`, 0 QKV modules):
   - `12b` (`top12_no_qkv`): layers `36..47` (48 modules)
   - `e4b` (`e4b_upper20_no_qkv`): layers `22..41` (80 modules)
   - `26b-a4b` (`26b_a4b_top10_no_qkv`): layers `20..29` (40 modules, dense/shared projections)
2. Exact `<|"|>` (`52`) per-token forward-pass gating (`set_model_token_gate_mask`) so training
   is 100% matched to Split-Phase inference.
3. Exact Gemma 4 FP32 softcap (`30.0 * tanh(logits / 30.0)`) chunked NLL + single-model DPO
   (`policy` vs frozen `dpo_ref` branch switching).
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING, Any

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    HAS_TORCH = True
except ImportError:  # pragma: no cover - allows CPU dry-run/inspection without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    checkpoint = None  # type: ignore[assignment]
    HAS_TORCH = False

if TYPE_CHECKING:
    import torch as _torch

sys.dont_write_bytecode = True

CARTRIDGE_IDS = ("neko_clingy", "older_tease", "brat_mesugaki", "fragile_porcelain")
CARTRIDGE_LABELS = {
    "neko_clingy": "家养粘人猫娘",
    "older_tease": "坏心眼年上御姐",
    "brat_mesugaki": "恶劣雌小鬼",
    "fragile_porcelain": "易碎体温症",
}

NON_QKV_NAMES = frozenset(("o_proj", "gate_proj", "up_proj", "down_proj"))
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.([a-z0-9_.]+)$")

ARCH_DEFAULTS: dict[str, dict[str, object]] = {
    "12b": {
        "model_id": "google/gemma-4-12B-it",
        "layer_scope": "top12_no_qkv",
        "expected_layers": 48,
    },
    "26b-a4b": {
        "model_id": "google/gemma-4-26B-A4B-it",
        "layer_scope": "26b_a4b_top10_no_qkv",
        "expected_layers": 30,
    },
    "e4b": {
        "model_id": "google/gemma-4-E4B-it",
        "layer_scope": "e4b_upper20_no_qkv",
        "expected_layers": 42,
    },
}

SCOPE_SPECS: dict[str, tuple[int, int, frozenset[str], int]] = {
    "top12_no_qkv": (36, 47, NON_QKV_NAMES, 48),
    "top8_no_qkv": (40, 47, NON_QKV_NAMES, 32),
    "26b_a4b_top10_no_qkv": (20, 29, NON_QKV_NAMES, 40),
    "e4b_upper20_no_qkv": (22, 41, NON_QKV_NAMES, 80),
    "e4b_all42_no_qkv": (0, 41, NON_QKV_NAMES, 168),
}


@dataclass
class Layout:
    positions: torch.Tensor
    documents: torch.Tensor
    cumulative: torch.Tensor
    maximum: int
    global_mask: torch.Tensor


def layout(lengths: list[int], device: torch.device, max_pack_tokens: int = 8192) -> Layout:
    if not lengths or min(lengths) <= 0 or sum(lengths) > max_pack_tokens:
        raise ValueError(f"pack must contain positive documents within {max_pack_tokens} tokens")
    positions = torch.cat([torch.arange(n, device=device) for n in lengths])
    docs = torch.repeat_interleave(
        torch.arange(len(lengths), device=device),
        torch.tensor(lengths, device=device),
    )
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    mask = ((docs[:, None] == docs[None, :]) & (positions[:, None] >= positions[None, :]))[None, None]
    return Layout(positions[None], docs, cu, max(lengths), mask)


def packed_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout: float = 0.0,
    scaling: float = 1.0,
    sliding_window=None,
    cu_seq_lens_q=None,
    cu_seq_lens_k=None,
    max_length_q=None,
    max_length_k=None,
    **kwargs,
):
    if scaling != 1.0 or dropout != 0.0:
        raise ValueError("Gemma 4 native attention requires scale=1 and zero attention dropout")
    if module.is_sliding:
        sw = int(sliding_window or getattr(module, "sliding_window", None) or 512)
        if cu_seq_lens_q is not None:
            cu_list = cu_seq_lens_q.tolist()
            lengths = [b - a for a, b in zip(cu_list[:-1], cu_list[1:])]
            plan = layout(lengths, query.device, max_pack_tokens=max(8192, sum(lengths)))
            win_mask = plan.global_mask & ((plan.positions[:, :, None] - plan.positions[:, None, :]) < sw)[:, None]
            out = F.scaled_dot_product_attention(
                query, key, value, attn_mask=win_mask, dropout_p=0.0, is_causal=False, scale=1.0, enable_gqa=True
            )
            return out.transpose(1, 2), None
        seq_len = query.shape[2]
        pos = torch.arange(seq_len, device=query.device)
        diff = pos[:, None] - pos[None, :]
        win_mask = ((diff >= 0) & (diff < sw))[None, None]
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=win_mask, dropout_p=0.0, is_causal=False, scale=1.0, enable_gqa=True
        )
        return out.transpose(1, 2), None

    if attention_mask is None:
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True, scale=1.0, enable_gqa=True
        )
        return out.transpose(1, 2), None

    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=1.0, enable_gqa=True
    )
    return out.transpose(1, 2), None


def hidden_packed(model, ids: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    plan = layout(lengths, ids.device, max_pack_tokens=max(8192, int(ids.shape[1])))
    return model.model.language_model(
        input_ids=ids,
        position_ids=plan.positions,
        attention_mask={"sliding_attention": None, "full_attention": plan.global_mask},
        use_cache=False,
        cu_seq_lens_q=plan.cumulative,
        cu_seq_lens_k=plan.cumulative,
        max_length_q=plan.maximum,
        max_length_k=plan.maximum,
    ).last_hidden_state


def hidden_single(model, ids_1d_or_2d: torch.Tensor) -> torch.Tensor:
    ids = ids_1d_or_2d.unsqueeze(0) if ids_1d_or_2d.dim() == 1 else ids_1d_or_2d
    seq_len = int(ids.shape[1])
    return hidden_packed(model, ids, [seq_len])


def token_nll(h: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, chunk_size: int = 48) -> torch.Tensor:
    """Exact Gemma 4 FP32 softcap (`30 * tanh(logits / 30)`) NLL on scored positions only."""

    def chunk(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = 30.0 * torch.tanh(F.linear(x, weight).float() / 30.0)
        return F.cross_entropy(logits, y, reduction="none")

    if h.requires_grad or weight.requires_grad:
        return torch.cat(
            [
                checkpoint(chunk, h[i : i + chunk_size], labels[i : i + chunk_size], use_reentrant=False)
                for i in range(0, len(labels), chunk_size)
            ]
        )
    return torch.cat([chunk(h[i : i + chunk_size], labels[i : i + chunk_size]) for i in range(0, len(labels), chunk_size)])


_NN_MODULE_BASE = nn.Module if HAS_TORCH else object


class LoRALinear(_NN_MODULE_BASE):  # type: ignore[misc]
    """Linear wrapper with active trainable branch, frozen additive branches, and per-token gate mask."""

    def __init__(
        self,
        base: Any,
        rank: int = 16,
        alpha: float = 32.0,
        trainable: bool = True,
        active_scale: float = 1.0,
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to instantiate LoRALinear")
        super().__init__()
        if isinstance(base, LoRALinear):
            raise TypeError("Wrap underlying Linear directly or add branches to existing LoRALinear")
        self.base = base
        self.base.requires_grad_(False)
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank) if rank > 0 else 0.0
        self.active_scale = float(active_scale)

        dev = base.weight.device
        if rank > 0:
            self.A = nn.Parameter(
                torch.empty(rank, self.in_features, device=dev, dtype=torch.float32),
                requires_grad=trainable,
            )
            self.B = nn.Parameter(
                torch.zeros(self.out_features, rank, device=dev, dtype=torch.float32),
                requires_grad=trainable,
            )
            if not self.A.is_meta:
                nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

        self._frozen_scales: dict[str, float] = {}
        self._token_gate_mask: torch.Tensor | None = None

    @staticmethod
    def _branch_delta(x: torch.Tensor, a_mat: torch.Tensor, b_mat: torch.Tensor, eff_scale: float) -> torch.Tensor:
        orig_dtype = x.dtype
        if torch.is_autocast_enabled() or a_mat.dtype == orig_dtype:
            return (F.linear(F.linear(x, a_mat), b_mat) * eff_scale).to(orig_dtype)
        return (F.linear(F.linear(x.float(), a_mat.float()), b_mat.float()) * eff_scale).to(orig_dtype)

    def _apply_token_gate(self, delta: torch.Tensor) -> torch.Tensor:
        mask = self._token_gate_mask
        if mask is None:
            return delta
        if delta.dim() == 3 and mask.shape[1] == delta.shape[1]:
            return delta * mask.to(dtype=delta.dtype, device=delta.device)
        if delta.dim() == 2 and mask.shape[1] == delta.shape[0]:
            return delta * mask.squeeze(0).to(dtype=delta.dtype, device=delta.device)
        raise ValueError(f"Token gate mask shape {tuple(mask.shape)} incompatible with delta {tuple(delta.shape)}")

    def add_frozen_branch(
        self,
        branch_name: str,
        a_weight: torch.Tensor,
        b_weight: torch.Tensor,
        scaling: float,
        multiplier: float = 1.0,
    ) -> None:
        eff = float(scaling) * float(multiplier)
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", branch_name)
        dev = self.base.weight.device
        self.register_buffer(f"_frz_A_{safe}", a_weight.detach().to(device=dev, dtype=torch.float32), persistent=False)
        self.register_buffer(f"_frz_B_{safe}", b_weight.detach().to(device=dev, dtype=torch.float32), persistent=False)
        self._frozen_scales[safe] = eff

    def set_frozen_multiplier(self, branch_name: str, scaling: float, multiplier: float) -> None:
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", branch_name)
        if safe in self._frozen_scales:
            self._frozen_scales[safe] = float(scaling) * float(multiplier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        total_delta: torch.Tensor | None = None

        for safe, eff in self._frozen_scales.items():
            if eff != 0.0:
                a_buf = getattr(self, f"_frz_A_{safe}")
                b_buf = getattr(self, f"_frz_B_{safe}")
                d = self._branch_delta(x, a_buf, b_buf, eff)
                total_delta = d if total_delta is None else (total_delta + d)

        if self.A is not None and self.B is not None:
            eff_active = self.scaling * self.active_scale
            if eff_active != 0.0:
                d_act = self._branch_delta(x, self.A, self.B, eff_active)
                total_delta = d_act if total_delta is None else (total_delta + d_act)

        if total_delta is not None:
            out = out + self._apply_token_gate(total_delta)
        return out


def set_model_token_gate_mask(
    model: nn.Module,
    mask: torch.Tensor | None,
    lora_modules: list[LoRALinear] | None = None,
) -> None:
    mods = lora_modules if lora_modules is not None else [m for m in model.modules() if isinstance(m, LoRALinear)]
    for mod in mods:
        mod._token_gate_mask = mask


def strip_unused_multimodal_towers(model: nn.Module) -> list[str]:
    dropped: list[str] = []
    inner = getattr(model, "model", None)
    if inner is None:
        return dropped
    for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector"):
        if getattr(inner, attr, None) is not None:
            setattr(inner, attr, None)
            dropped.append(attr)
    return dropped


def resolve_target_modules(model: nn.Module, layer_scope: str) -> list[tuple[str, nn.Linear]]:
    if layer_scope not in SCOPE_SPECS:
        raise ValueError(f"Unsupported layer_scope: {layer_scope!r}; valid={sorted(SCOPE_SPECS)}")
    min_layer, max_layer, allowed_leaves, _ = SCOPE_SPECS[layer_scope]

    targets: list[tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        m = LAYER_RE.match(name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        leaf = name.rsplit(".", 1)[-1]
        if not (min_layer <= layer_idx <= max_layer) or leaf not in allowed_leaves:
            continue
        if isinstance(mod, LoRALinear):
            targets.append((name, mod.base))  # type: ignore[arg-type]
        elif isinstance(mod, nn.Linear):
            targets.append((name, mod))
    return targets


def attach_lora(
    model: nn.Module,
    layer_scope: str,
    rank: int = 16,
    alpha: float = 32.0,
    trainable: bool = True,
    active_scale: float = 1.0,
    enable_checkpointing: bool = True,
    strict_count: bool = True,
) -> tuple[list[str], list[LoRALinear]]:
    model.requires_grad_(False)
    targets = resolve_target_modules(model, layer_scope=layer_scope)
    _, _, _, expected = SCOPE_SPECS[layer_scope]
    if strict_count and len(targets) != expected:
        raise ValueError(f"Expected {expected} projections for layer_scope={layer_scope!r}, found {len(targets)}")

    names: list[str] = []
    lora_mods: list[LoRALinear] = []
    for name, base_lin in targets:
        if any(q in name for q in ("q_proj", "k_proj", "v_proj")):
            raise AssertionError(f"Forbidden QKV module leaked into non-QKV scope: {name}")
        cur = model.get_submodule(name)
        parent_name, leaf = name.rsplit(".", 1)
        new_mod = LoRALinear(base_lin, rank=rank, alpha=alpha, trainable=trainable, active_scale=active_scale)
        if isinstance(cur, LoRALinear):
            for safe, eff in cur._frozen_scales.items():
                new_mod.register_buffer(f"_frz_A_{safe}", getattr(cur, f"_frz_A_{safe}"), persistent=False)
                new_mod.register_buffer(f"_frz_B_{safe}", getattr(cur, f"_frz_B_{safe}"), persistent=False)
                new_mod._frozen_scales[safe] = eff
        setattr(model.get_submodule(parent_name), leaf, new_mod)
        names.append(name)
        lora_mods.append(new_mod)

    if trainable and enable_checkpointing and hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lm.enable_input_require_grads()
    return names, lora_mods


def build_dual_param_groups(model: nn.Module, mlp_lr: float, attn_lr: float) -> list[dict]:
    mlp_params: list[nn.Parameter] = []
    attn_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(q in name for q in (".q_proj.", ".k_proj.", ".v_proj.")):
            raise ValueError(f"QKV parameter unexpectedly marked trainable: {name}")
        if ".mlp." in name:
            mlp_params.append(param)
        elif ".self_attn.o_proj." in name:
            attn_params.append(param)
        else:
            mlp_params.append(param)
    groups: list[dict] = []
    if mlp_params:
        groups.append({"name": "mlp", "params": mlp_params, "lr": mlp_lr, "peak_lr": mlp_lr, "weight_decay": 0.0})
    if attn_params:
        groups.append({"name": "self_attn_o_proj", "params": attn_params, "lr": attn_lr, "peak_lr": attn_lr, "weight_decay": 0.0})
    if not groups:
        raise ValueError("No trainable LoRA parameters found")
    return groups


def switch_dpo_policy_vs_ref(lora_modules: list[LoRALinear], mode: str, ref_scaling: float) -> None:
    if mode == "policy":
        for mod in lora_modules:
            mod.active_scale = 1.0
            mod.set_frozen_multiplier("dpo_ref", ref_scaling, 0.0)
    elif mode == "ref":
        for mod in lora_modules:
            mod.active_scale = 0.0
            mod.set_frozen_multiplier("dpo_ref", ref_scaling, 1.0)
    else:
        raise ValueError(f"Unknown DPO mode: {mode!r}")


def load_pretrained_into_trainable_lora(model: nn.Module, src_adapter_dir: Path) -> dict:
    from safetensors.torch import load_file

    meta = json.loads((src_adapter_dir / "adapter.json").read_text(encoding="utf8"))
    raw = load_file(str(src_adapter_dir / "adapter.safetensors"), device="cpu")
    src_rank = int(meta["rank"])
    src_alpha = float(meta["alpha"])
    src_scaling = src_alpha / src_rank

    for mod_name in meta["targets"]:
        sub = model.get_submodule(mod_name)
        if not isinstance(sub, LoRALinear):
            raise TypeError(f"Expected LoRALinear at {mod_name}, got {type(sub)}")
        a_w = raw.pop(f"{mod_name}.A")
        b_w = raw.pop(f"{mod_name}.B")
        with torch.no_grad():
            sub.A.copy_(a_w.to(device=sub.A.device, dtype=sub.A.dtype))
            sub.B.copy_(b_w.to(device=sub.B.device, dtype=sub.B.dtype))
        sub.add_frozen_branch("dpo_ref", a_w, b_w, scaling=src_scaling, multiplier=0.0)

    if raw:
        raise ValueError(f"Unused tensors in {src_adapter_dir}: {sorted(raw)[:5]}")
    return meta


def save_custom_and_peft_adapter(
    model: nn.Module,
    out_dir: Path,
    base_model_id: str,
    revision: str | None,
    layer_scope: str,
    targets: list[str],
    rank: int,
    alpha: float,
    extra_meta: dict | None = None,
) -> dict:
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    values = {n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}
    if len(values) != len(targets) * 2:
        raise AssertionError(f"Expected {len(targets)*2} trainable tensors, got {len(values)}")

    save_file(values, str(out_dir / "adapter.safetensors"))

    peft_tensors: dict[str, torch.Tensor] = {}
    layers_seen: set[int] = set()
    for mod_name in targets:
        m = LAYER_RE.match(mod_name)
        if m:
            layers_seen.add(int(m.group(1)))
        peft_tensors[f"base_model.model.{mod_name}.lora_A.weight"] = values[f"{mod_name}.A"]
        peft_tensors[f"base_model.model.{mod_name}.lora_B.weight"] = values[f"{mod_name}.B"]

    peft_dir = out_dir / "peft"
    peft_dir.mkdir(parents=True, exist_ok=True)
    save_file(peft_tensors, str(peft_dir / "adapter_model.safetensors"))
    peft_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model_id,
        "r": int(rank),
        "lora_alpha": float(alpha),
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "target_modules": ["o_proj", "gate_proj", "up_proj", "down_proj"],
        "layers_to_transform": sorted(layers_seen),
    }
    (peft_dir / "adapter_config.json").write_text(
        json.dumps(peft_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )

    meta = {
        "base": base_model_id,
        "revision": revision,
        "layer_scope": layer_scope,
        "rank": int(rank),
        "alpha": float(alpha),
        "scaling": round(float(alpha) / float(rank), 6),
        "linear_modules": len(targets),
        "qkv_modules": 0,
        "trainable_parameters": int(sum(t.numel() for t in values.values())),
        "token_gate_matched_pc2": True,
        "targets": targets,
    }
    if extra_meta:
        meta.update(extra_meta)
    (out_dir / "adapter.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return meta


def load_gemma4_model_for_training(
    base_path_or_id: str,
    revision: str | None = None,
    expected_layers: int | None = None,
):
    from transformers import AutoModelForImageTextToText
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register("g4_lab_packed_sdpa", packed_attention)
    model = AutoModelForImageTextToText.from_pretrained(
        base_path_or_id,
        revision=revision,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": "cuda"},
        low_cpu_mem_usage=True,
    )
    strip_unused_multimodal_towers(model)
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    cfg = model.config.text_config
    if expected_layers is not None and int(cfg.num_hidden_layers) != int(expected_layers):
        raise ValueError(f"Expected {expected_layers} text layers, got {cfg.num_hidden_layers}")
    if float(cfg.final_logit_softcapping or 0.0) != 30.0:
        raise ValueError(f"Expected Gemma 4 final_logit_softcapping=30.0, got {cfg.final_logit_softcapping}")

    model.config.text_config._attn_implementation = "g4_lab_packed_sdpa"
    model.config.use_cache = False
    return model


def sequence_masked_logprob_sum(
    model: nn.Module,
    lora_modules: list[LoRALinear],
    ids_list: list[int],
    labels_list: list[int],
    use_token_gate: bool = True,
) -> tuple[torch.Tensor, int]:
    """Compute exact sum of log-probabilities over supervised tokens (`labels != -100`)."""
    device = next(model.parameters()).device
    ids = torch.tensor([ids_list], dtype=torch.long, device=device)
    labels = torch.tensor([labels_list], dtype=torch.long, device=device)

    shifted = labels[0, 1:]
    valid = shifted != -100
    n_sup = int(valid.sum().item())
    if n_sup <= 0:
        raise ValueError("Sequence has 0 supervised tokens for DPO logprob")

    gate_mask = None
    if use_token_gate:
        gate_1d = torch.zeros(ids.shape[1], dtype=torch.float32, device=device)
        gate_1d[:-1] = valid.float()
        gate_mask = gate_1d.view(1, -1, 1)

    set_model_token_gate_mask(model, gate_mask, lora_modules=lora_modules)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = hidden_single(model, ids)
            nll = token_nll(h[0, :-1][valid], model.get_output_embeddings().weight, shifted[valid])
            logp_sum = -nll.float().sum()
    finally:
        set_model_token_gate_mask(model, None, lora_modules=lora_modules)

    return logp_sum, n_sup
