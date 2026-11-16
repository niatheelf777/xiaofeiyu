#!/usr/bin/env python3
"""Residual 4-Style Personality Cartridge Trainer (`c1..c4`) over Frozen `s0` Backbone.

Trains strictly the 4 open-source personality cartridges (`neko_clingy`, `older_tease`,
`brat_mesugaki`, `fragile_porcelain`) on upper-layer non-QKV modules with `s0` mounted
as a frozen branch (`multiplier=1.0`), ensuring clean linear additivity:
  W = W_base + s0 + c1*w1 + c2*w2 + c3*w3 + c4*w4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from kernels_g4 import ARCH_DEFAULTS, CARTRIDGE_IDS, CARTRIDGE_LABELS, SCOPE_SPECS  # noqa: E402
from train_thin_voice_sft import cosine_ratio, forward_and_backward_pack, load_and_verify_sft_data  # noqa: E402


def build_cartridge_packs(
    rows: list[dict],
    cid: str,
    pack_tokens: int = 2560,
    epochs: int = 3,
    seed: int = 20261115,
) -> list[list[tuple[list[int], list[int]]]]:
    anchors = [r for r in rows if "anchor" in str(r.get("source", ""))]
    gen_rows = [r for r in rows if "anchor" not in str(r.get("source", ""))]

    packs: list[list[tuple[list[int], list[int]]]] = []
    for ep in range(epochs):
        rng = random.Random(seed + hash(cid) % 10007 + ep * 131)
        g_shuf = list(gen_rows)
        rng.shuffle(g_shuf)
        a_pool = list(anchors) * 3
        rng.shuffle(a_pool)

        merged: list[dict] = []
        if a_pool:
            stride = max(1, len(g_shuf) // len(a_pool))
            a_idx = 0
            for idx, r in enumerate(g_shuf):
                merged.append(r)
                if (idx + 1) % stride == 0 and a_idx < len(a_pool):
                    merged.append(a_pool[a_idx])
                    a_idx += 1
            while a_idx < len(a_pool):
                merged.append(a_pool[a_idx])
                a_idx += 1
        else:
            merged = g_shuf

        cur: list[tuple[list[int], list[int]]] = []
        cur_tok = 0
        for row in merged:
            ids = row["input_ids"]
            lbl = row["labels"]
            if cur_tok + len(ids) > pack_tokens and cur:
                packs.append(cur)
                cur = []
                cur_tok = 0
            cur.append((ids, lbl))
            cur_tok += len(ids)
        if cur:
            packs.append(cur)
    return packs


def mount_frozen_backbone_on_lora(model, backbone_dir: Path) -> dict:
    from safetensors.torch import load_file
    from kernels_g4 import LoRALinear

    meta = json.loads((backbone_dir / "adapter.json").read_text(encoding="utf8"))
    weights = load_file(str(backbone_dir / "adapter.safetensors"))
    scaling = float(meta["alpha"]) / float(meta["rank"])

    for mod_name in meta["targets"]:
        sub = model.get_submodule(mod_name)
        if not isinstance(sub, LoRALinear):
            raise TypeError(f"Expected LoRALinear at {mod_name}, got {type(sub)}")
        sub.add_frozen_branch(
            "xiaofeiyu_backbone",
            weights.pop(f"{mod_name}.A"),
            weights.pop(f"{mod_name}.B"),
            scaling=scaling,
            multiplier=1.0,
        )
    if weights:
        raise AssertionError(f"Unused backbone tensors: {len(weights)}")
    return meta


def reset_trainable_lora_weights(lora_modules: list, seed: int) -> None:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    for mod in lora_modules:
        with torch.no_grad():
            nn.init.kaiming_uniform_(mod.A, a=math.sqrt(5))
            mod.B.zero_()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=["12b", "26b-a4b", "e4b"], default="12b")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--backbone-dir", type=Path, required=True)
    ap.add_argument("--out-cartridges-dir", type=Path, required=True)
    ap.add_argument("--steps-per-cartridge", type=int, default=190)
    ap.add_argument("--pack-tokens", type=int, default=2560)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--mlp-lr", type=float, default=5.0e-5)
    ap.add_argument("--attn-lr", type=float, default=2.0e-5)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    datasets: dict[str, tuple[list[dict], list]] = {}
    for cid in CARTRIDGE_IDS:
        rows, _ = load_and_verify_sft_data(args.data_root / cid)
        packs = build_cartridge_packs(rows, cid=cid, pack_tokens=args.pack_tokens, epochs=3)
        datasets[cid] = (rows, packs)

    if args.dry_run:
        arch_cfg = ARCH_DEFAULTS[args.arch]
        scope = str(arch_cfg["layer_scope"])
        print(
            json.dumps(
                {
                    "trainer": "residual-style-cartridges",
                    "arch": args.arch,
                    "layer_scope": scope,
                    "linear_modules": SCOPE_SPECS[scope][3],
                    "slots": list(CARTRIDGE_IDS),
                    "steps_per_cartridge": args.steps_per_cartridge,
                    "cartridges": {
                        cid: {
                            "label": CARTRIDGE_LABELS[cid],
                            "sft_records": len(r),
                            "available_packs": len(p),
                        }
                        for cid, (r, p) in datasets.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    import torch
    from kernels_g4 import (
        attach_lora,
        build_dual_param_groups,
        load_gemma4_model_for_training,
        save_custom_and_peft_adapter,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for cartridge training (use --dry-run on CPU)")

    arch_cfg = ARCH_DEFAULTS[args.arch]
    model_id = args.model_id or str(arch_cfg["model_id"])
    layer_scope = str(arch_cfg["layer_scope"])
    model = load_gemma4_model_for_training(model_id, expected_layers=int(arch_cfg["expected_layers"]))  # type: ignore[arg-type]
    targets, lora_modules = attach_lora(model, layer_scope=layer_scope, rank=args.rank, alpha=args.alpha, trainable=True)
    mount_frozen_backbone_on_lora(model, args.backbone_dir)

    for c_idx, cid in enumerate(CARTRIDGE_IDS):
        _, packs = datasets[cid]
        while len(packs) < args.steps_per_cartridge:
            packs.extend(packs[: args.steps_per_cartridge - len(packs)])
        packs = packs[: args.steps_per_cartridge]

        reset_trainable_lora_weights(lora_modules, seed=20261115 + c_idx * 101)
        param_groups = build_dual_param_groups(model, mlp_lr=args.mlp_lr, attn_lr=args.attn_lr)
        all_trainable = [p for g in param_groups for p in g["params"]]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0, foreach=False)
        warmup = max(1, math.ceil(0.08 * args.steps_per_cartridge))
        model.train()
        for step, pack in enumerate(packs, 1):
            ratio = cosine_ratio(step, args.steps_per_cartridge, warmup)
            for g in optimizer.param_groups:
                g["lr"] = g["peak_lr"] * ratio
            optimizer.zero_grad(set_to_none=True)
            ce_val, _ = forward_and_backward_pack(model, lora_modules, pack, grad_accum=1)
            torch.nn.utils.clip_grad_norm_(all_trainable, args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 40 == 0 or step == args.steps_per_cartridge:
                print(json.dumps({"cartridge_id": cid, "step": step, "bubble_ce": round(ce_val, 6)}), flush=True)

        save_custom_and_peft_adapter(
            model,
            args.out_cartridges_dir / cid,
            base_model_id=model_id,
            revision=None,
            layer_scope=layer_scope,
            targets=targets,
            rank=args.rank,
            alpha=args.alpha,
            extra_meta={"mode": "cartridge", "cartridge_id": cid, "cartridge_label": CARTRIDGE_LABELS[cid]},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
