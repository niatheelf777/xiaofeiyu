#!/usr/bin/env python3
"""Parameterized Thin Upper-Layer Non-QKV BF16 Voice Core SFT Trainer (`12b`, `26b-a4b`, `e4b`).

Guarantees:
1. Freezes lower layers and freezes ALL Q/K/V projections (`q_proj`, `k_proj`, `v_proj`).
2. Trains ONLY upper-layer `{o_proj, gate_proj, up_proj, down_proj}` (`48` modules for `12b` L36..47,
   `80` modules for `e4b` L22..41, `40` modules for `26b-a4b` L20..29).
3. Enforces exact `<|"|>` (`52`) bubble Loss Mask AND forward-pass `token_gate_mask[t] = (labels[t+1] != -100)`
   so SFT training is 100% forward-matched to Split-Phase inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import chain
import json
import math
from pathlib import Path
import random
import signal
import sys

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from kernels_g4 import ARCH_DEFAULTS, SCOPE_SPECS  # noqa: E402

QUOTE_DELIM_ID = 52
STOP = False


def stop_handler(*_) -> None:
    global STOP
    STOP = True


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_and_verify_sft_data(root: Path, max_seq_len: int = 4096) -> tuple[list[dict], dict]:
    root = Path(root)
    manifest_path = root / "export-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8")) if manifest_path.is_file() else {}
    raw_rows = read_jsonl(root / "sft.jsonl")
    if not raw_rows:
        raise ValueError(f"Empty sft.jsonl in {root}")

    verified: list[dict] = []
    seen_keys: set[str] = set()
    for r in raw_rows:
        rk = str(r.get("key") or r["id"])
        if rk in seen_keys:
            raise ValueError(f"Duplicate SFT key: {rk}")
        seen_keys.add(rk)

        ids = r["input_ids"]
        labels = r["labels"]
        regions = r["regions"]
        if len(ids) != len(labels) or len(ids) > max_seq_len or labels[0] != -100 or labels[-1] != -100:
            raise ValueError(f"Invalid sequence bounds in {rk}")
        if not regions:
            raise ValueError(f"Missing supervised bubble regions in {rk}")

        for s_idx, e_idx in regions:
            if ids[s_idx - 1] != QUOTE_DELIM_ID or labels[s_idx - 1] != -100:
                raise ValueError(f"Opening <|\\\"|> (52) mask violation in {rk}")
            if ids[e_idx - 1] != QUOTE_DELIM_ID or labels[e_idx - 1] != QUOTE_DELIM_ID:
                raise ValueError(f"Closing <|\\\"|> (52) supervision violation in {rk}")
            if labels[e_idx] != -100:
                raise ValueError(f"Post-bubble token mask violation in {rk}")
        verified.append(r)

    return verified, manifest


def build_interleaved_packs(
    rows: list[dict],
    pack_tokens: int = 2560,
    epochs: int = 3,
    seed: int = 20261115,
) -> list[list[tuple[list[int], list[int]]]]:
    id_scenes = [r for r in rows if "identity" in str(r.get("source", ""))]
    general_rows = [r for r in rows if "identity" not in str(r.get("source", ""))]

    all_packs: list[list[tuple[list[int], list[int]]]] = []
    for ep in range(epochs):
        rng = random.Random(seed + ep * 97)
        gen_shuf = list(general_rows)
        rng.shuffle(gen_shuf)
        id_pool = list(id_scenes) * 2
        rng.shuffle(id_pool)

        merged: list[dict] = []
        if id_pool:
            stride = max(1, len(gen_shuf) // len(id_pool))
            id_idx = 0
            for g_idx, g_row in enumerate(gen_shuf):
                merged.append(g_row)
                if (g_idx + 1) % stride == 0 and id_idx < len(id_pool):
                    merged.append(id_pool[id_idx])
                    id_idx += 1
            while id_idx < len(id_pool):
                merged.append(id_pool[id_idx])
                id_idx += 1
        else:
            merged = gen_shuf

        cur_pack: list[tuple[list[int], list[int]]] = []
        cur_tokens = 0
        for row in merged:
            ids = row["input_ids"]
            lbl = row["labels"]
            w = len(ids)
            if cur_tokens + w > pack_tokens and cur_pack:
                all_packs.append(cur_pack)
                cur_pack = []
                cur_tokens = 0
            cur_pack.append((ids, lbl))
            cur_tokens += w
        if cur_pack:
            all_packs.append(cur_pack)
    return all_packs


def cosine_ratio(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 0.15) -> float:
    if step <= warmup_steps:
        return step / max(1, warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def forward_and_backward_pack(model, lora_modules, pack, grad_accum: int) -> tuple[float, int]:
    import torch
    from kernels_g4 import hidden_packed, set_model_token_gate_mask, token_nll

    device = next(model.parameters()).device
    lengths = [len(ids) for ids, _ in pack]
    ids = torch.tensor([list(chain.from_iterable(i for i, _ in pack))], dtype=torch.long, device=device)
    labels = torch.tensor([list(chain.from_iterable(l for _, l in pack))], dtype=torch.long, device=device)

    shifted = labels[0, 1:]
    valid = shifted != -100
    n_sup = int(valid.sum().item())
    if n_sup <= 0:
        raise ValueError("Pack contains 0 supervised bubble tokens")

    gate_1d = torch.zeros(ids.shape[1], dtype=torch.float32, device=device)
    gate_1d[:-1] = valid.float()
    gate_mask = gate_1d.view(1, -1, 1)

    set_model_token_gate_mask(model, gate_mask, lora_modules=lora_modules)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = hidden_packed(model, ids, lengths)
            nll = token_nll(h[0, :-1][valid], model.get_output_embeddings().weight, shifted[valid])
            ce = nll.float().mean()
            (ce / float(grad_accum)).backward()
    finally:
        set_model_token_gate_mask(model, None, lora_modules=lora_modules)

    return float(ce.detach().item()), n_sup


def build_dry_run_plan(args: argparse.Namespace, rows: list[dict], micro_packs: list) -> dict:
    arch_cfg = ARCH_DEFAULTS[args.arch]
    layer_scope = args.layer_scope or str(arch_cfg["layer_scope"])
    expected_modules = SCOPE_SPECS[layer_scope][3]
    ckpt_steps = sorted({int(x.strip()) for x in args.checkpoint_steps.split(",") if x.strip()})
    return {
        "trainer": "thin-voice-sft",
        "arch": args.arch,
        "model_id": args.model_id or str(arch_cfg["model_id"]),
        "layer_scope": layer_scope,
        "linear_modules": expected_modules,
        "qkv_modules": 0,
        "rank": args.rank,
        "alpha": args.alpha,
        "scaling": args.alpha / args.rank,
        "sft_records": len(rows),
        "available_micro_packs": len(micro_packs),
        "pack_tokens": args.pack_tokens,
        "grad_accum": args.grad_accum,
        "planned_steps": args.max_steps,
        "checkpoint_steps": ckpt_steps,
        "mlp_lr": args.mlp_lr,
        "attn_lr": args.attn_lr,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=["12b", "26b-a4b", "e4b"], default="12b")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--layer-scope", choices=sorted(SCOPE_SPECS), default=None)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--pack-tokens", type=int, default=2560)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--checkpoint-steps", default="300,600")
    ap.add_argument("--mlp-lr", type=float, default=3.0e-5)
    ap.add_argument("--attn-lr", type=float, default=1.2e-5)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    rows, _ = load_and_verify_sft_data(args.data)
    micro_packs = build_interleaved_packs(rows, pack_tokens=args.pack_tokens, epochs=3, seed=args.seed)

    if args.dry_run:
        print(json.dumps(build_dry_run_plan(args, rows, micro_packs), ensure_ascii=False, indent=2))
        return 0

    needed_micro = args.max_steps * args.grad_accum
    while len(micro_packs) < needed_micro:
        micro_packs.extend(build_interleaved_packs(rows, pack_tokens=args.pack_tokens, epochs=1, seed=args.seed + len(micro_packs)))
    micro_packs = micro_packs[:needed_micro]

    import torch
    from kernels_g4 import (
        attach_lora,
        build_dual_param_groups,
        load_gemma4_model_for_training,
        save_custom_and_peft_adapter,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for training (use --dry-run on CPU)")

    signal.signal(signal.SIGTERM, stop_handler)
    arch_cfg = ARCH_DEFAULTS[args.arch]
    model_id = args.model_id or str(arch_cfg["model_id"])
    layer_scope = args.layer_scope or str(arch_cfg["layer_scope"])
    expected_layers = int(arch_cfg["expected_layers"])  # type: ignore[arg-type]

    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = load_gemma4_model_for_training(model_id, expected_layers=expected_layers)
    targets, lora_modules = attach_lora(model, layer_scope=layer_scope, rank=args.rank, alpha=args.alpha, trainable=True)
    param_groups = build_dual_param_groups(model, mlp_lr=args.mlp_lr, attn_lr=args.attn_lr)
    all_trainable = [p for g in param_groups for p in g["params"]]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0, foreach=False)

    ckpt_steps = sorted({int(x.strip()) for x in args.checkpoint_steps.split(",") if x.strip()})
    warmup = max(1, math.ceil(0.06 * args.max_steps))
    t0 = datetime.now(timezone.utc)
    losses: list[float] = []

    model.train()
    for step in range(1, args.max_steps + 1):
        if STOP:
            break
        ratio = cosine_ratio(step, args.max_steps, warmup)
        for g in optimizer.param_groups:
            g["lr"] = g["peak_lr"] * ratio
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for micro_idx in range(args.grad_accum):
            ce_val, _ = forward_and_backward_pack(model, lora_modules, micro_packs[(step - 1) * args.grad_accum + micro_idx], args.grad_accum)
            step_loss += ce_val / float(args.grad_accum)
        torch.nn.utils.clip_grad_norm_(all_trainable, args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        losses.append(step_loss)
        if step == 1 or step % 25 == 0 or step in ckpt_steps:
            print(json.dumps({"step": step, "steps": args.max_steps, "bubble_ce": round(step_loss, 6)}), flush=True)
        if step in ckpt_steps:
            save_custom_and_peft_adapter(
                model,
                args.output / "checkpoints" / f"ckpt-{step}",
                base_model_id=model_id,
                revision=None,
                layer_scope=layer_scope,
                targets=targets,
                rank=args.rank,
                alpha=args.alpha,
                extra_meta={"mode": "backbone", "step": step, "elapsed_s": round((datetime.now(timezone.utc) - t0).total_seconds(), 2)},
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
