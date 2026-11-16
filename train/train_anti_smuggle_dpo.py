#!/usr/bin/env python3
"""On-Policy Phase-B Anti-Smuggle DPO Trainer.

Suppresses prose smuggling (wrapping customer-service numbered lists inside ```` ```markdown ````
code blocks) and defensive boilerplate using single-model LoRA `policy` vs frozen `dpo_ref`
branch switching over exact `<|"|>` (`52`) token-gated bubble sequences.
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

from kernels_g4 import ARCH_DEFAULTS, SCOPE_SPECS  # noqa: E402
from train_thin_voice_sft import cosine_ratio  # noqa: E402


def load_dpo_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels"):
                if not isinstance(row.get(key), list) or not row[key]:
                    raise ValueError(f"Missing {key} in DPO pair {row.get('id')}")
            if len(row["chosen_input_ids"]) != len(row["chosen_labels"]):
                raise ValueError(f"chosen length mismatch in {row.get('id')}")
            if len(row["rejected_input_ids"]) != len(row["rejected_labels"]):
                raise ValueError(f"rejected length mismatch in {row.get('id')}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No DPO pairs found in {path}")
    return rows


def compute_dpo_loss(
    policy_chosen_logp,
    policy_rejected_logp,
    ref_chosen_logp,
    ref_rejected_logp,
    *,
    beta: float = 0.08,
    sft_nll=None,
    sft_alpha: float = 0.05,
):
    import torch.nn.functional as F

    pi_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    logits = beta * (pi_logratios - ref_logratios)
    dpo_loss = -F.logsigmoid(logits)
    if sft_nll is not None and sft_alpha > 0.0:
        total = dpo_loss + float(sft_alpha) * sft_nll
    else:
        total = dpo_loss
    return total, float(logits.detach().item())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=["12b", "26b-a4b", "e4b"], default="12b")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--init-adapter", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--beta", type=float, default=0.08)
    ap.add_argument("--sft-alpha", type=float, default=0.05)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--mlp-lr", type=float, default=1.0e-5)
    ap.add_argument("--attn-lr", type=float, default=4.0e-6)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    pairs = load_dpo_pairs(args.pairs)
    init_meta = json.loads((args.init_adapter / "adapter.json").read_text(encoding="utf8"))

    if args.dry_run:
        arch_cfg = ARCH_DEFAULTS[args.arch]
        scope = str(init_meta.get("layer_scope") or arch_cfg["layer_scope"])
        print(
            json.dumps(
                {
                    "trainer": "anti-smuggle-dpo",
                    "arch": args.arch,
                    "layer_scope": scope,
                    "linear_modules": SCOPE_SPECS.get(scope, (0, 0, frozenset(), len(init_meta.get("targets", []))))[3],
                    "pairs_loaded": len(pairs),
                    "beta": args.beta,
                    "sft_alpha": args.sft_alpha,
                    "grad_accum": args.grad_accum,
                    "planned_steps": args.max_steps,
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
        load_pretrained_into_trainable_lora,
        save_custom_and_peft_adapter,
        sequence_masked_logprob_sum,
        switch_dpo_policy_vs_ref,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for DPO training (use --dry-run on CPU)")

    arch_cfg = ARCH_DEFAULTS[args.arch]
    model_id = args.model_id or str(arch_cfg["model_id"])
    layer_scope = str(init_meta.get("layer_scope") or arch_cfg["layer_scope"])
    rank = int(init_meta["rank"])
    alpha = float(init_meta["alpha"])
    ref_scaling = alpha / float(rank)

    model = load_gemma4_model_for_training(model_id, expected_layers=int(arch_cfg["expected_layers"]))  # type: ignore[arg-type]
    targets, lora_modules = attach_lora(model, layer_scope=layer_scope, rank=rank, alpha=alpha, trainable=True)
    load_pretrained_into_trainable_lora(model, args.init_adapter)

    param_groups = build_dual_param_groups(model, mlp_lr=args.mlp_lr, attn_lr=args.attn_lr)
    all_trainable = [p for g in param_groups for p in g["params"]]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0, foreach=False)

    rng = random.Random(args.seed)
    pool: list[dict] = []
    while len(pool) < args.max_steps * args.grad_accum:
        batch = list(pairs)
        rng.shuffle(batch)
        pool.extend(batch)

    warmup = max(1, math.ceil(0.08 * args.max_steps))
    model.train()
    for step in range(1, args.max_steps + 1):
        ratio = cosine_ratio(step, args.max_steps, warmup)
        for g in optimizer.param_groups:
            g["lr"] = g["peak_lr"] * ratio
        optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        step_margin = 0.0
        for micro in range(args.grad_accum):
            pair = pool[(step - 1) * args.grad_accum + micro]
            with torch.no_grad():
                switch_dpo_policy_vs_ref(lora_modules, "ref", ref_scaling)
                ref_c, n_c = sequence_masked_logprob_sum(model, lora_modules, pair["chosen_input_ids"], pair["chosen_labels"])
                ref_r, n_r = sequence_masked_logprob_sum(model, lora_modules, pair["rejected_input_ids"], pair["rejected_labels"])

            switch_dpo_policy_vs_ref(lora_modules, "policy", ref_scaling)
            pi_c, _ = sequence_masked_logprob_sum(model, lora_modules, pair["chosen_input_ids"], pair["chosen_labels"])
            pi_r, _ = sequence_masked_logprob_sum(model, lora_modules, pair["rejected_input_ids"], pair["rejected_labels"])
            sft_nll = -pi_c / float(max(1, n_c))
            loss, margin = compute_dpo_loss(
                pi_c / float(max(1, n_c)),
                pi_r / float(max(1, n_r)),
                ref_c / float(max(1, n_c)),
                ref_r / float(max(1, n_r)),
                beta=args.beta,
                sft_nll=sft_nll,
                sft_alpha=args.sft_alpha,
            )
            (loss / float(args.grad_accum)).backward()
            step_loss += float(loss.detach().item()) / float(args.grad_accum)
            step_margin += margin / float(args.grad_accum)

        torch.nn.utils.clip_grad_norm_(all_trainable, args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == args.max_steps:
            print(json.dumps({"step": step, "dpo_loss": round(step_loss, 6), "margin": round(step_margin, 6)}), flush=True)

    save_custom_and_peft_adapter(
        model,
        args.output,
        base_model_id=model_id,
        revision=None,
        layer_scope=layer_scope,
        targets=targets,
        rank=rank,
        alpha=alpha,
        extra_meta={"mode": "dpo_anti_smuggle", "beta": args.beta, "sft_alpha": args.sft_alpha},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
