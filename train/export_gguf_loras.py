#!/usr/bin/env python3
"""Export strictly 5-slot Xiaofeiyu LoRA adapters (`s0, c1..c4`) to thin PEFT & GGUF format.

Slices any custom `adapter.json + adapter.safetensors` to upper-layer non-QKV projections
(`o_proj`, `gate_proj`, `up_proj`, `down_proj`, zero Q/K/V) and invokes `llama.cpp/convert_lora_to_gguf.py`
for the 5 open-source slots:
  - Slot 0: `s0_xiaofeiyu_backbone.gguf`
  - Slot 1: `c1_neko_clingy.gguf`
  - Slot 2: `c2_older_tease.gguf`
  - Slot 3: `c3_brat_mesugaki.gguf`
  - Slot 4: `c4_fragile_porcelain.gguf`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

ORDERED_GGUF_SLOTS: tuple[tuple[int, str, str], ...] = (
    (0, "xiaofeiyu_backbone", "s0_xiaofeiyu_backbone.gguf"),
    (1, "neko_clingy", "c1_neko_clingy.gguf"),
    (2, "older_tease", "c2_older_tease.gguf"),
    (3, "brat_mesugaki", "c3_brat_mesugaki.gguf"),
    (4, "fragile_porcelain", "c4_fragile_porcelain.gguf"),
)

ARCH_CONFIGS = {
    "12b": ("google/gemma-4-12B-it", 36, 47),
    "26b-a4b": ("google/gemma-4-26B-A4B-it", 20, 29),
    "e4b": ("google/gemma-4-E4B-it", 22, 41),
}

LAYER_MOD_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.(?:self_attn\.o_proj|mlp\.(?:gate_proj|up_proj|down_proj))$"
)


def is_thin_voice_target(module_name: str, min_layer: int = 36, max_layer: int = 47) -> bool:
    m = LAYER_MOD_RE.match(module_name)
    if not m:
        return False
    return min_layer <= int(m.group(1)) <= max_layer


def convert_custom_adapter_to_thin_peft(
    src_adapter_dir: Path,
    out_peft_dir: Path,
    *,
    arch: str = "12b",
) -> dict:
    from safetensors.torch import load_file, save_file

    base_model_id, min_layer, max_layer = ARCH_CONFIGS[arch]
    src_adapter_dir = Path(src_adapter_dir)
    out_peft_dir = Path(out_peft_dir)
    meta_path = src_adapter_dir / "adapter.json"
    st_path = src_adapter_dir / "adapter.safetensors"
    if not st_path.is_file():
        raise FileNotFoundError(f"Missing adapter.safetensors in {src_adapter_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf8")) if meta_path.is_file() else {}
    raw_tensors = load_file(str(st_path), device="cpu")
    discovered = sorted({k[:-2] for k in raw_tensors if k.endswith(".A") and f"{k[:-2]}.B" in raw_tensors})

    peft_tensors: dict[str, object] = {}
    kept: list[str] = []
    skipped = 0
    inferred_rank: int | None = None

    for mod_name in discovered:
        if not is_thin_voice_target(mod_name, min_layer=min_layer, max_layer=max_layer):
            skipped += 1
            continue
        a_t = raw_tensors[f"{mod_name}.A"].contiguous()
        b_t = raw_tensors[f"{mod_name}.B"].contiguous()
        if inferred_rank is None:
            inferred_rank = int(a_t.shape[0])
        peft_tensors[f"base_model.model.{mod_name}.lora_A.weight"] = a_t
        peft_tensors[f"base_model.model.{mod_name}.lora_B.weight"] = b_t
        kept.append(mod_name)

    if not kept or inferred_rank is None:
        raise ValueError(f"Zero thin non-QKV modules matched in {src_adapter_dir}")

    rank = int(meta.get("rank") or inferred_rank)
    alpha = float(meta.get("alpha") if meta.get("alpha") is not None else rank)
    out_peft_dir.mkdir(parents=True, exist_ok=True)
    peft_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model_id,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "target_modules": ["o_proj", "gate_proj", "up_proj", "down_proj"],
        "layers_to_transform": list(range(min_layer, max_layer + 1)),
    }
    (out_peft_dir / "adapter_config.json").write_text(
        json.dumps(peft_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    save_file(peft_tensors, str(out_peft_dir / "adapter_model.safetensors"))  # type: ignore[arg-type]
    return {"peft_dir": str(out_peft_dir), "rank": rank, "lora_alpha": alpha, "kept_modules": len(kept), "skipped_modules": skipped}


def convert_peft_to_gguf(
    peft_dir: Path,
    out_gguf_path: Path,
    *,
    convert_script: Path,
    base_config_dir: Path,
    outtype: str = "f16",
) -> dict:
    out_gguf_path = Path(out_gguf_path)
    out_gguf_path.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        [
            sys.executable,
            str(convert_script),
            "--base",
            str(base_config_dir),
            "--outtype",
            outtype,
            "--outfile",
            str(out_gguf_path),
            str(peft_dir),
        ],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"convert_lora_to_gguf.py failed ({res.returncode}):\n{(res.stderr or res.stdout)[-1200:]}")
    return {"gguf_path": str(out_gguf_path), "size_bytes": out_gguf_path.stat().st_size}


def convert_all_five_slots_to_gguf(
    *,
    backbone_dir: Path,
    cartridges: dict[str, Path],
    out_dir: Path,
    convert_script: Path,
    base_config_dir: Path,
    arch: str = "12b",
    outtype: str = "f16",
) -> dict[str, dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slot_sources: dict[str, Path] = {"xiaofeiyu_backbone": Path(backbone_dir)}
    for _, branch, _ in ORDERED_GGUF_SLOTS[1:]:
        if branch not in cartridges or not Path(cartridges[branch]).is_dir():
            raise FileNotFoundError(f"Missing cartridge directory for {branch!r}")
        slot_sources[branch] = Path(cartridges[branch])

    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="xfy_peft_") as tmp_root:
        for slot_id, branch, gguf_filename in ORDERED_GGUF_SLOTS:
            peft_sub = Path(tmp_root) / branch
            peft_info = convert_custom_adapter_to_thin_peft(slot_sources[branch], peft_sub, arch=arch)
            gguf_info = convert_peft_to_gguf(
                peft_sub,
                out_dir / gguf_filename,
                convert_script=convert_script,
                base_config_dir=base_config_dir,
                outtype=outtype,
            )
            results[branch] = {"lora_id": slot_id, **peft_info, **gguf_info}
    (out_dir / "lora_gguf_manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=sorted(ARCH_CONFIGS), default="12b")
    ap.add_argument("--backbone-dir", type=Path, required=True)
    ap.add_argument("--cartridges-dir", type=Path, required=True)
    ap.add_argument("--convert-script", type=Path, default=Path("convert_lora_to_gguf.py"))
    ap.add_argument("--base-config-dir", type=Path, default=Path("./base_config"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--outtype", default="f16", choices=["f16", "bf16", "f32", "q8_0"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "arch": args.arch,
                    "slots": [s[2] for s in ORDERED_GGUF_SLOTS],
                    "outtype": args.outtype,
                },
                indent=2,
            )
        )
        return 0

    cart_map = {branch: args.cartridges_dir / branch for _, branch, _ in ORDERED_GGUF_SLOTS[1:]}
    res = convert_all_five_slots_to_gguf(
        backbone_dir=args.backbone_dir,
        cartridges=cart_map,
        out_dir=args.out_dir,
        convert_script=args.convert_script,
        base_config_dir=args.base_config_dir,
        arch=args.arch,
        outtype=args.outtype,
    )
    print(json.dumps({"event": "gguf_loras_converted", "slots": res}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
