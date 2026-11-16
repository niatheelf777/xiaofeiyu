#!/usr/bin/env python3
"""Unit tests for Xiaofeiyu open-source training, quality gate, Online DPO miner, and 5-slot GGUF exporter."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

TRAIN_DIR = Path(__file__).resolve().parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from export_gguf_loras import ORDERED_GGUF_SLOTS, is_thin_voice_target  # noqa: E402
from kernels_g4 import ARCH_DEFAULTS, SCOPE_SPECS, LoRALinear, set_model_token_gate_mask  # noqa: E402
from mine_online_dpo import mine_logs  # noqa: E402
from quality_gate import evaluate_rewrite, select_persona  # noqa: E402
import train_anti_smuggle_dpo  # noqa: E402
import train_style_cartridges  # noqa: E402
import train_thin_voice_sft  # noqa: E402


def _make_sample_sft_row(row_id: str = "r1") -> dict:
    # Token 52 (<|"|>) wraps two supervised tokens [100, 101] then closing 52
    ids = [2, 10, 52, 100, 101, 52, 11, 106]
    labels = [-100, -100, -100, 100, 101, 52, -100, -100]
    return {"id": row_id, "key": row_id, "source": "daily", "input_ids": ids, "labels": labels, "regions": [[3, 6]]}


def test_quality_gate_and_persona_schedule():
    assert select_persona(0)[0] == "xiaofeiyu_v4"
    assert select_persona(1)[0] == "neko_clingy"

    ok = evaluate_rewrite("帮你写好了\n```python\ndef rate_limit():\n    return True\n```\n拿去直接跑", frontier_reference="`rate_limit()`")
    assert ok.passed is True

    smuggled = evaluate_rewrite("给你整理好了\n```markdown\n1. 第一点\n2. 第二点\n```", frontier_reference="")
    assert smuggled.passed is False
    assert "prose_wrapped_in_markdown_fence" in smuggled.reasons

    emoji_bad = evaluate_rewrite("今晚风好大呀 😊")
    assert emoji_bad.passed is False
    assert "contains_emoji" in emoji_bad.reasons


def test_arch_specs_and_bounds():
    assert ARCH_DEFAULTS["12b"]["layer_scope"] == "top12_no_qkv"
    assert SCOPE_SPECS["top12_no_qkv"][3] == 48
    assert SCOPE_SPECS["e4b_upper20_no_qkv"][3] == 80
    assert SCOPE_SPECS["26b_a4b_top10_no_qkv"][3] == 40
    assert len(ORDERED_GGUF_SLOTS) == 5
    assert is_thin_voice_target("model.language_model.layers.36.mlp.gate_proj", 36, 47) is True
    assert is_thin_voice_target("model.language_model.layers.36.self_attn.q_proj", 36, 47) is False


def test_trainers_dry_run_and_five_slot_export(tmp_path: Path, capsys):
    assert len(ORDERED_GGUF_SLOTS) == 5
    assert is_thin_voice_target("model.language_model.layers.36.mlp.gate_proj", 36, 47) is True
    assert is_thin_voice_target("model.language_model.layers.36.self_attn.q_proj", 36, 47) is False

    sft_dir = tmp_path / "sft"
    sft_dir.mkdir()
    (sft_dir / "sft.jsonl").write_text(json.dumps(_make_sample_sft_row("s1")) + "\n", encoding="utf8")

    rc = train_thin_voice_sft.main(["--arch", "e4b", "--data", str(sft_dir), "--output", str(tmp_path / "out_sft"), "--dry-run"])
    assert rc == 0
    plan_sft = json.loads(capsys.readouterr().out)
    assert plan_sft["linear_modules"] == 80
    assert plan_sft["qkv_modules"] == 0

    cart_root = tmp_path / "carts"
    for cid in ("neko_clingy", "older_tease", "brat_mesugaki", "fragile_porcelain"):
        d = cart_root / cid
        d.mkdir(parents=True)
        (d / "sft.jsonl").write_text(json.dumps(_make_sample_sft_row(cid)) + "\n", encoding="utf8")

    rc2 = train_style_cartridges.main([
        "--arch", "12b",
        "--data-root", str(cart_root),
        "--backbone-dir", str(tmp_path / "bb"),
        "--out-cartridges-dir", str(tmp_path / "out_carts"),
        "--dry-run",
    ])
    assert rc2 == 0
    plan_c = json.loads(capsys.readouterr().out)
    assert plan_c["linear_modules"] == 48
    assert len(plan_c["slots"]) == 4

    init_dir = tmp_path / "init_adapter"
    init_dir.mkdir()
    (init_dir / "adapter.json").write_text(json.dumps({"layer_scope": "e4b_upper20_no_qkv", "rank": 16, "alpha": 32.0}), encoding="utf8")
    pairs_path = tmp_path / "pairs.jsonl"
    row = _make_sample_sft_row("p1")
    pairs_path.write_text(
        json.dumps({
            "id": "p1",
            "chosen_input_ids": row["input_ids"],
            "chosen_labels": row["labels"],
            "rejected_input_ids": row["input_ids"],
            "rejected_labels": row["labels"],
        }) + "\n",
        encoding="utf8",
    )
    rc3 = train_anti_smuggle_dpo.main([
        "--arch", "e4b",
        "--init-adapter", str(init_dir),
        "--pairs", str(pairs_path),
        "--output", str(tmp_path / "out_dpo"),
        "--dry-run",
    ])
    assert rc3 == 0
    plan_dpo = json.loads(capsys.readouterr().out)
    assert plan_dpo["linear_modules"] == 80


def test_online_dpo_miner_scrubs_pii_and_separates_smuggling(tmp_path: Path):
    log_path = tmp_path / "turns.jsonl"
    rows = [
        {
            "visitorId": "v_user_1",
            "conversationId": "c1",
            "userMessage": "联系我 alice@example.com 帮我看看异步限流",
            "assistantReply": "邮箱先帮你记下啦\n今天风大记得穿外套",
            "latencyMs": 1200,
            "weights": {"xiaofeiyu_v4": 1.0, "neko_clingy": 0.4},
        },
        {
            "visitorId": "v_user_2",
            "conversationId": "c2",
            "userMessage": "讲讲 GIL",
            "assistantReply": "给你整理好了\n```markdown\n1. GIL 是全局锁\n```",
            "latencyMs": 900,
        },
    ]
    log_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")
    out_dir = tmp_path / "mined"
    st = mine_logs(log_path, out_dir, write=True)
    assert st.scanned == 2
    assert st.kept_clean == 1
    assert st.dropped_tainted == 1
    pos_text = (out_dir / "clean-positives.jsonl").read_text(encoding="utf8")
    assert "[EMAIL]" in pos_text
    assert "alice@example.com" not in pos_text
