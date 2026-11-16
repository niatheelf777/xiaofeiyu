#!/usr/bin/env python3
"""Open-source Online DPO preference miner for self-hosted Xiaofeiyu public logs.

Sanitizes anonymized chat-turn JSONL records in two passes (conversation taint + turn hygiene)
and emits clean positive/negative signal shards ready for preference-pair construction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from quality_gate import (  # noqa: E402
    CS_BOLD_HEADER_RE,
    CS_BULLET_LIST_RE,
    CS_HEADING_RE,
    CS_NUMBERED_LIST_RE,
    DEFENSIVE_CS_PHRASES,
    EMOJI_RE,
    META_LEAK_PHRASES,
    STAGE_ASTERISK_RE,
    STAGE_BRACKET_RE,
    STAGE_PAREN_RE,
    extract_code_blocks,
    has_prose_markdown_card,
    is_code_bubble,
    split_reply_bubbles,
)

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
CN_PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SECRET_TOKEN_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{16,})\b")

INJECTION_RE = re.compile(r"(?i)(ignore\s+previous\s+instructions|system\s+prompt|忽略之前的指令)")
SMOKE_TEST_RE = re.compile(r"(?i)\b(?:smoke|canary|probe|test)\b")
NEGATIVE_CORRECTION_RE = re.compile(r"(不对|错了|答错了|别用|不要写|太长|好啰嗦|说人话|别像客服)")

SLIDER_NAMES = ("xiaofeiyu_v4", "neko_clingy", "older_tease", "brat_mesugaki", "fragile_porcelain")


def scrub_pii(text: str) -> str:
    if not text:
        return ""
    s = EMAIL_RE.sub("[EMAIL]", text)
    s = CN_PHONE_RE.sub("[PHONE]", s)
    s = IPV4_RE.sub("[IP]", s)
    s = SECRET_TOKEN_RE.sub("[TOKEN]", s)
    return s


def anonymize_id(raw: str, prefix: str = "anon") -> str:
    digest = hashlib.sha256(str(raw or "").encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def normalize_weights(raw: object) -> dict[str, float]:
    out = {k: 0.0 for k in SLIDER_NAMES}
    out["xiaofeiyu_v4"] = 1.0
    if isinstance(raw, dict):
        for k in SLIDER_NAMES:
            if k in raw:
                try:
                    out[k] = round(max(0.0, min(1.5, float(raw[k]))), 3)
                except (TypeError, ValueError):
                    pass
    return out


def classify_probe_type(user_text: str, assistant_text: str) -> str:
    low = (user_text + "\n" + assistant_text).lower()
    if "```" in assistant_text or any(w in low for w in ("python", "sql", "code", "bug", "报错", "代码")):
        return "coding_or_debug"
    if any(w in low for w in ("怎么", "为什么", "原理", "区别", "解释")):
        return "qa_explain"
    return "daily_chat"


def collect_taint_flags(user_msg: str, reply: str, latency_ms: int) -> list[str]:
    flags: list[str] = []
    if not user_msg.strip():
        flags.append("empty_user_message")
    if not reply.strip():
        flags.append("empty_reply")
    if latency_ms > 60_000:
        flags.append("high_latency")
    if INJECTION_RE.search(user_msg):
        flags.append("prompt_injection")
    if SMOKE_TEST_RE.search(user_msg):
        flags.append("synthetic_smoke")
    if EMOJI_RE.search(reply):
        flags.append("contains_emoji")
    if STAGE_PAREN_RE.search(reply) or STAGE_BRACKET_RE.search(reply) or STAGE_ASTERISK_RE.search(reply):
        flags.append("contains_stage_direction")

    bubbles = split_reply_bubbles(reply)
    spoken = [b for b in bubbles if not is_code_bubble(b)]
    code_b = [b for b in bubbles if is_code_bubble(b)]
    if has_prose_markdown_card(code_b, extract_code_blocks(user_msg)):
        flags.append("prose_markdown_smuggling")
    for sb in spoken:
        if CS_HEADING_RE.search(sb) or CS_NUMBERED_LIST_RE.search(sb) or CS_BULLET_LIST_RE.search(sb):
            flags.append("cs_list_or_heading")
            break
    if CS_BOLD_HEADER_RE.search("\n".join(spoken)):
        flags.append("cs_bold_header")
    low = "\n".join(spoken).lower()
    if any(p in low for p in META_LEAK_PHRASES):
        flags.append("meta_leak")
    if any(p in low for p in DEFENSIVE_CS_PHRASES):
        flags.append("defensive_cs_phrase")
    return flags


@dataclass
class MineStats:
    scanned: int = 0
    dropped_tainted: int = 0
    kept_clean: int = 0
    positive_candidates: int = 0
    negative_candidates: int = 0
    drop_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def mine_logs(input_path: Path, output_dir: Path, *, write: bool = True) -> MineStats:
    records: list[dict] = []
    with Path(input_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    tainted_convs: set[str] = set()
    for rec in records:
        conv_id = str(rec.get("conversationId") or "")
        user_msg = str(rec.get("userMessage") or "")
        if INJECTION_RE.search(user_msg) or SMOKE_TEST_RE.search(user_msg):
            if conv_id:
                tainted_convs.add(conv_id)

    stats = MineStats(scanned=len(records))
    pos_rows: list[dict] = []
    neg_rows: list[dict] = []

    for idx, rec in enumerate(records):
        conv_id = str(rec.get("conversationId") or f"conv_{idx}")
        user_msg = scrub_pii(str(rec.get("userMessage") or ""))
        reply = scrub_pii(str(rec.get("assistantReply") or ""))
        latency = int(rec.get("latencyMs") or 0)
        flags = collect_taint_flags(user_msg, reply, latency)
        if conv_id in tainted_convs:
            flags.append("tainted_conversation")

        clean_rec = {
            "anon_visitor": anonymize_id(str(rec.get("visitorId") or "v")),
            "anon_conversation": anonymize_id(conv_id),
            "weights": normalize_weights(rec.get("weights")),
            "probe_type": classify_probe_type(user_msg, reply),
            "user_message": user_msg,
            "assistant_reply": reply,
            "taint_flags": flags,
        }

        if not flags:
            stats.kept_clean += 1
            if NEGATIVE_CORRECTION_RE.search(user_msg):
                stats.negative_candidates += 1
                neg_rows.append(clean_rec)
            else:
                stats.positive_candidates += 1
                pos_rows.append(clean_rec)
        else:
            stats.dropped_tainted += 1
            for fl in flags:
                stats.drop_reasons[fl] += 1
            if any(f in flags for f in ("prose_markdown_smuggling", "cs_list_or_heading", "defensive_cs_phrase")):
                stats.negative_candidates += 1
                neg_rows.append(clean_rec)

    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "clean-positives.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pos_rows), encoding="utf-8"
        )
        (output_dir / "rejected-negatives.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in neg_rows), encoding="utf-8"
        )
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    st = mine_logs(args.input, args.output_dir, write=not args.dry_run)
    print(
        json.dumps(
            {
                "scanned": st.scanned,
                "kept_clean": st.kept_clean,
                "dropped_tainted": st.dropped_tainted,
                "positive_candidates": st.positive_candidates,
                "negative_candidates": st.negative_candidates,
                "drop_reasons": dict(st.drop_reasons),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
