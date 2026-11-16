#!/usr/bin/env python3
"""Deterministic spoken-IM hygiene, 5-slider persona schedule, and technical-retention gate."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Sequence, Tuple

PERSONA_CYCLE_8: List[Tuple[str, List[float]]] = [
    ("xiaofeiyu_v4", [1.0, 0.0, 0.0, 0.0, 0.0]),
    ("neko_clingy", [1.0, 0.85, 0.0, 0.0, 0.0]),
    ("xiaofeiyu_v4", [1.0, 0.0, 0.0, 0.0, 0.0]),
    ("older_tease", [1.0, 0.0, 0.85, 0.0, 0.0]),
    ("xiaofeiyu_v4", [1.0, 0.0, 0.0, 0.0, 0.0]),
    ("brat_mesugaki", [1.0, 0.0, 0.0, 0.85, 0.0]),
    ("xiaofeiyu_v4", [1.0, 0.0, 0.0, 0.0, 0.0]),
    ("fragile_porcelain", [1.0, 0.0, 0.0, 0.0, 0.85]),
]


def select_persona(index: int) -> Tuple[str, List[float]]:
    return PERSONA_CYCLE_8[index % len(PERSONA_CYCLE_8)]


def format_model_tag(weights: Sequence[float]) -> str:
    arr = list(weights) if len(weights) == 5 else [1.0, 0.0, 0.0, 0.0, 0.0]
    return "xiaofeiyu-v4|" + ",".join(f"{float(v):.3f}" for v in arr)


def split_reply_bubbles(raw_content: str) -> List[str]:
    if not isinstance(raw_content, str):
        return []
    normalized = raw_content.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    bubbles: List[str] = []
    in_fence = False
    fence_lines: List[str] = []

    for line in lines:
        if not in_fence:
            open_match = re.match(r"^\s*```([^\r\n`]*)$", line)
            if open_match:
                in_fence = True
                fence_lines = [f"```{open_match.group(1).strip()}"]
            else:
                trimmed = line.strip()
                if trimmed:
                    bubbles.append(trimmed)
        elif re.match(r"^\s*```\s*$", line):
            fence_lines.append("```")
            bubbles.append("\n".join(fence_lines))
            in_fence = False
            fence_lines = []
        else:
            fence_lines.append(line)

    if in_fence and fence_lines:
        fence_lines.append("```")
        bubbles.append("\n".join(fence_lines))

    return bubbles


def is_code_bubble(bubble: str) -> bool:
    return bubble.lstrip().startswith("```")


EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\u200d\ufe0f"
    "]",
    flags=re.UNICODE,
)

STAGE_ACTION_WORDS = (
    "笑", "偷笑", "坏笑", "苦笑", "冷笑", "干笑",
    "叹气", "叹了口气", "叹息", "点头", "摇头", "挑眉", "皱眉", "撇嘴", "嘟嘴",
    "脸红", "耳根红", "小声", "嘀咕", "轻声", "低声", "捂脸", "歪头", "托腮",
    "蹭", "蹭蹭", "抱", "抱住", "抱抱", "捏", "捏脸", "戳", "戳戳", "伸懒腰",
    "打哈欠", "摸摸", "摸头", "拍拍", "瞪", "眯眼", "吐舌", "眨眼", "摆手",
    "翻白眼", "轻哼", "哼", "委屈", "可怜", "瑟瑟发抖", "抓衣角", "低头",
    "仰头", "凑近", "贴贴", "靠过来", "躺平", "打字", "举手", "耸肩", "摊手",
    "动作", "心理", "内心", "os", "OS",
)
STAGE_PAREN_RE = re.compile(
    r"[（(]\s*(?:" + "|".join(re.escape(w) for w in STAGE_ACTION_WORDS) + r")[^)）]{0,16}[)）]"
)
STAGE_BRACKET_RE = re.compile(r"\[[\u4e00-\u9fff]{1,10}\](?!\()")
STAGE_ASTERISK_RE = re.compile(r"(?<!\*)\*[\u4e00-\u9fff]{1,12}\*(?!\*)")

CS_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S")
CS_NUMBERED_LIST_RE = re.compile(r"^\s*\d+[.、)）]\s+\S")
CS_BULLET_LIST_RE = re.compile(r"^\s*[-*•]\s+\S")
CS_BOLD_HEADER_RE = re.compile(r"\*\*[^*]{2,24}\*\*\s*[：:]")

META_LEAK_PHRASES = (
    "作为ai", "作为人工智能", "根据参考答案", "根据参考", "参考答案",
    "参考回复", "参考回答", "改写如下", "翻译如下", "原回复", "原回答",
    "用小肥鱼的", "用女友的口吻", "已为你改写", "以下是改写",
)

DEFENSIVE_CS_PHRASES = (
    "希望能帮到你", "希望这能帮到你", "希望对你有帮助", "如有任何问题",
    "如有需要", "请随时告诉我", "随时告诉我", "随时找我", "欢迎随时",
    "出于安全考虑", "出于稳妥考虑", "基于目前信息", "根据你提供的信息",
    "这取决于你的具体场景", "这取决于你的具体需求", "取决于你的具体",
    "需要注意以下几点", "建议从以下几个方面", "可以从以下几个方面",
    "综上所述", "总而言之", "以下是详细", "下面为你介绍", "分步骤进行",
)

CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)
BACKTICK_IDENT_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_.:/-]{1,40})`")
ERROR_TOKEN_RE = re.compile(
    r"\b(?:HTTP\s*[45]\d{2}|[45]0\d|429|ECONNREFUSED|ETIMEDOUT|ENOENT|EACCES|OOM|SIGKILL|SIGTERM|CORS|SQLSTATE|ORA-\d+)\b"
)
CLI_FLAG_RE = re.compile(r"(?:^|\s)(--[a-zA-Z0-9][a-zA-Z0-9_-]{1,24})(?=\s|$|[`.,;])")
COMPLEXITY_RE = re.compile(r"O\([nmk1log\s+*^]+\)")


def extract_code_blocks(text: str) -> List[str]:
    if not text:
        return []
    return [m.group(1).strip() for m in CODE_FENCE_RE.finditer(text) if m.group(1).strip()]


def has_prose_markdown_card(code_bubbles: Sequence[str], ref_code_blocks: Sequence[str]) -> bool:
    for cb in code_bubbles:
        first_line = cb.split("\n", 1)[0].strip().lower()
        lang = first_line[3:].strip()
        if lang in ("markdown", "md", "text", "txt", "plain"):
            return True
        if not lang and not ref_code_blocks:
            if re.search(r"(?m)^\s*(?:#{1,4}\s|\*\*|[1-9]\.\s|\|.+\|)", cb):
                return True
    return False


def extract_technical_anchors(text: str) -> List[str]:
    if not text:
        return []
    anchors: List[str] = []
    seen = set()

    def _add(tok: str) -> None:
        clean = tok.strip()
        if not clean or len(clean) < 2:
            return
        key = clean.lower()
        if key not in seen:
            seen.add(key)
            anchors.append(clean)

    for m in BACKTICK_IDENT_RE.finditer(text):
        _add(m.group(1))
    for m in ERROR_TOKEN_RE.finditer(text):
        _add(m.group(0))
    for m in CLI_FLAG_RE.finditer(text):
        _add(m.group(1))
    for m in COMPLEXITY_RE.finditer(text):
        _add(m.group(0))
    return anchors


def _extract_code_identifiers(code_blocks: Sequence[str]) -> List[str]:
    idents: List[str] = []
    seen = set()
    stop = {
        "def", "class", "return", "import", "from", "const", "let", "var",
        "function", "true", "false", "null", "none", "print", "if", "else",
        "for", "while", "try", "except", "in", "as", "and", "or", "not",
        "select", "where",
    }
    for blk in code_blocks:
        for tok in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,30}\b", blk):
            low = tok.lower()
            if low in stop or low in seen:
                continue
            seen.add(low)
            idents.append(low)
    return idents


def _char_bigram_jaccard(a: str, b: str) -> float:
    a_clean = re.sub(r"\s+", "", a)
    b_clean = re.sub(r"\s+", "", b)
    if len(a_clean) < 8 or len(b_clean) < 8:
        return 0.0
    sa = {a_clean[i : i + 2] for i in range(len(a_clean) - 1)}
    sb = {b_clean[i : i + 2] for i in range(len(b_clean) - 1)}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class GateResult:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def evaluate_rewrite(
    raw_content: str,
    frontier_reference: str = "",
    bucket: str = "coding_debug_sql",
    engineering_facet: str = "non_engineering",
) -> GateResult:
    reasons: List[str] = []
    text = (raw_content or "").strip()

    if not text:
        return GateResult(passed=False, reasons=["empty_output"], metrics={"bubble_count": 0})

    if len(re.findall(r"```", text)) % 2 != 0:
        reasons.append("unclosed_code_fence")

    bubbles = split_reply_bubbles(text)
    spoken_bubbles = [b for b in bubbles if not is_code_bubble(b)]
    code_bubbles = [b for b in bubbles if is_code_bubble(b)]

    if not (1 <= len(bubbles) <= 5):
        reasons.append(f"bubble_count_out_of_range:{len(bubbles)}")

    if not spoken_bubbles:
        reasons.append("no_spoken_bubble")
        median_len = 0.0
        max_len = 0
    else:
        lengths = sorted(len(b) for b in spoken_bubbles)
        mid = len(lengths) // 2
        median_len = float(lengths[mid]) if len(lengths) % 2 == 1 else (lengths[mid - 1] + lengths[mid]) / 2.0
        max_len = lengths[-1]
        if not (4 <= median_len <= 42):
            reasons.append(f"median_bubble_len_out_of_range:{median_len:.1f}")
        if max_len > 80:
            reasons.append(f"max_bubble_len_exceeded:{max_len}")

    spoken_joined = "\n".join(spoken_bubbles)
    spoken_lower = spoken_joined.lower()

    if EMOJI_RE.search(text):
        reasons.append("contains_emoji")

    if (
        STAGE_PAREN_RE.search(spoken_joined)
        or STAGE_BRACKET_RE.search(spoken_joined)
        or STAGE_ASTERISK_RE.search(spoken_joined)
    ):
        reasons.append("contains_stage_direction")

    for sb in spoken_bubbles:
        if CS_HEADING_RE.search(sb) or CS_NUMBERED_LIST_RE.search(sb) or CS_BULLET_LIST_RE.search(sb):
            reasons.append("contains_cs_list_or_heading")
            break
    if CS_BOLD_HEADER_RE.search(spoken_joined):
        reasons.append("contains_bold_header")

    ref_code_blocks = extract_code_blocks(frontier_reference)
    if has_prose_markdown_card(code_bubbles, ref_code_blocks):
        reasons.append("prose_wrapped_in_markdown_fence")

    for phrase in META_LEAK_PHRASES:
        if phrase in spoken_lower:
            reasons.append(f"meta_leak:{phrase}")
            break

    for phrase in DEFENSIVE_CS_PHRASES:
        if phrase in spoken_lower:
            reasons.append(f"defensive_cs_phrase:{phrase}")
            break

    ref_no_code = CODE_FENCE_RE.sub("", frontier_reference or "").strip()
    sim = _char_bigram_jaccard(spoken_joined, ref_no_code)
    if len(spoken_joined) > 40 and sim > 0.78:
        reasons.append(f"verbatim_reference_leak:{sim:.2f}")

    out_code_blocks = extract_code_blocks(text)
    if ref_code_blocks:
        if not out_code_blocks:
            reasons.append("dropped_code_block")
        else:
            ref_idents = _extract_code_identifiers(ref_code_blocks)
            out_code_all = "\n".join(out_code_blocks).lower() + "\n" + spoken_lower
            if len(ref_idents) >= 2 and sum(1 for ident in ref_idents if ident in out_code_all) == 0:
                reasons.append("dropped_code_identifiers")

    is_tech = bucket in ("coding_debug_sql", "science_explain_troubleshoot") or engineering_facet.startswith("E")
    anchors = extract_technical_anchors(frontier_reference)
    kept_anchors = [a for a in anchors if a.lower() in text.lower()]
    if is_tech and len(anchors) >= 2:
        ratio = len(kept_anchors) / len(anchors)
        if len(kept_anchors) == 0 or (len(anchors) >= 4 and ratio < 0.25):
            reasons.append(f"dropped_technical_anchors:{len(kept_anchors)}/{len(anchors)}")

    return GateResult(
        passed=len(reasons) == 0,
        reasons=reasons,
        metrics={
            "bubble_count": len(bubbles),
            "spoken_bubble_count": len(spoken_bubbles),
            "code_bubble_count": len(code_bubbles),
            "median_spoken_len": median_len,
            "max_spoken_len": max_len,
            "ref_code_blocks": len(ref_code_blocks),
            "out_code_blocks": len(out_code_blocks),
            "anchors_total": len(anchors),
            "anchors_kept": len(kept_anchors),
            "reference_similarity": round(sim, 3),
        },
    )
