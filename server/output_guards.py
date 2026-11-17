#!/usr/bin/env python3
"""Sampling presets, streaming loop circuit breakers, slider simplex projection, and IM hygiene guards."""

from __future__ import annotations

import json
import math
import re
from typing import Any

STYLE_L1_BUDGET = 1.15
BACKBONE_MAX_SCALE = 1.15
DEFAULT_PUBLIC_MAX_THOUGHT_TOKENS = 320
DEFAULT_CASUAL_MAX_THOUGHT_TOKENS = 128
DEFAULT_AGENT_MAX_THOUGHT_TOKENS = 512
DEFAULT_MAX_SPOKEN_BUBBLES = 5
DEFAULT_MAX_BUBBLE_CHARS = 140

THOUGHT_DRY_BREAKERS: list[str] = ["\n", ":", '"', "*", '<|"|>', "<|tool_call>", "<tool_call|>"]
VOICE_DRY_BREAKERS: list[str] = ["\n", '<|"|>', "<|tool_call>", "<tool_call|>"]

PHASE_A_THOUGHT_SAMPLING: dict[str, Any] = {
    "dry_multiplier": 0.8,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": 768,
    "dry_sequence_breakers": THOUGHT_DRY_BREAKERS,
    "repeat_penalty": 1.06,
    "repeat_last_n": 256,
}

PHASE_B_VOICE_SAMPLING: dict[str, Any] = {
    "dry_multiplier": 0.85,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": 1024,
    "dry_sequence_breakers": VOICE_DRY_BREAKERS,
    "repeat_penalty": 1.10,
    "repeat_last_n": 512,
    "presence_penalty": 0.15,
    "frequency_penalty": 0.10,
}

PROSE_FENCE_LANGS: frozenset[str] = frozenset(
    {"", "text", "txt", "plain", "plaintext", "markdown", "md"}
)

EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\u2764\ufe0f\u200d]+")
_STAGE_ACTION_RE = re.compile(
    r"(?:（\s*(?:叹气|轻叹|轻笑|偷笑|冷笑|微笑|苦笑|皱眉|挑眉|低头|抬头|摇头|点头|沉默|停顿|喘口气|深呼吸|揉眼睛|眨眼|脸红|小声|轻声|翻白眼|伸懒腰|打哈欠)[^）\n]{0,12}）"
    r"|\[\s*(?:动作|叹气|轻笑|笑|微笑|皱眉|沉默|停顿|脸红|翻白眼)[^\]\n]{0,12}\])"
)
_THOUGHT_SUB_LOOP_RE = re.compile(r"(.{6,140}?)\1{2,}", re.S)
_THOUGHT_SHORT_STUTTER_RE = re.compile(r"([^\s#*`\-])\1{9,}|(.{2,5}?)\2{6,}", re.S)
_THOUGHT_CLAUSE_CLEAN_RE = re.compile(r"^[\s*\-•0-9.)]+")

_SINGLE_CHAR_STUTTER_RE = re.compile(
    r"([^\s哈呵嘿嘻呜喵哼嗯啊哦噢诶欸~～.。!！?？…—\-0-9A-Za-z])\1{3,}"
)
_EXPRESSIVE_OVERSTUTTER_RE = re.compile(r"(.)\1{7,}")
_SHORT_UNIT_LOOP_RE = re.compile(r"(.{2,8}?)\1{2,}")
_PHRASE_LOOP_RE = re.compile(r"(.{9,48}?)\1{1,}")
_JUNK_BUBBLE_RE = re.compile(r'^[\s"\'“”‘’`、，,.:：;；\-—_]+$')
_NORM_PUNCT_RE = re.compile(r'[\s"\'“”‘’`、，,。！？!?.:：;；~～…—\-（）()\[\]【】]+')

_EXPLICIT_DOC_REQUEST_RE = re.compile(
    r"(?:写一?[首篇份段个条]|创作|作一?[首篇]|诗|词|歌|赋|小说|故事|小作文|作文|长文|文章|信件|情书"
    r"|策划|方案|报告|大纲|文案|润色|改写|扩写|缩写|翻译|表格|明细表|清单|列表|课表|菜单|对比表"
    r"|汇总|总结|整理成|步骤|教程|指南|公式|证明|推导|代码|脚本|函数|实现|配置|正则|命令"
    r"|sql|html|css|json|yaml|toml|csv|markdown|\bmd\b|python|rust|java|cpp|js|ts"
    r"|背诵|圆周率|\d+\s*[字句行段条篇])",
    re.I,
)
_PRACTICAL_TABLE_QUERY_RE = re.compile(
    r"(?:怎么|如何|咋做|哪些|区别|对比|推荐|规划|模块|系统|方法|路径|注册表|环境变量"
    r"|久坐|压力|减肥|热量|食谱|行程|攻略|预算|账单|平摊|辅导|这道题|应用题)",
    re.I,
)
_CODE_SYNTAX_RE = re.compile(
    r"(?:^\s*(?:def |class |import |from \S+ import |function |const |let |SELECT |INSERT |UPDATE |CREATE TABLE"
    r"|<!DOCTYPE|<html|#include|fn \w+\s*\(|pub fn |package main)|\n\s*\{[\s\S]*\"[^\"]+\"\s*:)",
    re.M,
)
_MD_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]{3,}\|\s*$", re.M)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_WORD_RE = re.compile(r"\b[A-Za-z]{2,}\b")
_FOREIGN_LANG_REQ_RE = re.compile(
    r"(?:english|英文|英语|外语|日语|法语|德语|韩语|俄语|西语|翻译|translate|in english)",
    re.I,
)
_JSON_BUBBLES_RE = re.compile(r'"bubbles"\s*:\s*\[(.*?)\]', re.S)
_JSON_STR_ITEM_RE = re.compile(r'"((?:\\.|[^"\\])*)"')


def normalize_lora_weights(
    weights: tuple[float, ...] | list[float],
    *,
    max_backbone: float = BACKBONE_MAX_SCALE,
    max_style_l1: float = STYLE_L1_BUDGET,
) -> tuple[float, float, float, float, float]:
    """Clamp backbone `s0` to `[0, max_backbone]` and project style cartridges `c1..c4` onto `L1 <= max_style_l1`."""
    raw = list(weights)[:5]
    while len(raw) < 5:
        raw.append(0.0)
    w0 = float(raw[0]) if math.isfinite(float(raw[0])) else 1.0
    w0 = min(max_backbone, max(0.0, w0))

    styles = [max(0.0, float(x)) if math.isfinite(float(x)) else 0.0 for x in raw[1:5]]
    style_sum = sum(styles)
    if style_sum > max_style_l1 and style_sum > 1e-9:
        factor = max_style_l1 / style_sum
        styles = [s * factor for s in styles]

    return (
        round(w0, 4),
        round(styles[0], 4),
        round(styles[1], 4),
        round(styles[2], 4),
        round(styles[3], 4),
    )


def find_thought_loop_cut(thought_text: str) -> int | None:
    """Return character index to truncate `<|channel>thought` if a repetition loop is active, else None."""
    n = len(thought_text)
    if n < 100:
        return None

    win_len = min(n, 900)
    win_start = n - win_len
    window = thought_text[win_start:]

    m_sub = _THOUGHT_SUB_LOOP_RE.search(window)
    if m_sub:
        unit = m_sub.group(1)
        if unit.strip():
            return win_start + m_sub.start() + len(unit)

    m_stutter = _THOUGHT_SHORT_STUTTER_RE.search(window)
    if m_stutter and len(m_stutter.group(0)) >= 12:
        unit = m_stutter.group(1) or m_stutter.group(2) or ""
        return win_start + m_stutter.start() + max(1, min(len(unit) * 2, 4))

    lines = window.splitlines(keepends=True)
    if len(lines) >= 8:
        counts: dict[str, int] = {}
        pos = win_start
        for ln in lines:
            norm = _THOUGHT_CLAUSE_CLEAN_RE.sub("", ln).strip().lower()
            if len(norm) >= 8:
                c = counts.get(norm, 0) + 1
                counts[norm] = c
                if c >= 4:
                    return pos
            pos += len(ln)

    return None


def clean_spoken_bubble(text: str) -> str:
    """Remove leaked decorative emojis and explicit stage-action brackets from a spoken IM bubble."""
    if not text:
        return ""
    cleaned = EMOJI_RE.sub("", text)
    cleaned = _STAGE_ACTION_RE.sub("", cleaned)
    return cleaned.strip()


def find_bubble_stutter_cut(bubble_text: str) -> int | None:
    """Return safe cut offset if `bubble_text` develops an intra-bubble character/word/clause stutter."""
    if len(bubble_text) < 4:
        return None

    m1 = _SINGLE_CHAR_STUTTER_RE.search(bubble_text)
    if m1:
        return m1.start() + 2

    m_exp = _EXPRESSIVE_OVERSTUTTER_RE.search(bubble_text)
    if m_exp:
        ch = m_exp.group(1)
        if not ch.isalnum():
            return m_exp.start() + 4
        if ord(ch) > 127:
            return m_exp.start() + 4

    m2 = _SHORT_UNIT_LOOP_RE.search(bubble_text)
    if m2:
        unit = m2.group(1)
        span = len(m2.group(0))
        is_gentle_laugh = set(unit) <= set("哈呵嘿嘻呜喵哼嗯~～") and span <= 8
        if span >= 6 and not is_gentle_laugh:
            cut = m2.start() + len(unit)
            while cut > m2.start() and bubble_text[cut - 1] in "，,、":
                cut -= 1
            return cut

    m3 = _PHRASE_LOOP_RE.search(bubble_text)
    if m3:
        cut = m3.start() + len(m3.group(1))
        while cut > m3.start() and bubble_text[cut - 1] in "，,、":
            cut -= 1
        return cut

    return None


def is_junk_bubble(text: str) -> bool:
    """Return True for empty or pure quote/punctuation noise bubbles such as `\"`."""
    s = (text or "").strip()
    if not s:
        return True
    return bool(_JUNK_BUBBLE_RE.match(s))


def _char_bigrams(s: str) -> set[str]:
    if len(s) < 2:
        return {s} if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def is_duplicate_bubble(candidate: str, prior_bubbles: list[str]) -> bool:
    """Return True if `candidate` repeats or near-duplicates an already emitted bubble in the same turn."""
    cand_norm = _NORM_PUNCT_RE.sub("", candidate or "").lower()
    if not cand_norm:
        return True
    cand_bg = _char_bigrams(cand_norm)
    for prev in prior_bubbles:
        prev_norm = _NORM_PUNCT_RE.sub("", prev or "").lower()
        if not prev_norm:
            continue
        if cand_norm == prev_norm:
            return True
        if len(cand_norm) >= 6 and len(prev_norm) >= 6 and cand_bg:
            prev_bg = _char_bigrams(prev_norm)
            overlap = len(cand_bg & prev_bg)
            dice = (2.0 * overlap) / (len(cand_bg) + len(prev_bg))
            if dice >= 0.82:
                return True
    return False


def is_real_code_language(lang: str) -> bool:
    """Return True when `send_code_or_doc.language` denotes real source code, markup, query, or data."""
    return (lang or "").strip().lower() not in PROSE_FENCE_LANGS


def unwrap_prose_card_to_bubbles(body: str, *, max_bubbles: int = 3) -> list[str]:
    """Convert an unsolicited Markdown/text chat card into 1..max_bubbles natural spoken lines."""
    lines: list[str] = []
    for raw_ln in (body or "").splitlines():
        ln = raw_ln.strip()
        if not ln or _MD_TABLE_SEP_RE.match(ln):
            continue
        if ln.startswith("#"):
            continue
        if ln.startswith("|") and ln.endswith("|"):
            cells = [c.strip() for c in ln.strip("|").split("|") if c.strip()]
            if len(cells) >= 2:
                ln = "：".join(cells[:2])
            elif cells:
                ln = cells[0]
        ln = re.sub(r"^\s*(?:[-*•]+|\d+[.)、])\s*", "", ln)
        ln = re.sub(r"\*\*(.*?)\*\*", r"\1", ln)
        ln = re.sub(r"__(.*?)__", r"\1", ln)
        ln = re.sub(r"`([^`]+)`", r"\1", ln)
        ln = clean_spoken_bubble(ln)
        if not ln or is_junk_bubble(ln):
            continue
        if len(ln) > 90:
            parts = [p.strip() for p in re.split(r"[。！？!?]\s*", ln) if p.strip()]
            if parts:
                ln = parts[0]
        if ln and not is_duplicate_bubble(ln, lines):
            lines.append(ln[:110])
        if len(lines) >= max_bubbles:
            break
    return lines


def resolve_code_or_doc_emission(
    language: str,
    content: str,
    user_last_message: str,
    spoken_bubbles_emitted: int,
) -> tuple[str, str, str]:
    """Classify `send_code_or_doc` into `('fence', lang, body)`, `('bubbles', '', joined)`, or `('drop', '', '')`."""
    clean_lang = re.sub(r"[^a-zA-Z0-9_+#.-]", "", (language or "").strip())
    raw_body = (content or "").rstrip("\r\n")
    if not raw_body.strip():
        return ("drop", "", "")

    if is_real_code_language(clean_lang):
        return ("fence", clean_lang, raw_body)

    user_txt = user_last_message or ""
    wants_emoji = bool(re.search(r"emoji|表情", user_txt, re.I))
    clean_body = raw_body if wants_emoji else EMOJI_RE.sub("", raw_body)
    clean_body = re.sub(r"[ \t]+\n", "\n", clean_body).strip()
    if not clean_body:
        return ("drop", "", "")

    has_table = bool(_MD_TABLE_SEP_RE.search(clean_body))
    if (
        _EXPLICIT_DOC_REQUEST_RE.search(user_txt)
        or _CODE_SYNTAX_RE.search(clean_body)
        or (has_table and _PRACTICAL_TABLE_QUERY_RE.search(user_txt))
    ):
        return ("fence", clean_lang or "text", clean_body)

    if spoken_bubbles_emitted >= 1:
        return ("drop", "", "")

    unwrapped = unwrap_prose_card_to_bubbles(clean_body, max_bubbles=max(1, 3 - spoken_bubbles_emitted))
    if not unwrapped:
        return ("drop", "", "")
    return ("bubbles", "", "\n".join(unwrapped))


def extract_json_bubbles_fallback(text: str) -> list[str]:
    """Rescue spoken bubbles if Phase A accidentally formatted `send_message` as JSON inside thought/text."""
    if not text or '"bubbles"' not in text:
        return []
    m = _JSON_BUBBLES_RE.search(text)
    if not m:
        return []
    out: list[str] = []
    for sm in _JSON_STR_ITEM_RE.finditer(m.group(1)):
        raw_item = sm.group(1)
        try:
            decoded = json.loads(f'"{raw_item}"')
        except Exception:
            decoded = raw_item
        cleaned = clean_spoken_bubble(str(decoded))
        if cleaned and not is_junk_bubble(cleaned) and not is_duplicate_bubble(cleaned, out):
            out.append(cleaned[:DEFAULT_MAX_BUBBLE_CHARS])
        if len(out) >= 4:
            break
    return out


def is_unrequested_english_drift(bubbles: list[str], user_last_message: str) -> bool:
    """Return True when the user wrote in Chinese without requesting foreign text, yet all bubbles are English."""
    u_txt = user_last_message or ""
    if not _CJK_RE.search(u_txt) or _FOREIGN_LANG_REQ_RE.search(u_txt):
        return False
    joined = " ".join(b for b in bubbles if b and not b.lstrip().startswith("```")).strip()
    if not joined:
        return False
    if _CJK_RE.search(joined):
        return False
    return len(_LATIN_WORD_RE.findall(joined)) >= 3


_DEEP_REASONING_QUERY_RE = re.compile(
    r"(?:\d+\s*[+\-*/÷×%=<>]|平摊|算一?[算下]|几点|周几|星期几|日期|天数|概率|方程|几何|猜想|证明|推导"
    r"|东、?西|千米|速度|代码|脚本|函数|报错|bug|debug|实现|算法|复杂度|python|rust|js|ts|sql|html|css|json|csv"
    r"|写一?[首篇份段]|小说|故事|作文|策划|方案|文案|润色|改写|翻译|对比|表格|清单)",
    re.I,
)


_TOOL_NEGATION_RE = re.compile(
    r"(?:\bno\b|\bnot\b|don't|doesn't|didn't|won't|needn't|without|never|avoid|skip|unnecessary|none\b"
    r"|\?\s*no\b|constraints?\s*:|不需要|不用|无需|禁止|不要|不调|没必要|不发|不使)",
    re.I,
)


def thought_affirmatively_plans_tool(thought_text: str, tool_name: str) -> bool:
    """Return True only if `thought_text` mentions `tool_name` in an affirmative clause without local negation."""
    if not thought_text or not tool_name or tool_name not in thought_text:
        return False
    pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(tool_name)}(?![A-Za-z0-9_])")
    for m in pat.finditer(thought_text):
        line_start = thought_text.rfind("\n", 0, m.start()) + 1
        line_end = thought_text.find("\n", m.end())
        if line_end == -1:
            line_end = len(thought_text)
        clause = thought_text[max(line_start, m.start() - 64) : min(line_end, m.end() + 48)]
        if not _TOOL_NEGATION_RE.search(clause):
            return True
    return False


def compute_thought_token_budget(
    user_last_message: str,
    *,
    configured_max: int = DEFAULT_PUBLIC_MAX_THOUGHT_TOKENS,
    has_external_tools: bool = False,
) -> int:
    """Select a bounded Phase-A `<|channel>thought` budget appropriate to task complexity."""
    if configured_max <= 0:
        return 0
    if has_external_tools:
        return min(int(configured_max), DEFAULT_AGENT_MAX_THOUGHT_TOKENS)
    u_txt = (user_last_message or "").strip()
    if len(u_txt) >= 120 or _DEEP_REASONING_QUERY_RE.search(u_txt):
        return min(int(configured_max), DEFAULT_PUBLIC_MAX_THOUGHT_TOKENS)
    return min(int(configured_max), DEFAULT_CASUAL_MAX_THOUGHT_TOKENS)

