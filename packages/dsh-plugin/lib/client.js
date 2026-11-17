window.__ModuleLoader__.load({
  id: "@xiaofeiyu/dsh-plugin",
  factory: (require) => {
    var module = { exports: {} };
    var exports = module.exports;
    Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });

    const jsxRuntime = require("react/jsx-runtime");
    const react = require("react");

    const STYLE_TAG_ID = "@xiaofeiyu/dsh-plugin/styles.css";
    const STYLE_CSS = "/* Scoped Xiaofeiyu DSH WebUI styles: airy paper-and-pine IM bubbles, code cards & 5-slider mixer */\n.xfy-turn-tail {\n  --xfy-accent: #2d6a4f;\n  --xfy-accent-soft: rgba(45, 106, 79, 0.10);\n  --xfy-bubble-bg: var(--dsw-alias-bg-layer-2, #f7f6f1);\n  --xfy-bubble-border: var(--dsw-alias-border-l1, #e2ddd0);\n  --xfy-bubble-ink: var(--dsw-alias-label-primary, #1b2d24);\n  --xfy-meta-ink: var(--dsw-alias-label-tertiary, #6d7c74);\n  --xfy-code-bg: var(--dsw-alias-markdown-code-block, #f4f3ee);\n  display: flex;\n  flex-direction: column;\n  gap: 10px;\n  margin: 6px 0 10px;\n}\n\n.xfy-burst-row {\n  display: flex;\n  align-items: flex-start;\n  gap: 10px;\n  max-width: min(680px, 100%);\n}\n\n.xfy-avatar-mark {\n  width: 26px;\n  height: 26px;\n  border-radius: 9px;\n  background: var(--xfy-accent);\n  color: #faf9f4;\n  font-size: 12px;\n  font-weight: 600;\n  display: inline-flex;\n  align-items: center;\n  justify-content: center;\n  flex: none;\n  margin-top: 2px;\n  user-select: none;\n  box-shadow: 0 1px 3px rgba(27, 45, 36, 0.12);\n}\n\n.xfy-bubble-stack {\n  display: flex;\n  flex-direction: column;\n  align-items: flex-start;\n  gap: 5px;\n  min-width: 0;\n  flex: 1 1 auto;\n}\n\n.xfy-bubble-pill {\n  max-width: 100%;\n  padding: 9px 14px;\n  background: var(--xfy-bubble-bg);\n  border: 1px solid var(--xfy-bubble-border);\n  border-radius: 15px;\n  color: var(--xfy-bubble-ink);\n  font-size: var(--dsh-content-font-size-primary, 14px);\n  line-height: 1.56;\n  letter-spacing: 0.005em;\n  word-break: break-word;\n  white-space: pre-wrap;\n  box-shadow: 0 1px 4px rgba(27, 45, 36, 0.04);\n  animation: xfyBubbleRise 0.18s cubic-bezier(0.2, 0.8, 0.2, 1) both;\n}\n\n.xfy-bubble-pill[data-pos=\"first\"] {\n  border-top-left-radius: 5px;\n}\n\n.xfy-bubble-pill[data-pos=\"middle\"] {\n  border-top-left-radius: 5px;\n  border-bottom-left-radius: 5px;\n}\n\n.xfy-bubble-pill[data-pos=\"last\"] {\n  border-bottom-left-radius: 5px;\n}\n\n@keyframes xfyBubbleRise {\n  from {\n    opacity: 0;\n    transform: translateY(3px);\n  }\n  to {\n    opacity: 1;\n    transform: translateY(0);\n  }\n}\n\n.xfy-burst-footer {\n  display: inline-flex;\n  align-items: center;\n  gap: 7px;\n  margin-top: 2px;\n  padding-left: 2px;\n  font-size: 11.5px;\n  color: var(--xfy-meta-ink);\n}\n\n.xfy-flavor-chip {\n  display: inline-flex;\n  align-items: center;\n  gap: 5px;\n  padding: 1px 8px;\n  border-radius: 999px;\n  background: var(--xfy-accent-soft);\n  color: var(--xfy-accent);\n  font-size: 11px;\n  font-weight: 500;\n  line-height: 18px;\n}\n\n.xfy-weights-mono {\n  font-family: ui-monospace, \"SFMono-Regular\", \"Cascadia Mono\", Consolas, monospace;\n  font-variant-numeric: tabular-nums;\n  font-size: 11px;\n  opacity: 0.78;\n}\n\n.xfy-code-card {\n  border: 1px solid var(--xfy-bubble-border);\n  border-radius: 12px;\n  background: var(--xfy-code-bg);\n  overflow: hidden;\n  max-width: min(720px, 100%);\n}\n\n.xfy-code-header {\n  display: flex;\n  align-items: center;\n  justify-content: space-between;\n  gap: 8px;\n  padding: 6px 12px;\n  border-bottom: 1px solid var(--xfy-bubble-border);\n  font-size: 11.5px;\n  color: var(--xfy-meta-ink);\n}\n\n.xfy-code-lang {\n  font-family: ui-monospace, \"SFMono-Regular\", \"Cascadia Mono\", Consolas, monospace;\n  font-weight: 600;\n  color: var(--xfy-bubble-ink);\n  text-transform: lowercase;\n}\n\n.xfy-code-copy-btn {\n  appearance: none;\n  border: 1px solid var(--xfy-bubble-border);\n  background: transparent;\n  color: var(--xfy-bubble-ink);\n  border-radius: 6px;\n  padding: 2px 8px;\n  font-size: 11px;\n  cursor: pointer;\n  transition: background 0.12s ease;\n}\n\n.xfy-code-copy-btn:hover {\n  background: var(--xfy-accent-soft);\n}\n\n.xfy-code-pre {\n  margin: 0;\n  padding: 11px 14px;\n  overflow-x: auto;\n  font-family: ui-monospace, \"SFMono-Regular\", \"Cascadia Mono\", Consolas, monospace;\n  font-size: 12.5px;\n  line-height: 1.55;\n  color: var(--xfy-bubble-ink);\n}\n\n.xfy-diff-add {\n  display: block;\n  background: rgba(45, 106, 79, 0.12);\n}\n\n.xfy-diff-del {\n  display: block;\n  background: rgba(185, 64, 64, 0.12);\n}\n\n/* Compact Trajectory / Process Tool Row */\n.xfy-tool-row {\n  display: flex;\n  flex-direction: column;\n  gap: 4px;\n}\n\n.xfy-tool-row-line {\n  display: inline-flex;\n  align-items: center;\n  gap: 7px;\n  min-height: 24px;\n  font-size: var(--dsh-content-font-size-secondary, 13px);\n  color: var(--dsw-alias-label-tertiary, #6d7c74);\n  cursor: pointer;\n  user-select: none;\n}\n\n.xfy-tool-row-line:hover {\n  color: var(--dsw-alias-label-secondary, #394a41);\n}\n\n.xfy-tool-dot {\n  width: 6px;\n  height: 6px;\n  border-radius: 999px;\n  background: #2d6a4f;\n  flex: none;\n}\n\n.xfy-tool-dot[data-running=\"true\"] {\n  opacity: 0.55;\n}\n\n.xfy-tool-preview {\n  margin: 2px 0 4px 13px;\n  padding: 6px 10px;\n  border-left: 2px solid rgba(45, 106, 79, 0.32);\n  font-size: 12.5px;\n  color: var(--dsw-alias-label-secondary, #394a41);\n  display: flex;\n  flex-direction: column;\n  gap: 3px;\n}\n\n/* Live 5-Slider Header Utility Control & Popover */\n.xfy-mixer-anchor {\n  position: relative;\n  display: inline-flex;\n  align-items: center;\n}\n\n.xfy-mixer-trigger {\n  appearance: none;\n  display: inline-flex;\n  align-items: center;\n  gap: 6px;\n  height: 28px;\n  padding: 0 10px;\n  border-radius: 999px;\n  border: 1px solid var(--dsw-alias-border-l1, #dfd9cc);\n  background: var(--dsw-alias-bg-layer-2, #faf8f3);\n  color: var(--dsw-alias-label-primary, #1b2d24);\n  font-size: 12px;\n  font-weight: 500;\n  cursor: pointer;\n  transition: border-color 0.14s ease, background 0.14s ease;\n}\n\n.xfy-mixer-trigger:hover {\n  border-color: #2d6a4f;\n}\n\n.xfy-mixer-dot {\n  width: 7px;\n  height: 7px;\n  border-radius: 999px;\n  background: var(--xfy-dot-color, #2d6a4f);\n  flex: none;\n}\n\n.xfy-mixer-popover {\n  position: absolute;\n  top: calc(100% + 8px);\n  right: 0;\n  z-index: 1200;\n  width: 304px;\n  padding: 14px 15px;\n  border-radius: 14px;\n  border: 1px solid var(--dsw-alias-border-l1, #ded8ca);\n  background: var(--dsw-alias-bg-layer-1, #fffdf9);\n  color: var(--dsw-alias-label-primary, #1b2d24);\n  box-shadow: 0 12px 32px rgba(20, 34, 27, 0.14);\n  display: flex;\n  flex-direction: column;\n  gap: 11px;\n}\n\n.xfy-mixer-head {\n  display: flex;\n  align-items: center;\n  justify-content: space-between;\n  gap: 8px;\n}\n\n.xfy-mixer-title {\n  font-size: 12.5px;\n  font-weight: 600;\n}\n\n.xfy-mixer-budget {\n  font-family: ui-monospace, \"SFMono-Regular\", \"Cascadia Mono\", Consolas, monospace;\n  font-variant-numeric: tabular-nums;\n  font-size: 11px;\n  color: var(--dsw-alias-label-tertiary, #6d7c74);\n}\n\n.xfy-preset-grid {\n  display: flex;\n  flex-wrap: wrap;\n  gap: 5px;\n}\n\n.xfy-preset-btn {\n  appearance: none;\n  border: 1px solid var(--dsw-alias-border-l1, #dfd9cc);\n  background: transparent;\n  color: var(--dsw-alias-label-secondary, #47584f);\n  border-radius: 999px;\n  padding: 3px 9px;\n  font-size: 11.5px;\n  cursor: pointer;\n}\n\n.xfy-preset-btn[data-active=\"true\"] {\n  background: #2d6a4f;\n  border-color: #2d6a4f;\n  color: #faf9f4;\n}\n\n.xfy-slider-list {\n  display: flex;\n  flex-direction: column;\n  gap: 7px;\n}\n\n.xfy-slider-row {\n  display: grid;\n  grid-template-columns: 76px 1fr 38px;\n  align-items: center;\n  gap: 8px;\n  font-size: 11.5px;\n}\n\n.xfy-slider-label {\n  color: var(--dsw-alias-label-secondary, #47584f);\n  white-space: nowrap;\n}\n\n.xfy-slider-input {\n  width: 100%;\n  accent-color: #2d6a4f;\n  cursor: pointer;\n}\n\n.xfy-slider-val {\n  text-align: right;\n  font-family: ui-monospace, \"SFMono-Regular\", \"Cascadia Mono\", Consolas, monospace;\n  font-variant-numeric: tabular-nums;\n  font-size: 11px;\n  color: var(--dsw-alias-label-tertiary, #6d7c74);\n}\n\n.xfy-mixer-foot {\n  display: flex;\n  align-items: center;\n  justify-content: space-between;\n  padding-top: 6px;\n  border-top: 1px solid var(--dsw-alias-border-l2, #ebe6da);\n  font-size: 11px;\n  color: var(--dsw-alias-label-tertiary, #6d7c74);\n}\n\n.xfy-link-btn {\n  appearance: none;\n  border: none;\n  background: none;\n  padding: 0;\n  color: #2d6a4f;\n  font-size: 11px;\n  cursor: pointer;\n}";

    function ensureStylesMounted() {
      if (typeof document === "undefined") return;
      if (document.querySelector("style[data-plugin-css=" + JSON.stringify(STYLE_TAG_ID) + "]") !== null) {
        return;
      }
      const tag = document.createElement("style");
      tag.dataset.plugin = "@xiaofeiyu/dsh-plugin";
      tag.dataset.pluginCss = STYLE_TAG_ID;
      tag.textContent = STYLE_CSS;
      document.head.appendChild(tag);
    }

    const NS = "xiaofeiyu";
    const zh = {
      "bubbles.rowTitle": "发送对话气泡",
      "bubbles.preparing": "正在组织口语气泡",
      "code.rowTitle": "发送代码卡片",
      "code.preparing": "正在整理代码卡片",
      "code.copy": "复制",
      "code.copied": "已复制",
      "mixer.label": "五滑条混音",
    };
    const en = {
      "bubbles.rowTitle": "Spoken Bubbles",
      "bubbles.preparing": "Composing chat bubbles",
      "code.rowTitle": "Code Card",
      "code.preparing": "Preparing code card",
      "code.copy": "Copy",
      "code.copied": "Copied",
      "mixer.label": "5-Slider Mixer",
    };

    const FLAVOR_META = {
      pure: { label: "元气日常", accent: "#2d6a4f" },
      neko: { label: "甜系软萌", accent: "#c87d62" },
      tease: { label: "知性姐姐", accent: "#8e5258" },
      brat: { label: "傲娇毒舌", accent: "#be8b32" },
      fragile: { label: "安静慢热", accent: "#587a82" },
      mix: { label: "双味混调", accent: "#2d6a4f" },
    };

    const PRESET_OPTIONS = [
      { id: "pure", label: "元气日常", values: [1.0, 0.0, 0.0, 0.0, 0.0], flavor: "pure" },
      { id: "neko_sweet", label: "微甜软萌", values: [1.0, 0.65, 0.0, 0.0, 0.0], flavor: "neko" },
      { id: "tease_neko", label: "知性甜混", values: [0.95, 0.4, 0.7, 0.0, 0.0], flavor: "tease" },
      { id: "brat_spice", label: "傲娇毒舌", values: [1.0, 0.0, 0.0, 0.85, 0.0], flavor: "brat" },
      { id: "fragile_tipsy", label: "安静轻柔", values: [0.95, 0.0, 0.0, 0.0, 0.6], flavor: "fragile" },
    ];

    const SLIDER_FIELDS = [
      { key: "xiaofeiyu", idx: 0, label: "s0 元气日常" },
      { key: "neko_clingy", idx: 1, label: "c1 甜系软萌" },
      { key: "older_tease", idx: 2, label: "c2 知性姐姐" },
      { key: "brat_mesugaki", idx: 3, label: "c3 傲娇毒舌" },
      { key: "fragile_porcelain", idx: 4, label: "c4 安静慢热" },
    ];

    function parseJsonObject(raw) {
      if (raw && typeof raw === "object" && !Array.isArray(raw)) return raw;
      if (typeof raw === "string" && raw.trim() !== "") {
        try {
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
        } catch {}
      }
      return null;
    }

    function parseBubbleArgs(raw) {
      const obj = parseJsonObject(raw);
      if (!obj) return { bubbles: [], flavor: "pure", weights: null };
      const bubbles = Array.isArray(obj.bubbles)
        ? obj.bubbles.map((x) => String(x ?? "").trim()).filter((x) => x.length > 0)
        : [];
      const flavor = typeof obj.flavor === "string" && obj.flavor in FLAVOR_META ? obj.flavor : "pure";
      const weights = Array.isArray(obj.weights) && obj.weights.length === 5
        ? obj.weights.map((n) => Number(n) || 0)
        : null;
      return { bubbles, flavor, weights };
    }

    function parseCodeArgs(raw) {
      const obj = parseJsonObject(raw);
      if (!obj) return { language: "text", content: "", title: "" };
      return {
        language: typeof obj.language === "string" && obj.language.trim() ? obj.language.trim() : "text",
        content: typeof obj.content === "string" ? obj.content.replace(/\r\n/g, "\n") : "",
        title: typeof obj.title === "string" ? obj.title.trim() : "",
      };
    }

    function resolveBlockArgsRaw(block) {
      if (!block || typeof block !== "object") return "";
      if ("kind" in block) return block.call?.argsRaw ?? block.call?.arguments ?? "";
      return block.argsRaw ?? block.arguments ?? "";
    }

    function isBlockRunning(block) {
      return Boolean(block && typeof block === "object" && !("kind" in block));
    }

    function BubbleBurstCard({ bubbles, flavor, weights }) {
      if (!bubbles || bubbles.length === 0) return null;
      const meta = FLAVOR_META[flavor] || FLAVOR_META.pure;
      const weightText = weights
        ? weights.map((n) => Number(n).toFixed(2)).join(" / ")
        : null;

      return (0, jsxRuntime.jsxs)("div", {
        className: "xfy-burst-row",
        style: { "--xfy-accent": meta.accent },
        "data-xfy-bubbles": bubbles.length,
        children: [
          (0, jsxRuntime.jsx)("span", {
            className: "xfy-avatar-mark",
            "aria-hidden": true,
            children: "鱼",
          }),
          (0, jsxRuntime.jsxs)("div", {
            className: "xfy-bubble-stack",
            children: [
              bubbles.map((line, idx) => {
                const pos =
                  bubbles.length === 1
                    ? "single"
                    : idx === 0
                      ? "first"
                      : idx === bubbles.length - 1
                        ? "last"
                        : "middle";
                return (0, jsxRuntime.jsx)(
                  "div",
                  {
                    className: "xfy-bubble-pill",
                    "data-pos": pos,
                    style: { animationDelay: `${Math.min(idx * 55, 220)}ms` },
                    children: line,
                  },
                  idx
                );
              }),
              (0, jsxRuntime.jsxs)("div", {
                className: "xfy-burst-footer",
                children: [
                  (0, jsxRuntime.jsx)("span", {
                    className: "xfy-flavor-chip",
                    children: meta.label,
                  }),
                  weightText !== null
                    ? (0, jsxRuntime.jsx)("span", {
                        className: "xfy-weights-mono",
                        title: "s0 / c1 / c2 / c3 / c4",
                        children: weightText,
                      })
                    : null,
                ],
              }),
            ],
          }),
        ],
      });
    }

    function CodeArtifactCard({ language, content, title, copyLabel, copiedLabel }) {
      const [copied, setCopied] = (0, react.useState)(false);
      if (!content) return null;
      const lines = content.split("\n");
      const isDiff = language === "diff" || language === "patch";

      const handleCopy = async () => {
        try {
          if (typeof navigator !== "undefined" && navigator.clipboard) {
            await navigator.clipboard.writeText(content);
            setCopied(true);
            setTimeout(() => setCopied(false), 1400);
          }
        } catch {}
      };

      return (0, jsxRuntime.jsxs)("div", {
        className: "xfy-code-card",
        "data-xfy-code": language,
        children: [
          (0, jsxRuntime.jsxs)("div", {
            className: "xfy-code-header",
            children: [
              (0, jsxRuntime.jsx)("span", {
                className: "xfy-code-lang",
                children: title ? `${title} · ${language}` : `${language} · ${lines.length} 行`,
              }),
              (0, jsxRuntime.jsx)("button", {
                type: "button",
                className: "xfy-code-copy-btn",
                onClick: handleCopy,
                children: copied ? copiedLabel : copyLabel,
              }),
            ],
          }),
          (0, jsxRuntime.jsx)("pre", {
            className: "xfy-code-pre",
            children: (0, jsxRuntime.jsx)("code", {
              children: isDiff
                ? lines.map((ln, idx) =>
                    (0, jsxRuntime.jsx)(
                      "span",
                      {
                        className: ln.startsWith("+")
                          ? "xfy-diff-add"
                          : ln.startsWith("-")
                            ? "xfy-diff-del"
                            : void 0,
                        children: idx < lines.length - 1 ? `${ln}\n` : ln,
                      },
                      idx
                    )
                  )
                : content,
            }),
          }),
        ],
      });
    }

    /* Turn-level event projection so completed turns always show IM bubbles & code cards */
    const xiaofeiyuEventDefinition = {
      kind: "xiaofeiyu",
      match: (event) => {
        if (event.type === "turn/start") {
          return { id: String(event.data.turn), role: "start" };
        }
        if (
          event.type === "tool/call" &&
          (event.data.name === "xiaofeiyu_send_bubbles" ||
            event.data.name === "send_bubbles" ||
            event.data.name === "xiaofeiyu_send_code")
        ) {
          return { id: String(event.data.turn), role: "update" };
        }
        if (event.type === "tool/result") {
          return { id: String(event.data.turn), role: "update" };
        }
        return null;
      },
      start: (_context, match) => ({
        turn: match.event.data.turn,
        pending: new Map(),
        items: [],
      }),
      update: (context, match) => {
        if (match.event.type === "tool/call") {
          const name = match.event.data.name;
          const callId = String(match.event.data.callId);
          const nextPending = new Map(context.state.pending);
          if (name === "xiaofeiyu_send_code") {
            nextPending.set(callId, { kind: "code", callId, ...parseCodeArgs(match.event.data.arguments) });
          } else {
            nextPending.set(callId, { kind: "bubbles", callId, ...parseBubbleArgs(match.event.data.arguments) });
          }
          return { ...context.state, pending: nextPending };
        }
        if (match.event.type === "tool/result") {
          if (match.event.data.message?.isError === true) return context.state;
          const callId = String(match.event.data.message?.source?.callId ?? "");
          const entry = context.state.pending.get(callId);
          if (entry === void 0) return context.state;
          const nextPending = new Map(context.state.pending);
          nextPending.delete(callId);
          return {
            ...context.state,
            pending: nextPending,
            items: [...context.state.items, { ...entry, seq: match.event.seq }],
          };
        }
        return context.state;
      },
      buildLocationData: (context, scope, previous) => {
        if (scope !== "turn" || context.state === void 0) return null;
        if (
          previous?.kind === "turn" &&
          previous.turn === context.state.turn &&
          previous.key === "xiaofeiyu" &&
          previous.value.items === context.state.items
        ) {
          return previous;
        }
        return {
          kind: "turn",
          turn: context.state.turn,
          key: "xiaofeiyu",
          value: { items: context.state.items },
        };
      },
    };

    function XiaofeiyuTurnTail(props) {
      const record = props.turn?.data?.get("xiaofeiyu");
      const rawItems = record?.items ?? [];
      const items = rawItems.filter((it) => typeof it.seq !== "number" || it.seq <= props.seq);
      if (items.length === 0) return null;

      return (0, jsxRuntime.jsx)("div", {
        className: "xfy-turn-tail",
        "data-xfy-turn-tail": true,
        children: items.map((item, idx) =>
          item.kind === "code"
            ? (0, jsxRuntime.jsx)(
                CodeArtifactCard,
                {
                  language: item.language,
                  content: item.content,
                  title: item.title,
                  copyLabel: props.t("code.copy"),
                  copiedLabel: props.t("code.copied"),
                },
                item.callId || idx
              )
            : (0, jsxRuntime.jsx)(
                BubbleBurstCard,
                {
                  bubbles: item.bubbles,
                  flavor: item.flavor,
                  weights: item.weights,
                },
                item.callId || idx
              )
        ),
      });
    }

    function XiaofeiyuBubblesToolRow(props) {
      const [open, setOpen] = (0, react.useState)(false);
      if (props.phase === "preparing") {
        return (0, jsxRuntime.jsx)("div", {
          className: "xfy-tool-row",
          "data-tool": "xiaofeiyu_send_bubbles",
          children: (0, jsxRuntime.jsxs)("div", {
            className: "xfy-tool-row-line",
            children: [
              (0, jsxRuntime.jsx)("span", { className: "xfy-tool-dot", "data-running": "true" }),
              (0, jsxRuntime.jsx)("span", { children: props.t("bubbles.preparing") }),
            ],
          }),
        });
      }
      const parsed = parseBubbleArgs(resolveBlockArgsRaw(props.block));
      const running = isBlockRunning(props.block);
      const meta = FLAVOR_META[parsed.flavor] || FLAVOR_META.pure;
      const summary =
        parsed.bubbles.length > 0
          ? `${props.t("bubbles.rowTitle")} · ${parsed.bubbles.length} 条 · ${meta.label}`
          : props.t("bubbles.rowTitle");

      return (0, jsxRuntime.jsxs)("div", {
        className: "xfy-tool-row",
        "data-tool": "xiaofeiyu_send_bubbles",
        children: [
          (0, jsxRuntime.jsxs)("div", {
            className: "xfy-tool-row-line",
            role: "button",
            tabIndex: 0,
            onClick: () => setOpen((v) => !v),
            onKeyDown: (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                setOpen((v) => !v);
              }
            },
            children: [
              (0, jsxRuntime.jsx)("span", {
                className: "xfy-tool-dot",
                style: { background: meta.accent },
                "data-running": running ? "true" : void 0,
              }),
              (0, jsxRuntime.jsx)("span", { children: summary }),
            ],
          }),
          (running || open) && parsed.bubbles.length > 0
            ? (0, jsxRuntime.jsx)("div", {
                className: "xfy-tool-preview",
                children: parsed.bubbles.map((line, i) => (0, jsxRuntime.jsx)("span", { children: line }, i)),
              })
            : null,
        ],
      });
    }

    function XiaofeiyuCodeToolRow(props) {
      const [open, setOpen] = (0, react.useState)(false);
      if (props.phase === "preparing") {
        return (0, jsxRuntime.jsx)("div", {
          className: "xfy-tool-row",
          "data-tool": "xiaofeiyu_send_code",
          children: (0, jsxRuntime.jsxs)("div", {
            className: "xfy-tool-row-line",
            children: [
              (0, jsxRuntime.jsx)("span", { className: "xfy-tool-dot", "data-running": "true" }),
              (0, jsxRuntime.jsx)("span", { children: props.t("code.preparing") }),
            ],
          }),
        });
      }
      const parsed = parseCodeArgs(resolveBlockArgsRaw(props.block));
      const lineCount = parsed.content ? parsed.content.split("\n").length : 0;
      return (0, jsxRuntime.jsxs)("div", {
        className: "xfy-tool-row",
        "data-tool": "xiaofeiyu_send_code",
        children: [
          (0, jsxRuntime.jsxs)("div", {
            className: "xfy-tool-row-line",
            role: "button",
            tabIndex: 0,
            onClick: () => setOpen((v) => !v),
            children: [
              (0, jsxRuntime.jsx)("span", { className: "xfy-tool-dot" }),
              (0, jsxRuntime.jsx)("span", {
                children: `${props.t("code.rowTitle")} · ${parsed.language} (${lineCount} 行)`,
              }),
            ],
          }),
          open && parsed.content
            ? (0, jsxRuntime.jsx)(CodeArtifactCard, {
                language: parsed.language,
                content: parsed.content,
                title: parsed.title,
                copyLabel: props.t("code.copy"),
                copiedLabel: props.t("code.copied"),
              })
            : null,
        ],
      });
    }

    function resolveBridgeSlidersUrl() {
      let port = "18964";
      try {
        const saved = typeof localStorage !== "undefined" ? localStorage.getItem("xfy_dsh_bridge_port") : null;
        if (saved && /^\d+$/.test(saved)) port = saved;
      } catch {}
      return `http://127.0.0.1:${port}/sliders`;
    }

    function inferFlavorFromValues(vals) {
      const c = [vals[1] || 0, vals[2] || 0, vals[3] || 0, vals[4] || 0];
      const maxC = Math.max(...c);
      if (maxC < 0.08) return "pure";
      const activeCount = c.filter((x) => x >= 0.25).length;
      if (activeCount >= 2) return "mix";
      const names = ["neko", "tease", "brat", "fragile"];
      return names[c.indexOf(maxC)] || "pure";
    }

    function XiaofeiyuMixerUtility() {
      const [open, setOpen] = (0, react.useState)(false);
      const [preset, setPreset] = (0, react.useState)("pure");
      const [vals, setVals] = (0, react.useState)([1.0, 0.0, 0.0, 0.0, 0.0]);
      const [overrideActive, setOverrideActive] = (0, react.useState)(false);
      const rootRef = (0, react.useRef)(null);

      const syncFromBridge = (0, react.useCallback)(async () => {
        try {
          const res = await fetch(resolveBridgeSlidersUrl());
          if (!res.ok) return;
          const data = await res.json();
          if (data?.effective?.weights && Array.isArray(data.effective.weights)) {
            setVals(data.effective.weights);
          }
          if (typeof data?.sliderConfig?.preset === "string") {
            setPreset(data.sliderConfig.preset);
          }
          setOverrideActive(Boolean(data?.overrideModelFlavor));
        } catch {}
      }, []);

      (0, react.useEffect)(() => {
        syncFromBridge();
      }, [syncFromBridge]);

      (0, react.useEffect)(() => {
        if (!open || typeof document === "undefined") return;
        const onDocClick = (ev) => {
          if (rootRef.current && !rootRef.current.contains(ev.target)) {
            setOpen(false);
          }
        };
        document.addEventListener("mousedown", onDocClick);
        return () => document.removeEventListener("mousedown", onDocClick);
      }, [open]);

      const pushSliders = async (nextPreset, nextVals, nextOverride) => {
        setPreset(nextPreset);
        setVals(nextVals);
        setOverrideActive(nextOverride);
        try {
          const res = await fetch(resolveBridgeSlidersUrl(), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              preset: nextPreset,
              xiaofeiyu: nextVals[0],
              neko_clingy: nextVals[1],
              older_tease: nextVals[2],
              brat_mesugaki: nextVals[3],
              fragile_porcelain: nextVals[4],
              overrideModelFlavor: nextOverride,
            }),
          });
          if (res.ok) {
            const data = await res.json();
            if (data?.effective?.weights) setVals(data.effective.weights);
          }
        } catch {}
      };

      const flavor = inferFlavorFromValues(vals);
      const meta = FLAVOR_META[flavor] || FLAVOR_META.pure;
      const filterSum = (vals[1] || 0) + (vals[2] || 0) + (vals[3] || 0) + (vals[4] || 0);

      return (0, jsxRuntime.jsxs)("div", {
        className: "xfy-mixer-anchor",
        ref: rootRef,
        children: [
          (0, jsxRuntime.jsxs)("button", {
            type: "button",
            className: "xfy-mixer-trigger",
            "data-xfy-mixer-btn": true,
            onClick: () => {
              const next = !open;
              setOpen(next);
              if (next) syncFromBridge();
            },
            title: "实时调节小肥鱼五滑条人格权重",
            children: [
              (0, jsxRuntime.jsx)("span", {
                className: "xfy-mixer-dot",
                style: { "--xfy-dot-color": meta.accent },
              }),
              (0, jsxRuntime.jsx)("span", {
                children: overrideActive ? `五滑条 · ${meta.label}` : "五滑条混音",
              }),
            ],
          }),
          open
            ? (0, jsxRuntime.jsxs)("div", {
                className: "xfy-mixer-popover",
                role: "dialog",
                "aria-label": "五滑条实时人格混音",
                children: [
                  (0, jsxRuntime.jsxs)("div", {
                    className: "xfy-mixer-head",
                    children: [
                      (0, jsxRuntime.jsx)("span", { className: "xfy-mixer-title", children: "五滑条实时混音" }),
                      (0, jsxRuntime.jsx)("span", {
                        className: "xfy-mixer-budget",
                        children: `滤镜和 ${filterSum.toFixed(2)} / 1.15`,
                      }),
                    ],
                  }),
                  (0, jsxRuntime.jsx)("div", {
                    className: "xfy-preset-grid",
                    children: PRESET_OPTIONS.map((p) =>
                      (0, jsxRuntime.jsx)(
                        "button",
                        {
                          type: "button",
                          className: "xfy-preset-btn",
                          "data-active": overrideActive && preset === p.id ? "true" : "false",
                          onClick: () => pushSliders(p.id, p.values, true),
                          children: p.label,
                        },
                        p.id
                      )
                    ),
                  }),
                  (0, jsxRuntime.jsx)("div", {
                    className: "xfy-slider-list",
                    children: SLIDER_FIELDS.map((field) =>
                      (0, jsxRuntime.jsxs)(
                        "label",
                        {
                          className: "xfy-slider-row",
                          children: [
                            (0, jsxRuntime.jsx)("span", { className: "xfy-slider-label", children: field.label }),
                            (0, jsxRuntime.jsx)("input", {
                              type: "range",
                              className: "xfy-slider-input",
                              min: "0",
                              max: "1.2",
                              step: "0.05",
                              value: String(vals[field.idx] ?? 0),
                              onChange: (ev) => {
                                const next = [...vals];
                                next[field.idx] = Number(ev.target.value);
                                pushSliders("custom", next, true);
                              },
                            }),
                            (0, jsxRuntime.jsx)("span", {
                              className: "xfy-slider-val",
                              children: Number(vals[field.idx] ?? 0).toFixed(2),
                            }),
                          ],
                        },
                        field.key
                      )
                    ),
                  }),
                  (0, jsxRuntime.jsxs)("div", {
                    className: "xfy-mixer-foot",
                    children: [
                      (0, jsxRuntime.jsx)("span", {
                        children: overrideActive ? "当前：滑条实时覆盖已开启" : "当前：跟随底部模型档位",
                      }),
                      overrideActive
                        ? (0, jsxRuntime.jsx)("button", {
                            type: "button",
                            className: "xfy-link-btn",
                            onClick: () => pushSliders("pure", [1.0, 0.0, 0.0, 0.0, 0.0], false),
                            children: "恢复跟随模型",
                          })
                        : null,
                    ],
                  }),
                ],
              })
            : null,
        ],
      });
    }

    const inject = ["slots", "locale", "uiConversation"];

    function apply(ctx) {
      ensureStylesMounted();
      ctx.effect(() => ctx.locale.register(NS, { zh, en }), "ui-xiaofeiyu: dictionaries");
      ctx.uiConversation.events.register(xiaofeiyuEventDefinition);

      ctx.slots.inject("conversation.chat.turnTail", () =>
        ctx.slots.register(
          {
            name: "conversation.chat.turnTail",
            id: "@xiaofeiyu/dsh-plugin",
            order: -50,
            locale: NS,
          },
          XiaofeiyuTurnTail
        )
      );

      ctx.slots.inject("tool.call.toolview", () =>
        ctx.slots.register(
          {
            name: "tool.call.toolview",
            key: "xiaofeiyu_send_bubbles",
            locale: NS,
          },
          XiaofeiyuBubblesToolRow
        )
      );

      ctx.slots.inject("tool.call.toolview", () =>
        ctx.slots.register(
          {
            name: "tool.call.toolview",
            key: "xiaofeiyu_send_code",
            locale: NS,
          },
          XiaofeiyuCodeToolRow
        )
      );

      ctx.slots.inject("conversation.session.header.utilities", () =>
        ctx.slots.register(
          {
            name: "conversation.session.header.utilities",
            id: "xiaofeiyu-five-slider-mixer",
            order: 10,
            label: "五滑条混音",
          },
          XiaofeiyuMixerUtility
        )
      );
    }

    exports.apply = apply;
    exports.inject = inject;
    return module.exports;
  },
});
