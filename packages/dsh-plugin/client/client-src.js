window.__ModuleLoader__.load({
  id: "@xiaofeiyu/dsh-plugin",
  factory: (require) => {
    var module = { exports: {} };
    var exports = module.exports;
    Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });

    const jsxRuntime = require("react/jsx-runtime");
    const react = require("react");

    const STYLE_TAG_ID = "@xiaofeiyu/dsh-plugin/styles.css";
    const STYLE_CSS = __XFY_CLIENT_CSS__;

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
      pure: { label: "纯原版", accent: "#2d6a4f" },
      neko: { label: "粘人猫娘", accent: "#c87d62" },
      tease: { label: "年上御姐", accent: "#8e5258" },
      brat: { label: "嘴硬雌小鬼", accent: "#be8b32" },
      fragile: { label: "低体温易碎", accent: "#587a82" },
      mix: { label: "双味混调", accent: "#2d6a4f" },
    };

    const PRESET_OPTIONS = [
      { id: "pure", label: "纯原版", values: [1.0, 0.0, 0.0, 0.0, 0.0], flavor: "pure" },
      { id: "neko_sweet", label: "微甜猫系", values: [1.0, 0.65, 0.0, 0.0, 0.0], flavor: "neko" },
      { id: "tease_neko", label: "御姐逗猫", values: [0.95, 0.4, 0.7, 0.0, 0.0], flavor: "tease" },
      { id: "brat_spice", label: "嘴硬雌小鬼", values: [1.0, 0.0, 0.0, 0.85, 0.0], flavor: "brat" },
      { id: "fragile_tipsy", label: "易碎微醺", values: [0.95, 0.0, 0.0, 0.0, 0.6], flavor: "fragile" },
    ];

    const SLIDER_FIELDS = [
      { key: "xiaofeiyu", idx: 0, label: "s0 原版底座" },
      { key: "neko_clingy", idx: 1, label: "c1 粘人猫娘" },
      { key: "older_tease", idx: 2, label: "c2 年上御姐" },
      { key: "brat_mesugaki", idx: 3, label: "c3 嘴硬雌小鬼" },
      { key: "fragile_porcelain", idx: 4, label: "c4 低体温易碎" },
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
