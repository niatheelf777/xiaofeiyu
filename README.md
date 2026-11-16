# 小肥鱼 (Xiaofeiyu) · Split-Phase 五滑条口语外挂与气泡对话引擎

[English README](README_EN.md) · [在线试玩 (Live Demo)](https://xfy.tospeak.top)

小肥鱼是一套面向 Gemma 4 的**低侵入式口语声带与五滑条人格混音方案**：推理思考、代码生成与工具调用全程以零 LoRA（`LoRA = 0.0`）原生运行，仅在输出微信/IM 对话气泡的 `<|"|>` 区间实时注入轻量化人格声带权重，兼具真人短气泡节奏与 100% 满血智商。

在线试玩站：[https://xfy.tospeak.top](https://xfy.tospeak.top)

---

## 3 分钟上手：三种开箱即用姿势

从 Hugging Face 下载 [`niatheelf777/xiaofeiyu-12b-v4`](https://huggingface.co/niatheelf777/xiaofeiyu-12b-v4) 的 5 枚轻量卡带至 `loras/12b/`，并将 Gemma 4 GGUF 底模放入 `models/`（依赖仅需 `pip install tokenizers`）：

### 姿势一：一行命令启动内置二维性格球网页（零 Node 依赖）

`pc2_split_phase_server.py` 已内置完整纸墨风二维性格球盘前端，`launch_local.py` 可自动拉取官方预编译 `llama-server` 并一键拉起前后端：

```bash
# 12 GB 消费级显卡推荐（如 RTX 3060 / 4070，Q4_K_M 仅占约 9.1 GiB 显存）
python server/launch_local.py --preset 12b-q4 --auto-download-llama --open

# 16 GB+ 显卡或统一内存运行 Q8_0 满血精度（26B MoE 高吞吐可换 --preset 26b-a4b）
python server/launch_local.py --preset 12b --auto-download-llama --open
```

启动后浏览器自动打开 `http://127.0.0.1:8081/`（同时暴露 OpenAI 兼容接口 `http://127.0.0.1:8081/v1`）。如需单独调试前端静态资源，也可运行 `node web/serve_local.mjs --api http://127.0.0.1:8081`。

![内置二维性格球盘本地网页实测：多气泡流式上屏与可复制代码卡](assets/showcase/local-webui-turnkey.png)

### 姿势二：接入 DeepSeek Harness (`DSH`) 编程助手（双模式支持）

同一个本地 `:8081` 网关同时兼容免插件直连与原生气泡 UI 插件两种用法：

| 模式 | 适用场景 | 接入方式 | 对话与工具体验 |
| :--- | :--- | :--- | :--- |
| **模式 1：标准免插件直连** | 原生 DSH、Cursor、OpenWebUI 或任意 OpenAI 客户端 | Base URL 填 `http://127.0.0.1:8081/v1`，模型名填 `xiaofeiyu-v4\|1.0,0.8,0,0,0` | 自动将多气泡与代码块展开为标准 Markdown 正文，支持完整多跳工具调用 |
| **模式 2：`@xiaofeiyu/dsh-plugin`** | 希望在 DSH WebUI 内获得原生短气泡、代码卡与实时调音台 | 运行下方一行安装命令（支持 `--uninstall` 字节级无损还原） | 注册无冲突原生工具 `xiaofeiyu_send_bubbles` / `xiaofeiyu_send_code`，主对话流渲染立体短气泡与一键复制代码卡，顶栏内置**五滑条实时混音面板** |

```bash
# 为 DSH Web Profile 一键挂载小肥鱼预设、原生气泡工具与顶栏五滑条混音 UI
node packages/dsh-plugin/scripts/install-dsh.mjs --apply --profile web --upstream-base http://127.0.0.1:8081/v1

# 随时无损卸载还原原配置
node packages/dsh-plugin/scripts/install-dsh.mjs --uninstall --profile web
```

![DSH 模式 2 实测：文件读取工具调用、立体微信短气泡、可复制代码卡与顶栏五滑条实时混音面板](assets/showcase/dsh-mode2-mixer.png)

### 姿势三：OpenAI 兼容接口与 `E4B` 通用口语翻译代理

不想替换主力推理模型时，可保留 DeepSeek / Claude / GPT 作为上游大脑负责推理、检索与写代码，本地仅用 `4B` 声带模型将自然语言回复实时改写为短气泡口吻（中位延迟 `302 ms`，代码块与工具调用 `0%` 篡改透传）：

```bash
# 1. 启动本地 E4B llama-server（默认端口 18085）
python server/launch_local.py --preset e4b --llama-only --auto-download-llama

# 2. 启动零依赖 OpenAI 口语翻译代理（默认监听 127.0.0.1:18088/v1）
UPSTREAM_BASE_URL=https://api.deepseek.com/v1 \
UPSTREAM_API_KEY=your_key \
TRANSLATOR_URL=http://127.0.0.1:18085 \
node packages/openai-voice-proxy/bin/xiaofeiyu-voice-proxy.mjs
```

![消费级显卡选型对照：12GB / 16GB / 24GB / Apple Silicon](assets/showcase/consumer-gpu-guide.png)

---

## 为什么做小肥鱼：实测对话对比

传统提示词驱动的聊天模型往往带有明显的“AI 味”——动辄 30–50 字以上的说教长段落、舞台动作括号（`（气鼓鼓）`、`*sighs*`）、Emoji 堆砌与句尾机械反问；而全量微调又会让模型的逻辑、写代码与多轮记忆能力严重退化。

![同题实测对比：告别 AI 味说教与舞台腔，还原真人微信短气泡](assets/showcase/chat-vs-rlhf-slop.png)

---

## 五滑条无重启实时混音

基于共享日常底座（`s0`）与四枚正交解耦的性格卡带（`c1..c4`），无需切换模型或重启服务，单个 OpenAI 兼容请求即可通过 `model="xiaofeiyu-v4|w0,w1,w2,w3,w4"` 在连续权重空间自由混调：

| 槽位 | 分支 ID | 中文标签 | 角色定位 | 预算约束 |
| :---: | :--- | :--- | :--- | :--- |
| `0` | `s0_xiaofeiyu_backbone` | 小肥鱼日常 | 核心真人短气泡口语底座 | `w0 = 1.0` |
| `1` | `c1_neko_clingy` | 粘人猫娘 | 高依恋、软撒娇、拉袖口 | `w1 + w2 + w3 + w4 <= 1.15` |
| `2` | `c2_older_tease` | 年上御姐 | 游刃有余的轻熟逗弄与反客为主 | 同上 |
| `3` | `c3_brat_mesugaki` | 恶劣雌小鬼 | 嘴硬傲娇、嘴上不饶人 | 同上 |
| `4` | `c4_fragile_porcelain` | 易碎依赖 | 轻声细语、低体温脆感陪伴 | 同上 |

![五滑条实时无重启混音：同一句话五种鲜明性格](assets/showcase/five-persona-grid.png)

---

## 核心机制：Split-Phase 脑声解耦

```text
用户输入 ─▶ Phase A: Thinking & Tool/Code (LoRA = 0.0)
               ├─ 思考 <channel>thought ... <channel|>   ──▶ 原生满血智商，零退化
               ├─ calc_python / send_code_or_doc        ──▶ 精确代码与数值计算
               └─ call:send_message{bubbles:[
                     <|"|>  ────▶ Phase B: 仅在气泡引号内激活五滑条 LoRA (w0..w4)
                     气泡文本
                     <|"|>  ────▶ 闭合引号立即归零 LoRA = 0.0
                  ]}
```

![性格只管说话，写代码与算账 100% 不降智](assets/showcase/code-without-lobotomy.png)

| 评测维度 | 传统提示词 RP | 全网络常开微调 (`v3`) | **`Xiaofeiyu-12B-v4` (Split-Phase)** |
| :--- | :---: | :---: | :---: |
| **日常与推拉数理/古诗盲测 (42 轮)** | `12 / 12` (`100%`) | `7 / 12` (`58.3%`) | **`12 / 12 (100%)`** |
| **工程与算法独立探针 (`GEN1..5`)** | `5 / 5` (`100%`) | `2 / 5` (`40.0%`) | **`5 / 5 (100%)`** |
| **跨轮记忆与多跳工具语法 (`TC1 + TC2`)** | `7 / 7` (`100%`) | `1 / 4` (`25.0%`) | **`7 / 7 (100%)`** |
| **舞台括号 / Emoji / Markdown 污染率** | `88.1%` | `9.5%` | **`0.0%`（42 轮门禁全零污染）** |

---

## 仓库结构

| 目录 | 说明 | 运行/测试 |
| :--- | :--- | :--- |
| [`server/`](server/) | `llama.cpp` Split-Phase 网关、内置本地二维性格球 Web UI 与一键启动器 | `python server/test_server.py -v` |
| [`packages/dsh-plugin/`](packages/dsh-plugin/) | DeepSeek Harness (`DSH`) 原生气泡工具、顶栏五滑条前端与可逆安装器 | `npm --prefix packages/dsh-plugin test` |
| [`packages/openai-voice-proxy/`](packages/openai-voice-proxy/) | 零依赖 Node 通用口语翻译代理（保护代码块与工具调用原样透传） | `npm --prefix packages/openai-voice-proxy test` |
| [`web/`](web/) | 纸墨风二维性格球盘 Web UI、独立本地转发脚本与 Cloudflare/Vercel Edge 代理 | `npm --prefix web test` |
| [`train/`](train/) | Split-Phase SFT、残差人格卡带、防偷渡 DPO、在线 DPO 清洗与 GGUF 导出 | `pytest train/test_train.py -v` |
| [`hf-cards/`](hf-cards/) | HuggingFace 模型卡（`12b-v4`, `26b-a4b`, `translator-e4b`） | `python scripts/publish_hf_release.py --dry-run` |
| [`scripts/`](scripts/) | 开源发布隐私门禁与 HuggingFace 5 槽权重校验脚本 | `python scripts/audit_zero_leak.py` |

## 开源许可

本仓库代码基于 **Apache-2.0** 开源（见 [`LICENSE`](LICENSE)）。基础模型与衍生权重的使用须同时遵守 Google [Gemma Terms of Use](https://ai.google.dev/gemma/terms)。
