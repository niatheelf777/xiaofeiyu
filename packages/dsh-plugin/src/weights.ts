export const BUDGET_LIMIT = 1.15;
export const SLIDER_MIN = 0.0;
export const SLIDER_MAX = 1.25;
export const SLIDER_STEP = 0.05;

export type SliderTuple = readonly [number, number, number, number, number];

export type SliderPresetId =
  | 'custom'
  | 'pure'
  | 'neko_sweet'
  | 'tease_neko'
  | 'brat_spice'
  | 'fragile_tipsy';

export interface SliderInputConfig {
  readonly preset?: SliderPresetId;
  readonly xiaofeiyu?: number;
  readonly neko_clingy?: number;
  readonly older_tease?: number;
  readonly brat_mesugaki?: number;
  readonly fragile_porcelain?: number;
}

export const SLIDER_PRESETS: Readonly<
  Record<Exclude<SliderPresetId, 'custom'>, { readonly label: string; readonly values: SliderTuple }>
> = {
  pure: { label: '元气日常', values: [1.0, 0.0, 0.0, 0.0, 0.0] },
  neko_sweet: { label: '微甜软萌', values: [1.0, 0.65, 0.0, 0.0, 0.0] },
  tease_neko: { label: '知性甜混', values: [0.95, 0.4, 0.7, 0.0, 0.0] },
  brat_spice: { label: '傲娇毒舌', values: [1.0, 0.0, 0.0, 0.85, 0.0] },
  fragile_tipsy: { label: '安静轻柔', values: [0.95, 0.0, 0.0, 0.0, 0.6] },
};

export const MODEL_FLAVOR_OVERRIDES: Readonly<Record<string, SliderTuple>> = {
  'xiaofeiyu-v4:pure': SLIDER_PRESETS.pure.values,
  'xiaofeiyu-v4:neko': [1.0, 0.85, 0.0, 0.0, 0.0],
  'xiaofeiyu-v4:tease': [1.0, 0.0, 0.85, 0.0, 0.0],
  'xiaofeiyu-v4:brat': [1.0, 0.0, 0.0, 0.85, 0.0],
  'xiaofeiyu-v4:fragile': [1.0, 0.0, 0.0, 0.0, 0.85],
  'xiaofeiyu-26b-a4b:pure': SLIDER_PRESETS.pure.values,
  'xiaofeiyu-26b-a4b:neko': [1.0, 0.85, 0.0, 0.0, 0.0],
  'xiaofeiyu-26b-a4b:tease': [1.0, 0.0, 0.85, 0.0, 0.0],
  'xiaofeiyu-26b-a4b:brat': [1.0, 0.0, 0.0, 0.85, 0.0],
  'xiaofeiyu-26b-a4b:fragile': [1.0, 0.0, 0.0, 0.0, 0.85],
  'xiaofeiyu-translator-e4b-v1:pure': SLIDER_PRESETS.pure.values,
  'xiaofeiyu-translator-e4b-v1:neko': [1.0, 0.85, 0.0, 0.0, 0.0],
  'xiaofeiyu-translator-e4b-v1:tease': [1.0, 0.0, 0.85, 0.0, 0.0],
  'xiaofeiyu-translator-e4b-v1:brat': [1.0, 0.0, 0.0, 0.85, 0.0],
  'xiaofeiyu-translator-e4b-v1:fragile': [1.0, 0.0, 0.0, 0.0, 0.85],
};

export const ADVERTISED_MODELS = [
  { id: 'xiaofeiyu-v4', name: '小肥鱼 Unified · 实时五滑条混音', description: '5-slider unified PC2 model' },
  { id: 'xiaofeiyu-v4:pure', name: '小肥鱼 Unified · 元气日常', description: '(1.0, 0, 0, 0, 0)' },
  { id: 'xiaofeiyu-v4:neko', name: '小肥鱼 Unified · 甜系软萌', description: '(1.0, 0.85, 0, 0, 0)' },
  { id: 'xiaofeiyu-v4:tease', name: '小肥鱼 Unified · 知性姐姐', description: '(1.0, 0, 0.85, 0, 0)' },
  { id: 'xiaofeiyu-v4:brat', name: '小肥鱼 Unified · 傲娇毒舌', description: '(1.0, 0, 0, 0.85, 0)' },
  { id: 'xiaofeiyu-v4:fragile', name: '小肥鱼 Unified · 安静慢热', description: '(1.0, 0, 0, 0, 0.85)' },
  {
    id: 'xiaofeiyu-translator-e4b-v1',
    name: '小肥鱼 E4B Voice Proxy · 实时五滑条混音',
    description: 'Zero-thought <0.8s 5-slider translator sidecar',
  },
  { id: 'xiaofeiyu-translator-e4b-v1:neko', name: '小肥鱼 E4B · 甜系软萌', description: '(1.0, 0.85, 0, 0, 0)' },
] as const;

export function round3(value: number): number {
  return Math.round(value * 1000) / 1000;
}

export function clampSlider(value: unknown, fallback = 0): number {
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  return round3(Math.min(SLIDER_MAX, Math.max(SLIDER_MIN, num)));
}

export interface EffectiveWeightsResult {
  readonly raw: SliderTuple;
  readonly weights: SliderTuple;
  readonly filterSum: number;
  readonly budgetLimit: number;
  readonly isClamped: boolean;
  readonly scale: number;
}

export function computeEffectiveWeights(rawSliders: readonly number[]): EffectiveWeightsResult {
  const s0 = clampSlider(rawSliders[0], 1.0);
  const s1 = clampSlider(rawSliders[1], 0.0);
  const s2 = clampSlider(rawSliders[2], 0.0);
  const s3 = clampSlider(rawSliders[3], 0.0);
  const s4 = clampSlider(rawSliders[4], 0.0);
  const raw: SliderTuple = [s0, s1, s2, s3, s4];
  const filterSum = s1 + s2 + s3 + s4;
  const scale = filterSum > BUDGET_LIMIT ? BUDGET_LIMIT / Math.max(filterSum, 1e-6) : 1.0;

  const weights: SliderTuple = [
    round3(s0),
    round3(s1 * scale),
    round3(s2 * scale),
    round3(s3 * scale),
    round3(s4 * scale),
  ];

  return {
    raw,
    weights,
    filterSum: round3(filterSum),
    budgetLimit: BUDGET_LIMIT,
    isClamped: filterSum > BUDGET_LIMIT + 1e-9,
    scale: round3(scale),
  };
}

export function resolveRawSlidersFromConfig(cfg: SliderInputConfig): SliderTuple {
  if (cfg.preset && cfg.preset !== 'custom' && cfg.preset in SLIDER_PRESETS) {
    return SLIDER_PRESETS[cfg.preset].values;
  }
  return [
    clampSlider(cfg.xiaofeiyu, 1.0),
    clampSlider(cfg.neko_clingy, 0.0),
    clampSlider(cfg.older_tease, 0.0),
    clampSlider(cfg.brat_mesugaki, 0.0),
    clampSlider(cfg.fragile_porcelain, 0.0),
  ];
}

export function formatUpstreamModelTag(weights: SliderTuple, baseModel = 'xiaofeiyu-v4'): string {
  const cleanBase = baseModel.split('|')[0]?.split(':')[0]?.trim() || 'xiaofeiyu-v4';
  return `${cleanBase}|${weights.map((w) => Number(w).toFixed(3)).join(',')}`;
}

export function resolveUpstreamModelTag(
  requestedModel: string | undefined,
  sliderConfig: SliderInputConfig,
  runtimeMode: 'unified' | 'translator' = 'unified',
  options: { readonly overrideModelFlavor?: boolean } = {},
): { readonly modelTag: string; readonly weights: SliderTuple } {
  const trimmed = (requestedModel ?? '').trim();
  if (trimmed.includes('|')) {
    const [prefix, spec] = trimmed.split('|', 2);
    const parts = (spec ?? '').split(',').map((p) => Number(p.trim()));
    if (parts.length === 5 && parts.every((n) => Number.isFinite(n) && n >= 0 && n <= 2.0)) {
      const tuple: SliderTuple = [
        round3(parts[0]!),
        round3(parts[1]!),
        round3(parts[2]!),
        round3(parts[3]!),
        round3(parts[4]!),
      ];
      return { modelTag: formatUpstreamModelTag(tuple, prefix || 'xiaofeiyu-v4'), weights: tuple };
    }
  }

  const flavorOverride = options.overrideModelFlavor ? undefined : MODEL_FLAVOR_OVERRIDES[trimmed];
  const raw = flavorOverride ?? resolveRawSlidersFromConfig(sliderConfig);
  const { weights } = computeEffectiveWeights(raw);
  const basePrefix =
    trimmed.startsWith('xiaofeiyu-translator-e4b') || runtimeMode === 'translator'
      ? 'xiaofeiyu-translator-e4b-v1'
      : trimmed.startsWith('xiaofeiyu-26b-a4b')
        ? 'xiaofeiyu-26b-a4b'
        : 'xiaofeiyu-v4';
  return {
    modelTag: formatUpstreamModelTag(weights, basePrefix),
    weights,
  };
}

export type BubbleJoinMode = 'paragraph' | 'newline';

export class BubbleStreamFormatter {
  private readonly mode: BubbleJoinMode;
  private inFence = false;
  private currentLine = '';
  private prevEndedWithBlank = false;

  constructor(mode: BubbleJoinMode = 'paragraph') {
    this.mode = mode;
  }

  push(chunk: string): string {
    if (!chunk || this.mode === 'newline') return chunk;
    const normalized = chunk.replace(/\r\n/g, '\n');
    let out = '';

    for (let i = 0; i < normalized.length; i += 1) {
      const ch = normalized[i]!;
      if (ch !== '\n') {
        this.currentLine += ch;
        this.prevEndedWithBlank = false;
        out += ch;
        continue;
      }

      const trimmedLine = this.currentLine.trim();
      const isFenceDelimiter = trimmedLine.startsWith('```');
      if (!this.inFence && isFenceDelimiter) {
        this.inFence = true;
        out += '\n';
      } else if (this.inFence) {
        if (/^```\s*$/.test(trimmedLine)) {
          this.inFence = false;
          out += '\n\n';
          this.prevEndedWithBlank = true;
        } else {
          out += '\n';
        }
      } else if (this.prevEndedWithBlank || trimmedLine.length === 0) {
        if (!this.prevEndedWithBlank) {
          out += '\n';
          this.prevEndedWithBlank = true;
        }
      } else {
        out += '\n\n';
        this.prevEndedWithBlank = true;
      }
      this.currentLine = '';
    }

    return out;
  }
}

export function formatCompleteBubbleText(text: string, mode: BubbleJoinMode = 'paragraph'): string {
  if (!text || mode === 'newline') return text;
  const fmt = new BubbleStreamFormatter(mode);
  return fmt.push(text).replace(/\n{3,}/g, '\n\n').trimEnd();
}
