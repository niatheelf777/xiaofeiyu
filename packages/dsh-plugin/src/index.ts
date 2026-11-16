import type { Context, Volatile } from '@deepseek-ai/cordis';
import z from '@deepseek-ai/schemastery';
import { startLocalOpenAiBridge, type RunningBridgeServer } from './http-bridge.ts';
import type { FetchLike, ResolvedUpstreamConnection, XiaofeiyuRuntimeMode } from './upstream.ts';
import {
  SLIDER_MAX,
  SLIDER_MIN,
  SLIDER_STEP,
  resolveUpstreamModelTag,
  type BubbleJoinMode,
  type SliderInputConfig,
  type SliderPresetId,
} from './weights.ts';

export const name = 'xiaofeiyu';
export const inject = ['credentials'];

export interface Config {
  readonly runtimeMode: Volatile<XiaofeiyuRuntimeMode> | XiaofeiyuRuntimeMode;
  readonly preset: Volatile<SliderPresetId> | SliderPresetId;
  readonly xiaofeiyu: Volatile<number> | number;
  readonly neko_clingy: Volatile<number> | number;
  readonly older_tease: Volatile<number> | number;
  readonly brat_mesugaki: Volatile<number> | number;
  readonly fragile_porcelain: Volatile<number> | number;
  readonly upstreamBaseUrl: Volatile<string> | string;
  readonly translatorProxyUrl: Volatile<string> | string;
  readonly apiKeyRef: Volatile<string> | string;
  readonly bubbleJoinMode: Volatile<BubbleJoinMode> | BubbleJoinMode;
  readonly bridgeEnabled: boolean;
  readonly bridgeHost: string;
  readonly bridgePort: number;
}

const sliderField = (defaultValue: number, description: string) =>
  z.number().min(SLIDER_MIN).max(SLIDER_MAX).step(SLIDER_STEP).default(defaultValue).description(description).volatile();

export const Config = z.object({
  runtimeMode: z
    .union(['unified', 'translator'] as const)
    .default('unified')
    .description('运行模式：unified 直连小肥鱼 26B/12B 网关；translator 走 E4B 语音翻译代理')
    .volatile(),
  preset: z
    .union(['custom', 'pure', 'neko_sweet', 'tease_neko', 'brat_spice', 'fragile_tipsy'] as const)
    .default('custom')
    .description('风格预设档位（选 custom 时由下方 5 根滑条自由混音）')
    .volatile(),
  xiaofeiyu: sliderField(1.0, '基底滑条：原版小肥鱼 (s0)'),
  neko_clingy: sliderField(0.0, '滤镜滑条 1：粘人猫娘 (c1)'),
  older_tease: sliderField(0.0, '滤镜滑条 2：年上御姐 (c2)'),
  brat_mesugaki: sliderField(0.0, '滤镜滑条 3：嘴硬雌小鬼 (c3)'),
  fragile_porcelain: sliderField(0.0, '滤镜滑条 4：低体温易碎感 (c4)'),
  upstreamBaseUrl: z.string().default('http://127.0.0.1:8081/v1').description('Unified 网关地址').volatile(),
  translatorProxyUrl: z.string().default('http://127.0.0.1:18088/v1').description('E4B 语音翻译代理地址').volatile(),
  apiKeyRef: z.string().default('XFY_API_KEY').description('可选上游凭据引用名').volatile(),
  bubbleJoinMode: z.union(['paragraph', 'newline'] as const).default('paragraph').volatile(),
  bridgeEnabled: z.boolean().default(true),
  bridgeHost: z.string().default('127.0.0.1'),
  bridgePort: z.number().min(0).max(65535).default(18964),
});

function readVal<T>(field: T | Volatile<T> | undefined, fallback: T): T {
  if (field !== null && (typeof field === 'object' || typeof field === 'function') && 'get' in field) {
    const val = (field as Volatile<T>).get();
    return (val === undefined ? fallback : val) as T;
  }
  return (field as T | undefined) ?? fallback;
}

export async function apply(
  ctx: Context,
  config: Partial<Config> = {},
  overrides: { readonly fetchImpl?: FetchLike; readonly env?: Readonly<Record<string, string | undefined>> } = {},
): Promise<void> {
  const env = overrides.env ?? process.env;

  let liveSliderOverride: SliderInputConfig | null = null;
  let overrideModelFlavor = false;

  const getSliderConfig = (): SliderInputConfig =>
    liveSliderOverride ?? {
      preset: readVal(config.preset, 'custom'),
      xiaofeiyu: readVal(config.xiaofeiyu, 1.0),
      neko_clingy: readVal(config.neko_clingy, 0.0),
      older_tease: readVal(config.older_tease, 0.0),
      brat_mesugaki: readVal(config.brat_mesugaki, 0.0),
      fragile_porcelain: readVal(config.fragile_porcelain, 0.0),
    };

  const getUpstreamConnection = async (): Promise<ResolvedUpstreamConnection> => ({
    mode: readVal(config.runtimeMode, 'unified'),
    upstreamBaseUrl: readVal(config.upstreamBaseUrl, env.XFY_UPSTREAM_URL ?? 'http://127.0.0.1:8081/v1'),
    translatorProxyUrl: readVal(config.translatorProxyUrl, env.XFY_TRANSLATOR_URL ?? 'http://127.0.0.1:18088/v1'),
    upstreamApiKey: String(env.XFY_API_KEY ?? ''),
  });

  const stateProvider = {
    getSliderConfig,
    setSliderConfig: (next: SliderInputConfig, overrideFlag: boolean) => {
      liveSliderOverride = next;
      overrideModelFlavor = overrideFlag;
    },
    getOverrideModelFlavor: () => overrideModelFlavor,
    getUpstreamConnection,
    getBubbleJoinMode: () => readVal(config.bubbleJoinMode, 'paragraph'),
    fetchImpl: overrides.fetchImpl,
  };

  let bridgeServer: RunningBridgeServer | undefined;
  if (config.bridgeEnabled ?? true) {
    bridgeServer = await startLocalOpenAiBridge({
      host: config.bridgeHost ?? '127.0.0.1',
      port: config.bridgePort ?? 18964,
      state: stateProvider,
    });
  }

  ctx.provide('xiaofeiyu', {
    ...(bridgeServer ? { bridgeBaseUrl: bridgeServer.baseUrl, bridgePort: bridgeServer.port } : {}),
    currentModelTag: () => resolveUpstreamModelTag(undefined, getSliderConfig(), readVal(config.runtimeMode, 'unified')).modelTag,
  });

  ctx.effect(() => async () => {
    if (bridgeServer) {
      await bridgeServer.close();
      bridgeServer = undefined;
    }
  });
}
