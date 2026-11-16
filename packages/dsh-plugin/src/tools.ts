import type { Context } from '@deepseek-ai/cordis';
import { defineTool } from '@deepseek-ai/dsh-tools';
import { XIAOFEIYU_SEND_BUBBLES_SPEC, XIAOFEIYU_SEND_CODE_SPEC } from './tool-specs.ts';

export const name = 'tool-xiaofeiyu';
export const inject = ['tools'];

export function apply(ctx: Context): void {
  const toolsSvc = (ctx as unknown as { tools: { register: (tool: unknown) => void } }).tools;
  toolsSvc.register(defineTool(XIAOFEIYU_SEND_BUBBLES_SPEC));
  toolsSvc.register(defineTool(XIAOFEIYU_SEND_CODE_SPEC));
}
