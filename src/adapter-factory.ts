import type { BaseAdapter } from './adapter.js';
import { AnthropicAdapter } from './anthropic/anthropic-adapter.js';
import { OpenAIAdapter } from './openai/openai-adapter.js';

export function createAdapter(
  model: string,
  baseUrl: string,
  headers: Record<string, string>,
  apiKey: string,
): BaseAdapter {
  if (model.startsWith('gpt-')) return new OpenAIAdapter(model, baseUrl, headers, apiKey);
  if (model.startsWith('claude-')) return new AnthropicAdapter(model, baseUrl, headers, apiKey);
  throw new Error(`Unsupported model/provider: ${model}`);
}
