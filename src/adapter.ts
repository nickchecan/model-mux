import type { BaseLlmConnection, LlmRequest, LlmResponse } from '@google/adk';

export function mapInput(req: LlmRequest): string {
  return req.contents
    .flatMap((content) => content.parts ?? [])
    .map((part) => part.text ?? '')
    .join('');
}

export function textResponse(text: string, partial?: boolean): LlmResponse {
  return {
    content: {
      role: 'model',
      parts: [{ text }],
    },
    ...(partial !== undefined ? { partial } : {}),
  };
}

export function providerError(provider: string, action: string, error: unknown): Error {
  const detail = error instanceof Error ? error.message : String(error);
  return new Error(`${provider} ${action} failed: ${detail}`);
}

export abstract class BaseAdapter {
  constructor(
    protected readonly model: string,
    protected readonly baseUrl: string,
    protected readonly headers: Record<string, string>,
    protected readonly apiKey: string,
  ) {}

  abstract stream(llmRequest: LlmRequest): AsyncGenerator<LlmResponse, void>;
  abstract generate(llmRequest: LlmRequest): Promise<LlmResponse>;

  async connect(_llmRequest: LlmRequest): Promise<BaseLlmConnection> {
    // BaseLlm requires connect(). Live sessions are not supported.
    throw new Error('Not implemented');
  }
}
