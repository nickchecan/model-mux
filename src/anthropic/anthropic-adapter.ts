import Anthropic from '@anthropic-ai/sdk';
import type { LlmRequest, LlmResponse } from '@google/adk';
import { BaseAdapter, mapInput, providerError, textResponse } from '../adapter.js';

export class AnthropicAdapter extends BaseAdapter {
  private readonly client: Anthropic;

  constructor(model: string, baseUrl: string, headers: Record<string, string>, apiKey: string) {
    super(model);
    this.client = new Anthropic({
      apiKey,
      baseURL: baseUrl,
      defaultHeaders: headers,
    });
  }

  async *stream(llmRequest: LlmRequest): AsyncGenerator<LlmResponse, void> {
    const stream = this.client.messages.stream(this.messageParams(llmRequest));

    try {
      for await (const event of stream) {
        if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
          yield textResponse(event.delta.text, true);
        }
      }

      yield textResponse('', false);
    } catch (error) {
      throw providerError('Anthropic', 'stream creation', error);
    }
  }

  async generate(llmRequest: LlmRequest): Promise<LlmResponse> {
    let response;
    try {
      response = await this.client.messages.create(this.messageParams(llmRequest));
    } catch (error) {
      throw providerError('Anthropic', 'response creation', error);
    }

    const text = response.content
      .filter((block) => block.type === 'text')
      .map((block) => block.text)
      .join('');

    return textResponse(text);
  }

  private messageParams(llmRequest: LlmRequest) {
    return {
      model: this.model,
      max_tokens: llmRequest.config?.maxOutputTokens ?? 1024,
      messages: [{ role: 'user' as const, content: mapInput(llmRequest) }],
      temperature: llmRequest.config?.temperature ?? undefined,
    };
  }
}
