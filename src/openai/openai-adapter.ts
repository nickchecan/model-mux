import type { LlmRequest, LlmResponse } from '@google/adk';
import OpenAI from 'openai';
import { BaseAdapter, mapInput, providerError, textResponse } from '../adapter.js';

export class OpenAIAdapter extends BaseAdapter {
  private readonly client: OpenAI;

  constructor(model: string, baseUrl: string, headers: Record<string, string>, apiKey: string) {
    super(model, baseUrl, headers, apiKey);
    this.client = new OpenAI({
      apiKey: apiKey,
      baseURL: baseUrl,
      defaultHeaders: headers,
    });
  }

  async *stream(llmRequest: LlmRequest): AsyncGenerator<LlmResponse, void> {
    let stream;
    try {
      stream = await this.client.responses.create({
        ...this.responseParams(llmRequest),
        stream: true,
      });
    } catch (error) {
      throw providerError('OpenAI', 'stream creation', error);
    }

    for await (const event of stream) {
      if (event.type === 'response.output_text.delta' && event.delta) {
        yield textResponse(event.delta, true);
      }

      if (event.type === 'response.completed') {
        yield textResponse('', false);
        return;
      }
    }
  }

  async generate(llmRequest: LlmRequest): Promise<LlmResponse> {
    let response;
    try {
      response = await this.client.responses.create(this.responseParams(llmRequest));
    } catch (error) {
      throw providerError('OpenAI', 'response creation', error);
    }

    return textResponse(response.output_text);
  }

  private responseParams(llmRequest: LlmRequest) {
    return {
      model: this.model,
      input: mapInput(llmRequest),
      temperature: llmRequest.config?.temperature ?? undefined,
      max_output_tokens: llmRequest.config?.maxOutputTokens ?? undefined,
    };
  }
}
