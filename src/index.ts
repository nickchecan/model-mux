import type { BaseLlmConnection, LlmRequest, LlmResponse } from '@google/adk';
import { BaseLlm } from '@google/adk';
import { createAdapter } from './adapter-factory.js';
import type { BaseAdapter } from './adapter.js';

const DEFAULT_API_KEY = 'sk-000000000000000000000000000000000000000000000000';

export interface ModelMuxOptions {
  model: string;
  baseUrl: string;
  headers?: Record<string, string>;
  apiKey?: string;
}

export class ModelMux extends BaseLlm {
  private readonly adapter: BaseAdapter;

  constructor(options: ModelMuxOptions) {
    super({ model: options.model });
    this.adapter = createAdapter(
      options.model,
      options.baseUrl,
      options.headers ?? {},
      options.apiKey ?? DEFAULT_API_KEY,
    );
  }

  override async *generateContentAsync(
    llmRequest: LlmRequest,
    stream?: boolean,
  ): AsyncGenerator<LlmResponse, void> {
    if (stream) {
      yield* this.adapter.stream(llmRequest);
      return;
    }

    const oneShot = await this.adapter.generate(llmRequest);
    yield oneShot;
  }

  async connect(llmRequest: LlmRequest): Promise<BaseLlmConnection> {
    return this.adapter.connect(llmRequest);
  }
}
