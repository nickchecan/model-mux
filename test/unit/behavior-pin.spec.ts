import type { LlmRequest, LlmResponse } from '@google/adk';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ModelMux } from '../../src/index.js';

const {
  mockOpenAICreate,
  mockOpenAICtor,
  mockAnthropicCreate,
  mockAnthropicStream,
  mockAnthropicCtor,
} = vi.hoisted(() => ({
  mockOpenAICreate: vi.fn(),
  mockOpenAICtor: vi.fn(),
  mockAnthropicCreate: vi.fn(),
  mockAnthropicStream: vi.fn(),
  mockAnthropicCtor: vi.fn(),
}));

vi.mock('openai', () => {
  class MockOpenAI {
    responses = { create: mockOpenAICreate };
    constructor(...args: unknown[]) {
      mockOpenAICtor(...args);
    }
  }
  return { default: MockOpenAI };
});

vi.mock('@anthropic-ai/sdk', () => {
  class MockAnthropic {
    messages = { create: mockAnthropicCreate, stream: mockAnthropicStream };
    constructor(...args: unknown[]) {
      mockAnthropicCtor(...args);
    }
  }
  return { default: MockAnthropic };
});

const BASE_URL = 'https://example.test/v1';
const DEFAULT_API_KEY = 'sk-000000000000000000000000000000000000000000000000';

const richRequest = {
  contents: [
    {
      role: 'user',
      parts: [{ text: 'Hello' }, { functionCall: { name: 'get_time', args: { city: 'Paris' } } }],
    },
    {
      role: 'model',
      parts: [
        { functionResponse: { name: 'get_time', response: { report: '10:30' } } },
        { text: ' World' },
      ],
    },
    { parts: [{ text: '!' }] },
  ],
  config: {
    temperature: 0,
    maxOutputTokens: 0,
    systemInstruction: 'Be brief',
    tools: [{ functionDeclarations: [{ name: 'get_time' }] }],
  },
} as unknown as LlmRequest;

function mux(
  model: string,
  options?: { headers?: Record<string, string>; apiKey?: string },
): ModelMux {
  return new ModelMux({ model, baseUrl: BASE_URL, ...options });
}

async function collect(source: AsyncGenerator<LlmResponse>): Promise<LlmResponse[]> {
  const chunks: LlmResponse[] = [];
  for await (const chunk of source) chunks.push(chunk);
  return chunks;
}

async function collectUntilError(source: AsyncGenerator<LlmResponse>): Promise<{
  chunks: LlmResponse[];
  message?: string;
}> {
  const chunks: LlmResponse[] = [];
  try {
    for await (const chunk of source) chunks.push(chunk);
    return { chunks };
  } catch (error) {
    return { chunks, message: error instanceof Error ? error.message : String(error) };
  }
}

describe('behavior pin', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('OpenAI', () => {
    it('flattens text parts into one response request and ignores tools, roles, and system instruction', async () => {
      mockOpenAICreate.mockResolvedValue({ output_text: 'yes' });

      const result = await collect(
        mux('gpt-4o', { apiKey: 'sk-test', headers: { 'X-Trace': '1' } }).generateContentAsync(
          richRequest,
          false,
        ),
      );

      expect(mockOpenAICtor).toHaveBeenCalledWith({
        apiKey: 'sk-test',
        baseURL: BASE_URL,
        defaultHeaders: { 'X-Trace': '1' },
      });
      expect(mockOpenAICreate).toHaveBeenCalledWith({
        model: 'gpt-4o',
        input: 'Hello World!',
        temperature: 0,
        max_output_tokens: 0,
      });
      expect(result).toEqual([{ content: { role: 'model', parts: [{ text: 'yes' }] } }]);
    });

    it('sends undefined sampling fields when config is missing and keeps an empty completion', async () => {
      mockOpenAICreate.mockResolvedValue({ output_text: '' });
      const request = { contents: [{ parts: [{ text: 'Hi' }] }] } as unknown as LlmRequest;

      const result = await collect(
        mux('gpt-4o', { apiKey: 'sk-test' }).generateContentAsync(request),
      );

      expect(mockOpenAICreate).toHaveBeenCalledWith({
        model: 'gpt-4o',
        input: 'Hi',
        temperature: undefined,
        max_output_tokens: undefined,
      });
      expect(mockOpenAICreate.mock.calls[0][0]).not.toHaveProperty('stream');
      expect(result).toEqual([{ content: { role: 'model', parts: [{ text: '' }] } }]);
    });

    it('coerces null sampling values to undefined', async () => {
      mockOpenAICreate.mockResolvedValue({ output_text: 'ok' });
      const request = {
        contents: [{ parts: [{ text: null }, { text: 'A' }] }],
        config: { temperature: null, maxOutputTokens: null },
      } as unknown as LlmRequest;

      await collect(mux('gpt-4o', { apiKey: 'sk-test' }).generateContentAsync(request, false));

      expect(mockOpenAICreate).toHaveBeenCalledWith({
        model: 'gpt-4o',
        input: 'A',
        temperature: undefined,
        max_output_tokens: undefined,
      });
    });

    it('wraps response creation failures, including non-Error values', async () => {
      const request = { contents: [{ parts: [{ text: 'Hi' }] }] } as unknown as LlmRequest;
      mockOpenAICreate.mockRejectedValueOnce(new Error('boom'));
      mockOpenAICreate.mockRejectedValueOnce({ status: 401 });

      const client = mux('gpt-4o', { apiKey: 'sk-test' });
      await expect(collect(client.generateContentAsync(request, false))).rejects.toThrow(
        'OpenAI response creation failed: boom',
      );
      await expect(collect(client.generateContentAsync(request, false))).rejects.toThrow(
        'OpenAI response creation failed: [object Object]',
      );
    });

    it('maps text deltas, skips falsy deltas and unrelated events, and stops at response.completed', async () => {
      async function* events() {
        yield { type: 'response.created' };
        yield { type: 'response.output_text.delta', delta: '' };
        yield { type: 'response.output_text.delta', delta: 'A' };
        yield { type: 'response.output_text.delta', delta: ' ' };
        yield { type: 'response.in_progress' };
        yield { type: 'response.completed' };
        yield { type: 'response.output_text.delta', delta: 'ignored' };
      }
      mockOpenAICreate.mockResolvedValue(events());

      const result = await collect(
        mux('gpt-4o', { apiKey: 'sk-test' }).generateContentAsync(richRequest, true),
      );

      expect(mockOpenAICreate).toHaveBeenCalledWith({
        model: 'gpt-4o',
        input: 'Hello World!',
        temperature: 0,
        max_output_tokens: 0,
        stream: true,
      });
      expect(result).toEqual([
        { content: { role: 'model', parts: [{ text: 'A' }] }, partial: true },
        { content: { role: 'model', parts: [{ text: ' ' }] }, partial: true },
        { content: { role: 'model', parts: [{ text: '' }] }, partial: false },
      ]);
    });

    it('wraps stream creation failures and lets later iteration failures through unwrapped', async () => {
      const request = { contents: [{ parts: [{ text: 'Hi' }] }] } as unknown as LlmRequest;
      const client = mux('gpt-4o', { apiKey: 'sk-test' });

      mockOpenAICreate.mockRejectedValueOnce(new Error('setup'));
      await expect(collect(client.generateContentAsync(request, true))).rejects.toThrow(
        'OpenAI stream creation failed: setup',
      );

      async function* events() {
        yield { type: 'response.output_text.delta', delta: 'A' };
        throw new Error('cutoff');
      }
      mockOpenAICreate.mockResolvedValueOnce(events());
      const streamed = await collectUntilError(client.generateContentAsync(request, true));
      expect(streamed.chunks).toEqual([
        { content: { role: 'model', parts: [{ text: 'A' }] }, partial: true },
      ]);
      expect(streamed.message).toBe('cutoff');
    });

    it('throws the base not-implemented error from connect', async () => {
      await expect(mux('gpt-4o', { apiKey: 'sk-test' }).connect(richRequest)).rejects.toThrow(
        'Not implemented',
      );
    });
  });

  describe('Anthropic', () => {
    it('sends one user message, defaults max_tokens, and concatenates text blocks only', async () => {
      mockAnthropicCreate.mockResolvedValue({
        content: [
          { type: 'text', text: 'A' },
          { type: 'tool_use', id: 'tool-1', name: 'get_time', input: {} },
          { type: 'text', text: '' },
          { type: 'text', text: 'B' },
        ],
      });

      const result = await collect(
        mux('claude-sonnet', {
          apiKey: 'sk-ant',
          headers: { 'X-Trace': '1' },
        }).generateContentAsync(richRequest, false),
      );

      expect(mockAnthropicCtor).toHaveBeenCalledWith({
        apiKey: 'sk-ant',
        baseURL: BASE_URL,
        defaultHeaders: { 'X-Trace': '1' },
      });
      expect(mockAnthropicCreate).toHaveBeenCalledWith({
        model: 'claude-sonnet',
        max_tokens: 0,
        messages: [{ role: 'user', content: 'Hello World!' }],
        temperature: 0,
      });
      expect(result).toEqual([{ content: { role: 'model', parts: [{ text: 'AB' }] } }]);
    });

    it('defaults max_tokens to 1024 and passes undefined temperature when config is missing', async () => {
      mockAnthropicCreate.mockResolvedValue({ content: [] });
      const request = { contents: [{ parts: [{ text: 'Hi' }] }] } as unknown as LlmRequest;

      const result = await collect(
        mux('claude-sonnet', { apiKey: 'sk-ant' }).generateContentAsync(request),
      );

      expect(mockAnthropicCreate).toHaveBeenCalledWith({
        model: 'claude-sonnet',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hi' }],
        temperature: undefined,
      });
      expect(result).toEqual([{ content: { role: 'model', parts: [{ text: '' }] } }]);
    });

    it('coerces null temperature to undefined and null maxOutputTokens to the 1024 default', async () => {
      mockAnthropicCreate.mockResolvedValue({ content: [{ type: 'text', text: 'ok' }] });
      const request = {
        contents: [{ parts: [{ text: 'Hi' }] }],
        config: { temperature: null, maxOutputTokens: null },
      } as unknown as LlmRequest;

      await collect(
        mux('claude-sonnet', { apiKey: 'sk-ant' }).generateContentAsync(request, false),
      );

      expect(mockAnthropicCreate).toHaveBeenCalledWith({
        model: 'claude-sonnet',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hi' }],
        temperature: undefined,
      });
    });

    it('wraps response creation failures, including non-Error values', async () => {
      const request = { contents: [{ parts: [{ text: 'Hi' }] }] } as unknown as LlmRequest;
      mockAnthropicCreate.mockRejectedValueOnce('nope');

      await expect(
        collect(mux('claude-sonnet', { apiKey: 'sk-ant' }).generateContentAsync(request, false)),
      ).rejects.toThrow('Anthropic response creation failed: nope');
    });

    it('yields every text delta, including empty text, then a terminal chunk', async () => {
      async function* events() {
        yield { type: 'message_start' };
        yield { type: 'content_block_delta', delta: { type: 'text_delta', text: '' } };
        yield { type: 'content_block_delta', delta: { type: 'text_delta', text: 'A' } };
        yield {
          type: 'content_block_delta',
          delta: { type: 'input_json_delta', partial_json: '{' },
        };
        yield { type: 'content_block_delta', delta: { type: 'text_delta', text: 'B' } };
        yield { type: 'message_stop' };
      }
      mockAnthropicStream.mockReturnValue(events());

      const result = await collect(
        mux('claude-sonnet', { apiKey: 'sk-ant' }).generateContentAsync(richRequest, true),
      );

      expect(mockAnthropicStream).toHaveBeenCalledWith({
        model: 'claude-sonnet',
        max_tokens: 0,
        messages: [{ role: 'user', content: 'Hello World!' }],
        temperature: 0,
      });
      expect(result).toEqual([
        { content: { role: 'model', parts: [{ text: '' }] }, partial: true },
        { content: { role: 'model', parts: [{ text: 'A' }] }, partial: true },
        { content: { role: 'model', parts: [{ text: 'B' }] }, partial: true },
        { content: { role: 'model', parts: [{ text: '' }] }, partial: false },
      ]);
    });

    it('still emits the terminal chunk when the stream is empty', async () => {
      async function* events() {}
      mockAnthropicStream.mockReturnValue(events());
      const request = { contents: [{ parts: [{ text: 'Hi' }] }] } as unknown as LlmRequest;

      const result = await collect(
        mux('claude-sonnet', { apiKey: 'sk-ant' }).generateContentAsync(request, true),
      );

      expect(mockAnthropicStream).toHaveBeenCalledWith({
        model: 'claude-sonnet',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hi' }],
        temperature: undefined,
      });
      expect(result).toEqual([
        { content: { role: 'model', parts: [{ text: '' }] }, partial: false },
      ]);
    });

    it('does not wrap a synchronous messages.stream failure, but wraps iteration failures', async () => {
      const request = { contents: [{ parts: [{ text: 'Hi' }] }] } as unknown as LlmRequest;
      const client = mux('claude-sonnet', { apiKey: 'sk-ant' });

      mockAnthropicStream.mockImplementationOnce(() => {
        throw new Error('bad key');
      });
      const setup = await collectUntilError(client.generateContentAsync(request, true));
      expect(setup.chunks).toEqual([]);
      expect(setup.message).toBe('bad key');

      async function* events() {
        yield { type: 'content_block_delta', delta: { type: 'text_delta', text: 'A' } };
        throw 'cutoff';
      }
      mockAnthropicStream.mockReturnValueOnce(events());
      const streamed = await collectUntilError(client.generateContentAsync(request, true));
      expect(streamed.chunks).toEqual([
        { content: { role: 'model', parts: [{ text: 'A' }] }, partial: true },
      ]);
      expect(streamed.message).toBe('Anthropic stream creation failed: cutoff');
    });

    it('throws the base not-implemented error from connect', async () => {
      await expect(mux('claude-sonnet', { apiKey: 'sk-ant' }).connect(richRequest)).rejects.toThrow(
        'Not implemented',
      );
    });
  });

  describe('routing and defaults', () => {
    it('routes a model by its prefix, including a bare prefix', () => {
      mux('gpt-claude', { apiKey: 'k' });
      mux('gpt-', { apiKey: 'k' });
      expect(mockOpenAICtor).toHaveBeenCalledTimes(2);
      expect(mockAnthropicCtor).not.toHaveBeenCalled();

      vi.clearAllMocks();

      mux('claude-gpt', { apiKey: 'k' });
      mux('claude-', { apiKey: 'k' });
      expect(mockAnthropicCtor).toHaveBeenCalledTimes(2);
      expect(mockOpenAICtor).not.toHaveBeenCalled();
    });

    it('rejects models that do not start with a known prefix', () => {
      for (const model of ['GPT-4o', 'Claude-opus', 'gpt', 'my-gpt-4', 'gemini-2', '']) {
        expect(() => mux(model, { apiKey: 'k' })).toThrow(`Unsupported model/provider: ${model}`);
      }
      expect(mockOpenAICtor).not.toHaveBeenCalled();
      expect(mockAnthropicCtor).not.toHaveBeenCalled();
    });

    it('fills in an empty header map and the sentinel api key, but keeps an empty api key', () => {
      mux('gpt-4o');
      mux('gpt-4o', { apiKey: '' });

      expect(mockOpenAICtor).toHaveBeenNthCalledWith(1, {
        apiKey: DEFAULT_API_KEY,
        baseURL: BASE_URL,
        defaultHeaders: {},
      });
      expect(mockOpenAICtor).toHaveBeenNthCalledWith(2, {
        apiKey: '',
        baseURL: BASE_URL,
        defaultHeaders: {},
      });
    });

    it('exposes the requested model name', () => {
      expect(mux('gpt-4o', { apiKey: 'k' }).model).toBe('gpt-4o');
      expect(mux('claude-sonnet', { apiKey: 'k' }).model).toBe('claude-sonnet');
    });
  });
});
