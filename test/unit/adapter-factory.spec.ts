import Chance from 'chance';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createAdapter } from '../../src/adapter-factory.js';
import { AnthropicAdapter } from '../../src/anthropic/anthropic-adapter.js';
import { OpenAIAdapter } from '../../src/openai/openai-adapter.js';

const mockOpenAIConstructor = vi.fn();
const mockAnthropicConstructor = vi.fn();

vi.mock('../../src/openai/openai-adapter.js', () => {
  class MockOpenAIAdapter {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    constructor(...args: any[]) {
      mockOpenAIConstructor(...args);
    }
  }
  return { OpenAIAdapter: MockOpenAIAdapter };
});

vi.mock('../../src/anthropic/anthropic-adapter.js', () => {
  class MockAnthropicAdapter {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    constructor(...args: any[]) {
      mockAnthropicConstructor(...args);
    }
  }
  return { AnthropicAdapter: MockAnthropicAdapter };
});

describe('createAdapter', () => {
  const chance = new Chance();

  let baseUrl: string;
  let headers: Record<string, string>;
  let apiKey: string;

  beforeEach(() => {
    vi.clearAllMocks();

    baseUrl = chance.url();
    headers = { [chance.word()]: chance.string(), [chance.word()]: chance.string() };
    apiKey = chance.string();
  });

  it("should create an OpenAIAdapter when model starts with 'gpt-'", () => {
    // Arrange
    const model = `gpt-${chance.word()}`;

    // Act
    createAdapter(model, baseUrl, headers, apiKey);

    // Assert
    expect(mockOpenAIConstructor).toHaveBeenCalledOnce();
    expect(mockOpenAIConstructor).toHaveBeenCalledWith(model, baseUrl, headers, apiKey);
  });

  it("should create an AnthropicAdapter when model starts with 'claude-'", () => {
    // Arrange
    const model = `claude-${chance.word()}`;

    // Act
    createAdapter(model, baseUrl, headers, apiKey);

    // Assert
    expect(mockAnthropicConstructor).toHaveBeenCalledOnce();
    expect(mockAnthropicConstructor).toHaveBeenCalledWith(model, baseUrl, headers, apiKey);
  });

  it('should return an instance of OpenAIAdapter for gpt- models', () => {
    // Arrange
    const model = `gpt-${chance.word()}`;

    // Act
    const result = createAdapter(model, baseUrl, headers, apiKey);

    // Assert
    expect(result).toBeInstanceOf(OpenAIAdapter);
  });

  it('should return an instance of AnthropicAdapter for claude- models', () => {
    // Arrange
    const model = `claude-${chance.word()}`;

    // Act
    const result = createAdapter(model, baseUrl, headers, apiKey);

    // Assert
    expect(result).toBeInstanceOf(AnthropicAdapter);
  });

  it('should pass model, baseUrl, headers, and apiKey to OpenAIAdapter', () => {
    // Arrange
    const model = `gpt-${chance.word()}`;
    const specificBaseUrl = chance.url();
    const specificHeaders = { [chance.word()]: chance.string() };
    const specificApiKey = chance.string();

    // Act
    createAdapter(model, specificBaseUrl, specificHeaders, specificApiKey);

    // Assert
    expect(mockOpenAIConstructor).toHaveBeenCalledWith(
      model,
      specificBaseUrl,
      specificHeaders,
      specificApiKey,
    );
  });

  it('should throw an error for unsupported model prefixes', () => {
    // Arrange
    const model = chance.word();

    // Act & Assert
    expect(() => createAdapter(model, baseUrl, headers, apiKey)).toThrow(
      `Unsupported model/provider: ${model}`,
    );
  });

  it('should throw an error when model is an empty string', () => {
    // Act & Assert
    expect(() => createAdapter('', baseUrl, headers, apiKey)).toThrow(
      'Unsupported model/provider: ',
    );
  });

  it('should not instantiate OpenAIAdapter for unsupported models', () => {
    // Arrange
    const model = chance.word();

    // Act
    try {
      createAdapter(model, baseUrl, headers, apiKey);
    } catch {
      // expected
    }

    // Assert
    expect(mockOpenAIConstructor).not.toHaveBeenCalled();
    expect(mockAnthropicConstructor).not.toHaveBeenCalled();
  });

  it("should handle model 'gpt-' with no suffix", () => {
    // Arrange
    const model = 'gpt-';

    // Act
    createAdapter(model, baseUrl, headers, apiKey);

    // Assert
    expect(mockOpenAIConstructor).toHaveBeenCalledOnce();
    expect(mockOpenAIConstructor).toHaveBeenCalledWith(model, baseUrl, headers, apiKey);
  });

  it("should handle model 'claude-' with no suffix", () => {
    // Arrange
    const model = 'claude-';

    // Act
    createAdapter(model, baseUrl, headers, apiKey);

    // Assert
    expect(mockAnthropicConstructor).toHaveBeenCalledOnce();
    expect(mockAnthropicConstructor).toHaveBeenCalledWith(model, baseUrl, headers, apiKey);
  });
});
