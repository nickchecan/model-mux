import type { BaseLlmConnection, LlmRequest, LlmResponse } from '@google/adk';
import Chance from 'chance';
import { describe, it, expect } from 'vitest';
import { BaseAdapter, mapInput } from '../../src/adapter.js';

// Concrete subclass to test the abstract BaseAdapter
class TestAdapter extends BaseAdapter {
  constructor(model: string, baseUrl: string, headers: Record<string, string>, apiKey: string) {
    super(model, baseUrl, headers, apiKey);
  }

  // Expose protected members for testing
  get exposedModel() {
    return this.model;
  }
  get exposedBaseUrl() {
    return this.baseUrl;
  }
  get exposedHeaders() {
    return this.headers;
  }
  get exposedApiKey() {
    return this.apiKey;
  }

  async *stream(_llmRequest: LlmRequest): AsyncGenerator<LlmResponse, void> {
    // stub
  }
  async generate(_llmRequest: LlmRequest): Promise<LlmResponse> {
    return {} as LlmResponse;
  }
  async connect(_llmRequest: LlmRequest): Promise<BaseLlmConnection> {
    return {} as BaseLlmConnection;
  }
}

describe('BaseAdapter', () => {
  const chance = new Chance();

  let model: string;
  let baseUrl: string;
  let headers: Record<string, string>;
  let apiKey: string;
  let adapter: TestAdapter;

  beforeEach(() => {
    model = chance.word();
    baseUrl = chance.url();
    headers = { [chance.word()]: chance.string(), [chance.word()]: chance.string() };
    apiKey = chance.string();
    adapter = new TestAdapter(model, baseUrl, headers, apiKey);
  });

  describe('constructor', () => {
    it('should store model', () => {
      // Assert
      expect(adapter.exposedModel).toBe(model);
    });

    it('should store baseUrl', () => {
      // Assert
      expect(adapter.exposedBaseUrl).toBe(baseUrl);
    });

    it('should store headers', () => {
      // Assert
      expect(adapter.exposedHeaders).toBe(headers);
    });

    it('should store apiKey', () => {
      // Assert
      expect(adapter.exposedApiKey).toBe(apiKey);
    });
  });

  describe('mapInput', () => {
    it('should extract text from contents with parts', () => {
      // Arrange
      const text1 = chance.sentence();
      const text2 = chance.sentence();
      const req = {
        contents: [{ parts: [{ text: text1 }] }, { parts: [{ text: text2 }] }],
      } as unknown as LlmRequest;

      // Act
      const result = mapInput(req);

      // Assert
      expect(result).toBe(text1 + text2);
    });

    it('should handle contents with multiple parts in a single content', () => {
      // Arrange
      const text1 = chance.sentence();
      const text2 = chance.sentence();
      const req = {
        contents: [{ parts: [{ text: text1 }, { text: text2 }] }],
      } as unknown as LlmRequest;

      // Act
      const result = mapInput(req);

      // Assert
      expect(result).toBe(text1 + text2);
    });

    it('should handle parts with missing text by using empty string', () => {
      // Arrange
      const text1 = chance.sentence();
      const req = {
        contents: [{ parts: [{ text: text1 }, { notText: chance.word() }] }],
      } as unknown as LlmRequest;

      // Act
      const result = mapInput(req);

      // Assert
      expect(result).toBe(text1);
    });

    it('should handle contents with missing parts by flatMapping to empty', () => {
      // Arrange
      const text1 = chance.sentence();
      const req = {
        contents: [{ parts: [{ text: text1 }] }, {}],
      } as unknown as LlmRequest;

      // Act
      const result = mapInput(req);

      // Assert
      expect(result).toBe(text1);
    });

    it('should return empty string when contents has no text parts', () => {
      // Arrange
      const req = {
        contents: [{ parts: [{ notText: chance.word() }] }],
      } as unknown as LlmRequest;

      // Act
      const result = mapInput(req);

      // Assert
      expect(result).toBe('');
    });

    it('should return empty string when contents is empty', () => {
      // Arrange
      const req = {
        contents: [],
      } as unknown as LlmRequest;

      // Act
      const result = mapInput(req);

      // Assert
      expect(result).toBe('');
    });
  });
});
