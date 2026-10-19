import type { LlmRequest, LlmResponse } from '@google/adk';
import Chance from 'chance';
import { describe, it, expect } from 'vitest';
import { BaseAdapter, mapInput } from '../../src/adapter.js';

class TestAdapter extends BaseAdapter {
  get exposedModel() {
    return this.model;
  }

  async *stream(_llmRequest: LlmRequest): AsyncGenerator<LlmResponse, void> {
    // stub
  }
  async generate(_llmRequest: LlmRequest): Promise<LlmResponse> {
    return {} as LlmResponse;
  }
}

describe('BaseAdapter', () => {
  const chance = new Chance();

  it('should store the model name', () => {
    const model = chance.word();
    const adapter = new TestAdapter(model);

    expect(adapter.exposedModel).toBe(model);
  });
});

describe('mapInput', () => {
  const chance = new Chance();

  it('should extract text from contents with parts', () => {
    const text1 = chance.sentence();
    const text2 = chance.sentence();
    const req = {
      contents: [{ parts: [{ text: text1 }] }, { parts: [{ text: text2 }] }],
    } as unknown as LlmRequest;

    expect(mapInput(req)).toBe(text1 + text2);
  });

  it('should handle contents with multiple parts in a single content', () => {
    const text1 = chance.sentence();
    const text2 = chance.sentence();
    const req = {
      contents: [{ parts: [{ text: text1 }, { text: text2 }] }],
    } as unknown as LlmRequest;

    expect(mapInput(req)).toBe(text1 + text2);
  });

  it('should handle parts with missing text by using empty string', () => {
    const text1 = chance.sentence();
    const req = {
      contents: [{ parts: [{ text: text1 }, { notText: chance.word() }] }],
    } as unknown as LlmRequest;

    expect(mapInput(req)).toBe(text1);
  });

  it('should handle contents with missing parts by flatMapping to empty', () => {
    const text1 = chance.sentence();
    const req = {
      contents: [{ parts: [{ text: text1 }] }, {}],
    } as unknown as LlmRequest;

    expect(mapInput(req)).toBe(text1);
  });

  it('should return empty string when contents has no text parts', () => {
    const req = {
      contents: [{ parts: [{ notText: chance.word() }] }],
    } as unknown as LlmRequest;

    expect(mapInput(req)).toBe('');
  });

  it('should return empty string when contents is empty', () => {
    const req = {
      contents: [],
    } as unknown as LlmRequest;

    expect(mapInput(req)).toBe('');
  });
});
