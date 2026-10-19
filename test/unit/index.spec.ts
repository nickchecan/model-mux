import type { BaseLlmConnection, LlmRequest, LlmResponse } from '@google/adk';
import Chance from 'chance';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createAdapter } from '../../src/adapter-factory.js';
import type { BaseAdapter } from '../../src/adapter.js';
import { ModelMux } from '../../src/index.js';

vi.mock('../../src/adapter-factory.js', () => ({
  createAdapter: vi.fn(),
}));

describe('ModelMux', () => {
  const chance = new Chance();
  const fakeAdapter = {
    generate: vi.fn(),
    stream: vi.fn(),
    connect: vi.fn(),
  } as unknown as BaseAdapter;

  let options: {
    model: string;
    baseUrl: string;
    headers: Record<string, string>;
    apiKey: string;
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(createAdapter).mockReturnValue(fakeAdapter);

    options = {
      model: chance.word(),
      baseUrl: chance.url(),
      headers: { [chance.word()]: chance.string(), [chance.word()]: chance.string() },
      apiKey: chance.string(),
    };
  });

  describe('constructor', () => {
    it('should pass model, baseUrl, headers, and apiKey to createAdapter', () => {
      // Act
      new ModelMux(options);

      // Assert
      expect(createAdapter).toHaveBeenCalledOnce();
      expect(createAdapter).toHaveBeenCalledWith(
        options.model,
        options.baseUrl,
        options.headers,
        options.apiKey,
      );
    });
  });

  describe('generateContentAsync', () => {
    it('should yield a single response when stream is false', async () => {
      // Arrange
      const mockResponse = { text: chance.sentence() } as unknown as LlmResponse;
      vi.mocked(fakeAdapter.generate).mockResolvedValue(mockResponse);

      const mux = new ModelMux(options);
      const results: LlmResponse[] = [];

      // Act
      for await (const chunk of mux.generateContentAsync({} as LlmRequest, false)) {
        results.push(chunk);
      }

      // Assert
      expect(fakeAdapter.generate).toHaveBeenCalledOnce();
      expect(results).toEqual([mockResponse]);
    });

    it('should yield a single response when stream is undefined', async () => {
      // Arrange
      const mockResponse = { text: chance.sentence() } as unknown as LlmResponse;
      vi.mocked(fakeAdapter.generate).mockResolvedValue(mockResponse);

      const mux = new ModelMux(options);
      const results: LlmResponse[] = [];

      // Act
      for await (const chunk of mux.generateContentAsync({} as LlmRequest)) {
        results.push(chunk);
      }

      // Assert
      expect(fakeAdapter.generate).toHaveBeenCalledOnce();
      expect(results).toEqual([mockResponse]);
    });

    it('should pass llmRequest to adapter.generate when stream is false', async () => {
      // Arrange
      const llmRequest = { prompt: chance.sentence() } as unknown as LlmRequest;
      const mockResponse = { text: chance.sentence() } as unknown as LlmResponse;
      vi.mocked(fakeAdapter.generate).mockResolvedValue(mockResponse);

      const mux = new ModelMux(options);

      // Act
      for await (const _ of mux.generateContentAsync(llmRequest, false)) {
        // consume
      }

      // Assert
      expect(fakeAdapter.generate).toHaveBeenCalledWith(llmRequest);
    });

    it('should delegate to adapter.stream when stream is true', async () => {
      // Arrange
      const chunk1 = { text: chance.sentence() } as unknown as LlmResponse;
      const chunk2 = { text: chance.sentence() } as unknown as LlmResponse;

      async function* fakeStream(): AsyncGenerator<LlmResponse, void> {
        yield chunk1;
        yield chunk2;
      }

      vi.mocked(fakeAdapter.stream).mockReturnValue(fakeStream());

      const mux = new ModelMux(options);
      const results: LlmResponse[] = [];

      // Act
      for await (const chunk of mux.generateContentAsync({} as LlmRequest, true)) {
        results.push(chunk);
      }

      // Assert
      expect(fakeAdapter.stream).toHaveBeenCalledOnce();
      expect(fakeAdapter.generate).not.toHaveBeenCalled();
      expect(results).toEqual([chunk1, chunk2]);
    });

    it('should pass llmRequest to adapter.stream when stream is true', async () => {
      // Arrange
      const llmRequest = { prompt: chance.sentence() } as unknown as LlmRequest;

      async function* fakeStream(): AsyncGenerator<LlmResponse, void> {
        // empty
      }

      vi.mocked(fakeAdapter.stream).mockReturnValue(fakeStream());

      const mux = new ModelMux(options);

      // Act
      for await (const _ of mux.generateContentAsync(llmRequest, true)) {
        // consume
      }

      // Assert
      expect(fakeAdapter.stream).toHaveBeenCalledWith(llmRequest);
    });

    it('should yield no results when stream returns nothing', async () => {
      // Arrange
      async function* emptyStream(): AsyncGenerator<LlmResponse, void> {
        // yields nothing
      }

      vi.mocked(fakeAdapter.stream).mockReturnValue(emptyStream());

      const mux = new ModelMux(options);
      const results: LlmResponse[] = [];

      // Act
      for await (const chunk of mux.generateContentAsync({} as LlmRequest, true)) {
        results.push(chunk);
      }

      // Assert
      expect(results).toEqual([]);
    });

    it('should propagate errors from adapter.generate', async () => {
      // Arrange
      const errorMessage = chance.sentence();
      vi.mocked(fakeAdapter.generate).mockRejectedValue(new Error(errorMessage));

      const mux = new ModelMux(options);

      // Act & Assert
      await expect(async () => {
        for await (const _ of mux.generateContentAsync({} as LlmRequest, false)) {
          // consume
        }
      }).rejects.toThrow(errorMessage);
    });

    it('should propagate errors from adapter.stream', async () => {
      // Arrange
      const errorMessage = chance.sentence();

      // eslint-disable-next-line require-yield
      async function* errorStream(): AsyncGenerator<LlmResponse, void> {
        throw new Error(errorMessage);
      }

      vi.mocked(fakeAdapter.stream).mockReturnValue(errorStream());

      const mux = new ModelMux(options);

      // Act & Assert
      await expect(async () => {
        for await (const _ of mux.generateContentAsync({} as LlmRequest, true)) {
          // consume
        }
      }).rejects.toThrow(errorMessage);
    });

    it('should not call adapter.stream when stream is false', async () => {
      // Arrange
      const mockResponse = { text: chance.sentence() } as unknown as LlmResponse;
      vi.mocked(fakeAdapter.generate).mockResolvedValue(mockResponse);

      const mux = new ModelMux(options);

      // Act
      for await (const _ of mux.generateContentAsync({} as LlmRequest, false)) {
        // consume
      }

      // Assert
      expect(fakeAdapter.stream).not.toHaveBeenCalled();
    });
  });

  describe('connect', () => {
    it('should delegate to adapter.connect and return the connection', async () => {
      // Arrange
      const llmRequest = { prompt: chance.sentence() } as unknown as LlmRequest;
      const mockConnection = { id: chance.guid() } as unknown as BaseLlmConnection;
      vi.mocked(fakeAdapter.connect).mockResolvedValue(mockConnection);

      const mux = new ModelMux(options);

      // Act
      const result = await mux.connect(llmRequest);

      // Assert
      expect(fakeAdapter.connect).toHaveBeenCalledOnce();
      expect(fakeAdapter.connect).toHaveBeenCalledWith(llmRequest);
      expect(result).toBe(mockConnection);
    });

    it('should propagate errors from adapter.connect', async () => {
      // Arrange
      const errorMessage = chance.sentence();
      const llmRequest = { prompt: chance.sentence() } as unknown as LlmRequest;
      vi.mocked(fakeAdapter.connect).mockRejectedValue(new Error(errorMessage));

      const mux = new ModelMux(options);

      // Act & Assert
      await expect(mux.connect(llmRequest)).rejects.toThrow(errorMessage);
    });
  });
});
