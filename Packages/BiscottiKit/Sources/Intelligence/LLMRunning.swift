import Foundation
import LocalLLM

/// The configured backend used for AI enhancements.
public enum LLMProviderConfiguration: Sendable, Equatable {
    case local(model: URL)
    case openAICompatible(baseURL: URL, model: String, apiKey: String?)
}

/// A single loaded-model session that supports N sequential generation calls.
/// Real impl wraps `LLMService.withConnection`; fakes in tests.
public protocol LLMRunning: Sendable {
    func withSession<T: Sendable>(
        model: URL,
        config: EngineConfig,
        _ body: @Sendable (any LLMSession) async throws -> T
    ) async throws -> T

    /// Opens a session for a configured provider.
    ///
    /// The default implementation preserves the original local-model seam for
    /// test doubles and callers that only support the built-in provider.
    func withSession<T: Sendable>(
        provider: LLMProviderConfiguration,
        _ body: @Sendable (any LLMSession) async throws -> T
    ) async throws -> T
}

public extension LLMRunning {
    func withSession<T: Sendable>(
        provider: LLMProviderConfiguration,
        _ body: @Sendable (any LLMSession) async throws -> T
    ) async throws -> T {
        guard case let .local(model) = provider else {
            throw LLMProviderError.unsupportedProvider
        }
        return try await withSession(model: model, config: .modelOnly, body)
    }
}

/// A single generation call within a session. Real impl wraps `LLMConnection`.
public protocol LLMSession: Sendable {
    /// Count the tokens for a message list using the model's tokenizer.
    ///
    /// Same template and tokenization as `generate`, but returns only the
    /// count. No context/KV-cache work, no sampling.
    func countTokens(
        messages: [LLMMessage]
    ) async throws -> Int

    /// Recreate the inference context with a new size.
    ///
    /// Frees the current KV cache and allocates a new one. The model stays
    /// loaded (no re-download/re-read of the GGUF weights). Used by the
    /// caller after `countTokens` to right-size the context before generating.
    func reconfigure(contextSize: Int) async throws

    func generate(
        messages: [LLMMessage], options: GenerationOptions
    ) async throws -> String

    func generateStreaming(
        messages: [LLMMessage], options: GenerationOptions
    ) async -> AsyncThrowingStream<StreamEvent, Error>
}

public enum LLMProviderError: LocalizedError, Sendable, Equatable {
    case unsupportedProvider
    case invalidEndpoint
    case invalidResponse
    case httpStatus(Int, String)
    case serverMessage(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedProvider:
            "The selected AI provider is not supported."
        case .invalidEndpoint:
            "The AI endpoint URL is invalid."
        case .invalidResponse:
            "The AI provider returned an invalid response."
        case let .httpStatus(status, message):
            "The AI provider returned HTTP \(status): \(message)"
        case let .serverMessage(message):
            message
        }
    }
}
