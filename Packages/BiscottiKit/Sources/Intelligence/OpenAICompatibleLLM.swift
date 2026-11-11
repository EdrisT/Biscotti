import Foundation
import LocalLLM

/// OpenAI-compatible `/chat/completions` client.
///
/// This intentionally lives outside `LocalLLM`: it is a provider transport,
/// not a local inference engine. It works with local servers such as omlx and
/// with compatible remote servers when explicitly configured by the user.
public struct OpenAICompatibleLLMRunner: LLMRunning {
    public init() {}

    public func withSession<T: Sendable>(
        model _: URL,
        config _: EngineConfig,
        _ body: @Sendable (any LLMSession) async throws -> T
    ) async throws -> T {
        throw LLMProviderError.unsupportedProvider
    }

    public func withSession<T: Sendable>(
        provider: LLMProviderConfiguration,
        _ body: @Sendable (any LLMSession) async throws -> T
    ) async throws -> T {
        guard case let .openAICompatible(baseURL, model, apiKey) = provider else {
            throw LLMProviderError.unsupportedProvider
        }
        let session = OpenAICompatibleSession(
            baseURL: baseURL, model: model, apiKey: apiKey
        )
        return try await body(session)
    }
}

private struct OpenAICompatibleSession: LLMSession {
    let baseURL: URL
    let model: String
    let apiKey: String?

    func countTokens(messages: [LLMMessage]) async throws -> Int {
        // OpenAI-compatible servers do not expose one universal tokenizer API.
        // This estimate is only used to reserve output/context budget.
        let characters = messages.reduce(0) { $0 + $1.content.count }
        return max(1, (characters + 3) / 4)
    }

    func reconfigure(contextSize _: Int) async throws {}

    func generate(
        messages: [LLMMessage], options: GenerationOptions
    ) async throws -> String {
        var text = ""
        for try await event in generateStreaming(messages: messages, options: options) {
            if case let .token(piece) = event {
                text += piece
            }
        }
        return text
    }

    func generateStreaming(
        messages: [LLMMessage], options: GenerationOptions
    ) async -> AsyncThrowingStream<StreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let request = try makeRequest(messages: messages, options: options)
                    let (bytes, response) = try await URLSession.shared.bytes(for: request)
                    guard let http = response as? HTTPURLResponse else {
                        throw LLMProviderError.invalidResponse
                    }
                    guard (200..<300).contains(http.statusCode) else {
                        let message = try await collectErrorMessage(from: bytes)
                        throw LLMProviderError.httpStatus(http.statusCode, message)
                    }

                    for try await line in bytes.lines {
                        guard line.hasPrefix("data:") else { continue }
                        let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                        if payload == "[DONE]" { break }
                        guard let data = payload.data(using: .utf8) else { continue }
                        let chunk = try JSONDecoder().decode(
                            ChatCompletionChunk.self, from: data
                        )
                        if let error = chunk.error?.message {
                            throw LLMProviderError.serverMessage(error)
                        }
                        let piece = chunk.choices.first?.delta.content ?? ""
                        if !piece.isEmpty {
                            continuation.yield(.token(piece))
                        }
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func makeRequest(
        messages: [LLMMessage], options: GenerationOptions
    ) throws -> URLRequest {
        let endpoint: URL
        if baseURL.path.hasSuffix("/chat/completions") {
            endpoint = baseURL
        } else {
            endpoint = baseURL.appendingPathComponent("chat/completions")
        }

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let apiKey, !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        let payload = ChatCompletionRequest(
            model: model,
            messages: messages.map { .init(role: $0.role.rawValue, content: $0.content) },
            maxTokens: options.maxTokens,
            temperature: options.temperature,
            stop: options.stopSequences.isEmpty ? nil : options.stopSequences,
            stream: true
        )
        request.httpBody = try JSONEncoder().encode(payload)
        return request
    }

    private func collectErrorMessage(
        from bytes: URLSession.AsyncBytes
    ) async throws -> String {
        var data = Data()
        for try await byte in bytes {
            data.append(byte)
        }
        if let error = try? JSONDecoder().decode(ChatCompletionErrorResponse.self, from: data) {
            return error.error.message
        }
        return String(data: data, encoding: .utf8) ?? "Unknown provider error"
    }

}

private struct ChatCompletionRequest: Encodable {
    struct Message: Encodable {
        let role: String
        let content: String
    }

    let model: String
    let messages: [Message]
    let maxTokens: Int
    let temperature: Float
    let stop: [String]?
    let stream: Bool

    enum CodingKeys: String, CodingKey {
        case model, messages, temperature, stop, stream
        case maxTokens = "max_tokens"
    }
}

private struct ChatCompletionChunk: Decodable {
    struct Choice: Decodable {
        struct Delta: Decodable {
            let content: String?
        }

        let delta: Delta
    }

    struct Error: Decodable {
        let message: String
    }

    let choices: [Choice]
    let error: Error?
}

private struct ChatCompletionErrorResponse: Decodable {
    struct Error: Decodable {
        let message: String
    }

    let error: Error
}
