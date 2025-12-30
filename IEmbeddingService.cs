namespace E5Embedding.Net;

/// <summary>
/// Service interface for generating text embeddings using E5 models.
/// </summary>
public interface IEmbeddingService
{
    /// <summary>
    /// Generates an embedding vector for a single text input.
    /// </summary>
    /// <param name="text">The input text to embed.</param>
    /// <param name="cancellationToken">Cancellation token to cancel the operation.</param>
    /// <returns>A normalized embedding vector as an array of floats.</returns>
    Task<float[]> EmbedAsync(string text, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates embedding vectors for multiple text inputs in batch.
    /// </summary>
    /// <param name="texts">The collection of input texts to embed.</param>
    /// <param name="cancellationToken">Cancellation token to cancel the operation.</param>
    /// <returns>An array of normalized embedding vectors, one for each input text.</returns>
    Task<float[][]> EmbedBatchAsync(IEnumerable<string> texts, CancellationToken cancellationToken = default);
}

