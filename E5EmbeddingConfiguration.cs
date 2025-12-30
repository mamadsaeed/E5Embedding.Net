namespace E5Embedding.Net;

/// <summary>
/// Configuration class for E5 embedding service.
/// </summary>
public class E5EmbeddingConfiguration
{
    /// <summary>
    /// Gets or sets the expected dimension of the embedding vectors.
    /// Default is 1024.
    /// </summary>
    public int Dimension { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the batch size for processing multiple texts.
    /// Default is 16.
    /// </summary>
    public int BatchSize { get; set; } = 16;

    /// <summary>
    /// Gets or sets the path to the ONNX model file.
    /// </summary>
    public string OnnxModelPath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the maximum sequence length for tokenization.
    /// </summary>
    public int MaxSequenceLength { get; set; }

    /// <summary>
    /// Gets or sets the path to the tokenizer configuration file (tokenizer_config.json).
    /// Default is "tokenizer_config.json".
    /// </summary>
    public string TokenizerConfigFile { get; set; } = "tokenizer_config.json";

    /// <summary>
    /// Gets or sets the path to the SentencePiece model file (sentencepiece.bpe.model).
    /// Required for SentencePiece tokenizer.
    /// Default is "sentencepiece.bpe.model".
    /// </summary>
    public string SentencePieceModelFile { get; set; } = "sentencepiece.bpe.model";

    /// <summary>
    /// Gets or sets the path to the tokenizer JSON file (tokenizer.json).
    /// Default is "tokenizer.json".
    /// </summary>
    public string TokenizerJsonFile { get; set; } = "tokenizer.json";
}

