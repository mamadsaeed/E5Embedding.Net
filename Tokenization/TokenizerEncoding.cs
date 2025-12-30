namespace E5Embedding.Net.Tokenization;

/// <summary>
/// Represents the encoding result from a tokenizer, containing input IDs, attention masks, and token type IDs.
/// </summary>
public sealed class TokenizerEncoding
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TokenizerEncoding"/> class.
    /// </summary>
    /// <param name="inputIds">The token IDs for the input sequence.</param>
    /// <param name="attentionMask">The attention mask indicating which tokens should be attended to.</param>
    /// <param name="tokenTypeIds">The token type IDs for distinguishing between different sequences (e.g., in pair encoding).</param>
    /// <param name="sequenceLength">The actual length of the sequence before padding.</param>
    public TokenizerEncoding(long[] inputIds, long[] attentionMask, long[] tokenTypeIds, int sequenceLength)
    {
        InputIds = inputIds;
        AttentionMask = attentionMask;
        TokenTypeIds = tokenTypeIds;
        SequenceLength = sequenceLength;
    }

    /// <summary>
    /// Gets the token IDs for the input sequence.
    /// </summary>
    public long[] InputIds { get; }

    /// <summary>
    /// Gets the attention mask indicating which tokens should be attended to (1) and which should be ignored (0).
    /// </summary>
    public long[] AttentionMask { get; }

    /// <summary>
    /// Gets the token type IDs for distinguishing between different sequences in pair encoding scenarios.
    /// </summary>
    public long[] TokenTypeIds { get; }

    /// <summary>
    /// Gets the actual length of the sequence before padding.
    /// </summary>
    public int SequenceLength { get; }
}

