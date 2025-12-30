using Microsoft.ML.Tokenizers;
using System.Text.Json;

namespace E5Embedding.Net.Tokenization;

/// <summary>
/// SentencePiece tokenizer implementation for E5 embedding models.
/// Supports tokenization using SentencePiece BPE models.
/// </summary>
public sealed class SentencePieceTokenizer
{
    private readonly Tokenizer _sentencePieceTokenizer;
    private readonly IReadOnlyDictionary<string, int> _vocab;
    private readonly int _bosTokenId;
    private readonly int _eosTokenId;
    private readonly int _padTokenId;
    private readonly int _unkTokenId;
    private readonly int _maxSequenceLength;
    private readonly string _bosToken;
    private readonly string _eosToken;
    private readonly string _padToken;

    /// <summary>
    /// Initializes a new instance of the <see cref="SentencePieceTokenizer"/> class.
    /// </summary>
    /// <param name="sentencePieceModelFile">Path to the SentencePiece model file (.model).</param>
    /// <param name="tokenizerConfigFile">Path to the tokenizer configuration file (tokenizer_config.json).</param>
    /// <param name="tokenizerJsonFile">Path to the tokenizer JSON file (tokenizer.json).</param>
    /// <param name="maxSequenceLength">Maximum sequence length for tokenization.</param>
    /// <exception cref="ArgumentNullException">Thrown when any required parameter is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when any required file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown when configuration is invalid.</exception>
    public SentencePieceTokenizer(
        string sentencePieceModelFile,
        string tokenizerConfigFile,
        string tokenizerJsonFile,
        int maxSequenceLength)
    {
        if (string.IsNullOrWhiteSpace(sentencePieceModelFile))
            throw new ArgumentException("SentencePiece model file path cannot be null or empty.", nameof(sentencePieceModelFile));
        if (string.IsNullOrWhiteSpace(tokenizerConfigFile))
            throw new ArgumentException("Tokenizer config file path cannot be null or empty.", nameof(tokenizerConfigFile));
        if (string.IsNullOrWhiteSpace(tokenizerJsonFile))
            throw new ArgumentException("Tokenizer JSON file path cannot be null or empty.", nameof(tokenizerJsonFile));

        var sentencePiecePath = Path.GetFullPath(sentencePieceModelFile);
        var tokenizerConfigPath = Path.GetFullPath(tokenizerConfigFile);
        var tokenizerJsonPath = Path.GetFullPath(tokenizerJsonFile);

        if (!File.Exists(sentencePiecePath))
        {
            throw new FileNotFoundException($"SentencePiece model not found at '{sentencePiecePath}'.", sentencePiecePath);
        }

        if (!File.Exists(tokenizerConfigPath))
        {
            throw new FileNotFoundException($"Tokenizer config not found at '{tokenizerConfigPath}'.", tokenizerConfigPath);
        }

        if (!File.Exists(tokenizerJsonPath))
        {
            throw new FileNotFoundException($"Tokenizer model metadata not found at '{tokenizerJsonPath}'.", tokenizerJsonPath);
        }

        var config = LoadTokenizerConfig(tokenizerConfigPath);
        _maxSequenceLength = maxSequenceLength;

        if (_maxSequenceLength < 2)
        {
            throw new InvalidOperationException("Tokenizer maximum sequence length must be at least 2.");
        }

        _vocab = LoadVocabFromTokenizerJson(tokenizerJsonPath);

        using var stream = File.OpenRead(sentencePiecePath);
        _sentencePieceTokenizer = Microsoft.ML.Tokenizers.SentencePieceTokenizer.Create(stream);

        _bosToken = config.BosToken;
        _eosToken = config.EosToken;
        _padToken = config.PadToken;

        _bosTokenId = ResolveTokenId(config.BosToken);
        _eosTokenId = ResolveTokenId(config.EosToken);
        _padTokenId = ResolveTokenId(config.PadToken);
        _unkTokenId = ResolveTokenId(config.UnkToken);
    }

    /// <summary>
    /// Gets the maximum sequence length configured for this tokenizer.
    /// </summary>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <summary>
    /// Encodes the input text into token IDs, attention mask, and token type IDs.
    /// </summary>
    /// <param name="text">The text to encode.</param>
    /// <returns>A <see cref="TokenizerEncoding"/> containing the encoded tokens.</returns>
    public TokenizerEncoding Encode(string text)
    {
        if (text == null)
        {
            text = string.Empty;
        }

        var encodedTokens = _sentencePieceTokenizer.EncodeToTokens(
            text,
            out _,
            considerPreTokenization: true,
            considerNormalization: true);

        var contentIds = new List<int>(capacity: encodedTokens.Count);
        foreach (var token in encodedTokens)
        {
            var value = token.Value;
            if (string.Equals(value, _bosToken, StringComparison.Ordinal) ||
                string.Equals(value, _eosToken, StringComparison.Ordinal) ||
                string.Equals(value, _padToken, StringComparison.Ordinal))
            {
                continue;
            }

            contentIds.Add(_vocab.TryGetValue(value, out var id) ? id : _unkTokenId);
        }

        var maxContentLength = _maxSequenceLength - 2;
        if (contentIds.Count > maxContentLength)
        {
            contentIds.RemoveRange(maxContentLength, contentIds.Count - maxContentLength);
        }

        var inputIds = new long[_maxSequenceLength];
        var attentionMask = new long[_maxSequenceLength];

        var index = 0;

        inputIds[index] = _bosTokenId;
        attentionMask[index] = 1;
        index++;

        foreach (var tokenId in contentIds)
        {
            inputIds[index] = tokenId;
            attentionMask[index] = 1;
            index++;
        }

        inputIds[index] = _eosTokenId;
        attentionMask[index] = 1;
        index++;

        var sequenceLength = index;

        while (index < _maxSequenceLength)
        {
            inputIds[index] = _padTokenId;
            attentionMask[index] = 0;
            index++;
        }

        var tokenTypeIds = new long[_maxSequenceLength];

        return new TokenizerEncoding(inputIds, attentionMask, tokenTypeIds, sequenceLength);
    }

    private int ResolveTokenId(string token)
    {
        if (_vocab.TryGetValue(token, out var id))
        {
            return id;
        }

        throw new InvalidOperationException($"Tokenizer vocabulary is missing required token '{token}'.");
    }

    private static IReadOnlyDictionary<string, int> LoadVocabFromTokenizerJson(string tokenizerJsonPath)
    {
        using var stream = File.OpenRead(tokenizerJsonPath);
        using var document = JsonDocument.Parse(stream);

        var model = document.RootElement.GetProperty("model");
        var vocabElement = model.GetProperty("vocab");

        var vocab = new Dictionary<string, int>(capacity: vocabElement.GetArrayLength(), comparer: StringComparer.Ordinal);
        var index = 0;

        foreach (var entry in vocabElement.EnumerateArray())
        {
            if (entry.ValueKind != JsonValueKind.Array || entry.GetArrayLength() < 1)
            {
                continue;
            }

            var token = entry[0].GetString();
            if (string.IsNullOrEmpty(token))
            {
                index++;
                continue;
            }

            vocab[token] = index++;
        }

        return vocab;
    }

    private static TokenizerConfig LoadTokenizerConfig(string configPath)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(configPath));
        var root = document.RootElement;

        return new TokenizerConfig
        {
            BosToken = root.GetProperty("bos_token").GetString() ?? "<s>",
            EosToken = root.GetProperty("eos_token").GetString() ?? "</s>",
            PadToken = root.GetProperty("pad_token").GetString() ?? "<pad>",
            UnkToken = root.GetProperty("unk_token").GetString() ?? "<unk>",
            ModelMaxLength = root.TryGetProperty("model_max_length", out var maxElement)
                ? maxElement.GetInt32()
                : 512
        };
    }

    private sealed class TokenizerConfig
    {
        public required string BosToken { get; init; }
        public required string EosToken { get; init; }
        public required string PadToken { get; init; }
        public required string UnkToken { get; init; }
        public required int ModelMaxLength { get; init; }
    }
}

