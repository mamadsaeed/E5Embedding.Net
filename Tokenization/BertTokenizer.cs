using System.Globalization;
using System.Text;
using System.Text.Json;

namespace E5Embedding.Net.Tokenization;

/// <summary>
/// BERT-style WordPiece tokenizer implementation for E5 embedding models.
/// Supports tokenization with WordPiece algorithm and pair encoding for sentence pairs.
/// </summary>
public sealed class BertTokenizer
{
    private readonly IReadOnlyDictionary<string, int> _vocab;
    private readonly bool _doLowerCase;
    private readonly bool _tokenizeChineseChars;
    private readonly bool _stripAccents;
    private readonly string _unkToken;
    private readonly string _padToken;
    private readonly string _clsToken;
    private readonly string _sepToken;
    private readonly int _padTokenTypeId;
    private readonly int _clsTokenId;
    private readonly int _sepTokenId;
    private readonly int _padTokenId;
    private readonly int _unkTokenId;
    private readonly string _continuingSubwordPrefix;
    private readonly int _maxInputCharsPerWord;
    private readonly int _maxSequenceLength;

    /// <summary>
    /// Initializes a new instance of the <see cref="BertTokenizer"/> class.
    /// </summary>
    /// <param name="tokenizerConfigFile">Path to the tokenizer configuration file (tokenizer_config.json).</param>
    /// <param name="tokenizerJsonFile">Path to the tokenizer JSON file (tokenizer.json).</param>
    /// <param name="maxSequenceLength">Maximum sequence length for tokenization. If null, uses value from config file.</param>
    /// <exception cref="ArgumentNullException">Thrown when any required parameter is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when any required file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown when configuration is invalid.</exception>
    public BertTokenizer(
        string tokenizerConfigFile,
        string tokenizerJsonFile,
        int? maxSequenceLength = null)
    {
        if (string.IsNullOrWhiteSpace(tokenizerConfigFile))
            throw new ArgumentException("Tokenizer config file path cannot be null or empty.", nameof(tokenizerConfigFile));
        if (string.IsNullOrWhiteSpace(tokenizerJsonFile))
            throw new ArgumentException("Tokenizer JSON file path cannot be null or empty.", nameof(tokenizerJsonFile));

        var tokenizerConfigPath = Path.GetFullPath(tokenizerConfigFile);
        var tokenizerJsonPath = Path.GetFullPath(tokenizerJsonFile);

        if (!File.Exists(tokenizerConfigPath))
        {
            throw new FileNotFoundException($"Tokenizer config not found at '{tokenizerConfigPath}'.", tokenizerConfigPath);
        }

        if (!File.Exists(tokenizerJsonPath))
        {
            throw new FileNotFoundException($"Tokenizer model metadata not found at '{tokenizerJsonPath}'.", tokenizerJsonPath);
        }

        _vocab = LoadVocabFromTokenizerJson(tokenizerJsonPath);

        var config = LoadTokenizerConfig(tokenizerConfigPath);
        var wordPiece = LoadWordPieceConfig(tokenizerJsonPath);

        _doLowerCase = config.DoLowerCase;
        _tokenizeChineseChars = config.TokenizeChineseChars;
        _stripAccents = config.StripAccents ?? config.DoLowerCase;
        _unkToken = config.UnkToken;
        _padToken = config.PadToken;
        _clsToken = config.ClsToken;
        _sepToken = config.SepToken;
        _padTokenTypeId = config.PadTokenTypeId;
        _continuingSubwordPrefix = wordPiece.ContinuingSubwordPrefix;
        _maxInputCharsPerWord = wordPiece.MaxInputCharsPerWord;
        _maxSequenceLength = maxSequenceLength ?? config.ModelMaxLength;

        if (_maxSequenceLength < 2)
        {
            throw new InvalidOperationException("Tokenizer maximum sequence length must be at least 2.");
        }

        _unkTokenId = ResolveTokenId(_unkToken);
        _padTokenId = ResolveTokenId(_padToken);
        _clsTokenId = ResolveTokenId(_clsToken);
        _sepTokenId = ResolveTokenId(_sepToken);
    }

    /// <summary>
    /// Gets the maximum sequence length configured for this tokenizer.
    /// </summary>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <summary>
    /// Encodes a single text into token IDs, attention mask, and token type IDs.
    /// </summary>
    /// <param name="text">The text to encode.</param>
    /// <returns>A <see cref="TokenizerEncoding"/> containing the encoded tokens.</returns>
    public TokenizerEncoding Encode(string text) => EncodeInternal(text, null);

    /// <summary>
    /// Encodes a pair of texts (e.g., query and passage) into token IDs, attention mask, and token type IDs.
    /// </summary>
    /// <param name="text">The first text to encode.</param>
    /// <param name="pairText">The second text to encode.</param>
    /// <returns>A <see cref="TokenizerEncoding"/> containing the encoded tokens with token type IDs distinguishing the two sequences.</returns>
    /// <exception cref="ArgumentNullException">Thrown when pairText is null.</exception>
    public TokenizerEncoding EncodePair(string text, string pairText)
    {
        if (pairText is null)
        {
            throw new ArgumentNullException(nameof(pairText));
        }

        return EncodeInternal(text, pairText);
    }

    private TokenizerEncoding EncodeInternal(string text, string? secondText)
    {
        var primaryTokens = TokenizeToIds(text);
        List<int>? secondaryTokens = null;

        if (secondText is not null)
        {
            secondaryTokens = TokenizeToIds(secondText);
        }

        Truncate(primaryTokens, secondaryTokens);

        var inputIds = new long[_maxSequenceLength];
        var attentionMask = new long[_maxSequenceLength];
        var tokenTypeIds = new long[_maxSequenceLength];

        var index = 0;

        AddToken(_clsTokenId, 0);

        foreach (var tokenId in primaryTokens)
        {
            AddToken(tokenId, 0);
        }

        AddToken(_sepTokenId, 0);

        if (secondaryTokens is not null)
        {
            foreach (var tokenId in secondaryTokens)
            {
                AddToken(tokenId, 1);
            }

            AddToken(_sepTokenId, 1);
        }

        var sequenceLength = index;

        while (index < _maxSequenceLength)
        {
            inputIds[index] = _padTokenId;
            tokenTypeIds[index] = _padTokenTypeId;
            attentionMask[index] = 0;
            index++;
        }

        return new TokenizerEncoding(inputIds, attentionMask, tokenTypeIds, sequenceLength);

        void AddToken(int tokenId, int tokenType)
        {
            if (index >= _maxSequenceLength)
            {
                return;
            }

            inputIds[index] = tokenId;
            tokenTypeIds[index] = secondText is null ? 0 : tokenType;
            attentionMask[index] = 1;
            index++;
        }
    }

    private void Truncate(List<int> primary, List<int>? secondary)
    {
        var reserved = secondary is null ? 2 : 3;
        var available = _maxSequenceLength - reserved;

        if (available <= 0)
        {
            primary.Clear();
            secondary?.Clear();
            return;
        }

        while (primary.Count + (secondary?.Count ?? 0) > available)
        {
            if (secondary is not null && secondary.Count > primary.Count)
            {
                secondary.RemoveAt(secondary.Count - 1);
            }
            else
            {
                primary.RemoveAt(primary.Count - 1);
            }
        }
    }

    private List<int> TokenizeToIds(string? text)
    {
        var wordPieces = new List<int>();
        foreach (var token in RunBasicTokenizer(text ?? string.Empty))
        {
            wordPieces.AddRange(WordPieceTokenize(token));
        }
        return wordPieces;
    }

    private IEnumerable<string> RunBasicTokenizer(string text)
    {
        text = CleanText(text);

        if (_tokenizeChineseChars)
        {
            text = TokenizeChineseChars(text);
        }

        var origTokens = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var splitTokens = new List<string>();

        foreach (var token in origTokens)
        {
            var processedToken = token;
            if (_doLowerCase)
            {
                processedToken = processedToken.ToLowerInvariant();
            }

            if (_stripAccents)
            {
                processedToken = StripAccents(processedToken);
            }

            splitTokens.AddRange(SplitOnPunctuation(processedToken));
        }

        return splitTokens
            .Select(t => t.Trim())
            .Where(t => t.Length > 0)
            .ToList();
    }

    private IEnumerable<int> WordPieceTokenize(string token)
    {
        if (token.Length == 0)
        {
            yield break;
        }

        if (token.Length > _maxInputCharsPerWord)
        {
            yield return _unkTokenId;
            yield break;
        }

        var start = 0;
        var subTokens = new List<int>();

        while (start < token.Length)
        {
            var end = token.Length;
            int? currentId = null;

            while (start < end)
            {
                var substr = token.Substring(start, end - start);
                if (start > 0)
                {
                    substr = _continuingSubwordPrefix + substr;
                }

                if (_vocab.TryGetValue(substr, out var id))
                {
                    currentId = id;
                    break;
                }

                end -= 1;
            }

            if (currentId is null)
            {
                subTokens.Clear();
                subTokens.Add(_unkTokenId);
                break;
            }

            subTokens.Add(currentId.Value);
            start = end;
        }

        foreach (var id in subTokens)
        {
            yield return id;
        }
    }

    private static string CleanText(string text)
    {
        var builder = new StringBuilder(text.Length);

        foreach (var ch in text)
        {
            if (ch == 0 || ch == 0xFFFD || IsControl(ch))
            {
                continue;
            }

            builder.Append(IsWhitespace(ch) ? ' ' : ch);
        }

        return builder.ToString();
    }

    private static string TokenizeChineseChars(string text)
    {
        var builder = new StringBuilder(text.Length * 2);
        foreach (var rune in text.EnumerateRunes())
        {
            if (IsChineseChar(rune.Value))
            {
                builder.Append(' ').Append(rune.ToString()).Append(' ');
            }
            else
            {
                builder.Append(rune.ToString());
            }
        }

        return builder.ToString();
    }

    private static IEnumerable<string> SplitOnPunctuation(string token)
    {
        if (token.Length == 0)
        {
            yield break;
        }

        var sb = new StringBuilder();
        foreach (var ch in token)
        {
            if (IsPunctuation(ch))
            {
                if (sb.Length > 0)
                {
                    yield return sb.ToString();
                    sb.Clear();
                }

                yield return ch.ToString();
            }
            else
            {
                sb.Append(ch);
            }
        }

        if (sb.Length > 0)
        {
            yield return sb.ToString();
        }
    }

    private static bool IsPunctuation(char ch)
    {
        var cp = (int)ch;

        if ((cp >= 33 && cp <= 47) ||
            (cp >= 58 && cp <= 64) ||
            (cp >= 91 && cp <= 96) ||
            (cp >= 123 && cp <= 126))
        {
            return true;
        }

        return char.GetUnicodeCategory(ch) switch
        {
            UnicodeCategory.ConnectorPunctuation => true,
            UnicodeCategory.DashPunctuation => true,
            UnicodeCategory.OpenPunctuation => true,
            UnicodeCategory.ClosePunctuation => true,
            UnicodeCategory.InitialQuotePunctuation => true,
            UnicodeCategory.FinalQuotePunctuation => true,
            UnicodeCategory.OtherPunctuation => true,
            _ => false
        };
    }

    private static bool IsControl(char ch)
    {
        if (ch == '\t' || ch == '\n' || ch == '\r')
        {
            return false;
        }

        return char.IsControl(ch);
    }

    private static bool IsWhitespace(char ch) => char.IsWhiteSpace(ch);

    private static bool IsChineseChar(int codePoint)
    {
        return (codePoint >= 0x4E00 && codePoint <= 0x9FFF) ||
               (codePoint >= 0x3400 && codePoint <= 0x4DBF) ||
               (codePoint >= 0x20000 && codePoint <= 0x2A6DF) ||
               (codePoint >= 0x2A700 && codePoint <= 0x2B73F) ||
               (codePoint >= 0x2B740 && codePoint <= 0x2B81F) ||
               (codePoint >= 0x2B820 && codePoint <= 0x2CEAF) ||
               (codePoint >= 0xF900 && codePoint <= 0xFAFF) ||
               (codePoint >= 0x2F800 && codePoint <= 0x2FA1F);
    }

    private static string StripAccents(string text)
    {
        var normalized = text.Normalize(NormalizationForm.FormD);
        var builder = new StringBuilder(normalized.Length);

        foreach (var ch in normalized)
        {
            if (CharUnicodeInfo.GetUnicodeCategory(ch) != UnicodeCategory.NonSpacingMark)
            {
                builder.Append(ch);
            }
        }

        return builder.ToString().Normalize(NormalizationForm.FormC);
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

        if (vocabElement.ValueKind == JsonValueKind.Object)
        {
            var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var entry in vocabElement.EnumerateObject())
            {
                if (entry.Value.ValueKind == JsonValueKind.Number && entry.Value.TryGetInt32(out var id))
                {
                    vocab[entry.Name] = id;
                }
            }

            return vocab;
        }

        if (vocabElement.ValueKind == JsonValueKind.Array)
        {
            var vocab = new Dictionary<string, int>(capacity: vocabElement.GetArrayLength(), comparer: StringComparer.Ordinal);
            var index = 0;

            foreach (var entry in vocabElement.EnumerateArray())
            {
                if (entry.ValueKind == JsonValueKind.Array && entry.GetArrayLength() >= 1)
                {
                    var token = entry[0].GetString();
                    if (!string.IsNullOrEmpty(token))
                    {
                        vocab[token] = index;
                    }
                }

                index++;
            }

            return vocab;
        }

        throw new InvalidOperationException($"Unsupported tokenizer.json vocab format: {vocabElement.ValueKind}");
    }

    private static TokenizerConfig LoadTokenizerConfig(string configPath)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(configPath));
        var root = document.RootElement;

        return new TokenizerConfig
        {
            DoLowerCase = root.TryGetProperty("do_lower_case", out var lowerElement) && lowerElement.GetBoolean(),
            TokenizeChineseChars = root.TryGetProperty("tokenize_chinese_chars", out var chineseElement) && chineseElement.GetBoolean(),
            StripAccents = root.TryGetProperty("strip_accents", out var stripElement) && stripElement.ValueKind != JsonValueKind.Null
                ? stripElement.GetBoolean()
                : null,
            UnkToken = root.GetProperty("unk_token").GetString() ?? "[UNK]",
            PadToken = root.GetProperty("pad_token").GetString() ?? "[PAD]",
            ClsToken = root.GetProperty("cls_token").GetString() ?? "[CLS]",
            SepToken = root.GetProperty("sep_token").GetString() ?? "[SEP]",
            PadTokenTypeId = root.TryGetProperty("pad_token_type_id", out var padTypeElement) ? padTypeElement.GetInt32() : 0,
            ModelMaxLength = root.TryGetProperty("model_max_length", out var maxElement) ? maxElement.GetInt32() : 512
        };
    }

    private static WordPieceConfig LoadWordPieceConfig(string tokenizerJsonPath)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(tokenizerJsonPath));
        var model = document.RootElement.GetProperty("model");

        return new WordPieceConfig
        {
            ContinuingSubwordPrefix = model.TryGetProperty("continuing_subword_prefix", out var prefixElement)
                ? prefixElement.GetString() ?? "##"
                : "##",
            MaxInputCharsPerWord = model.TryGetProperty("max_input_chars_per_word", out var maxElement)
                ? maxElement.GetInt32()
                : 100
        };
    }

    private sealed class TokenizerConfig
    {
        public required bool DoLowerCase { get; init; }
        public required bool TokenizeChineseChars { get; init; }
        public bool? StripAccents { get; init; }
        public required string UnkToken { get; init; }
        public required string PadToken { get; init; }
        public required string ClsToken { get; init; }
        public required string SepToken { get; init; }
        public required int PadTokenTypeId { get; init; }
        public required int ModelMaxLength { get; init; }
    }

    private sealed class WordPieceConfig
    {
        public required string ContinuingSubwordPrefix { get; init; }
        public required int MaxInputCharsPerWord { get; init; }
    }
}

