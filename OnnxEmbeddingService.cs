using E5Embedding.Net.Tokenization;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Reflection;
using System.Runtime.InteropServices;

namespace E5Embedding.Net;

/// <summary>
/// High-performance embedding service using ONNX Runtime for E5 models.
/// Supports GPU acceleration (CUDA/DirectML) with automatic CPU fallback.
/// </summary>
public sealed class OnnxEmbeddingService : IEmbeddingService, IDisposable
{
    private readonly InferenceSession _session;
    private readonly SentencePieceTokenizer _tokenizer;
    private readonly E5EmbeddingConfiguration _configuration;
    private readonly ILogger<OnnxEmbeddingService>? _logger;
    private const string LastHiddenStateOutputName = "last_hidden_state";
    private const int DefaultDeviceId = 0;

    /// <summary>
    /// Initializes a new instance of the <see cref="OnnxEmbeddingService"/> class.
    /// </summary>
    /// <param name="configuration">The configuration for the embedding service.</param>
    /// <param name="logger">Optional logger for diagnostic information.</param>
    /// <exception cref="ArgumentNullException">Thrown when configuration is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when configuration is invalid or model dimensions don't match.</exception>
    /// <exception cref="FileNotFoundException">Thrown when model or tokenizer files are not found.</exception>
    public OnnxEmbeddingService(E5EmbeddingConfiguration configuration, ILogger<OnnxEmbeddingService>? logger = null)
    {
        _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        _logger = logger;

        if (string.IsNullOrWhiteSpace(_configuration.OnnxModelPath))
        {
            throw new InvalidOperationException("OnnxModelPath is not configured.");
        }

        if (!File.Exists(_configuration.OnnxModelPath))
        {
            throw new FileNotFoundException($"Embedding model not found: {_configuration.OnnxModelPath}", _configuration.OnnxModelPath);
        }

        var sessionCreate = CreateSessionPreferGpu(_configuration.OnnxModelPath);
        _session = sessionCreate.Session;

        _logger?.LogInformation(
            "ONNX embedding session created. Provider={Provider}, UsedGpu={UsedGpu}, ModelPath={ModelPath}",
            sessionCreate.Provider,
            sessionCreate.UsedGpu,
            _configuration.OnnxModelPath);

        if (sessionCreate.GpuFallbackToCpu && sessionCreate.GpuException is not null)
        {
            _logger?.LogWarning(
                sessionCreate.GpuException,
                "GPU provider '{Provider}' was available but session initialization failed; fell back to CPU.",
                sessionCreate.Provider);
        }

        _tokenizer = new SentencePieceTokenizer(
            _configuration.SentencePieceModelFile,
            _configuration.TokenizerConfigFile,
            _configuration.TokenizerJsonFile,
            _configuration.MaxSequenceLength);

        if (_session.OutputMetadata.TryGetValue(LastHiddenStateOutputName, out var outputMeta))
        {
            var dims = outputMeta.Dimensions;
            if (dims is { Length: 3 } && dims[2] > 0 && _configuration.Dimension > 0 && dims[2] != _configuration.Dimension)
            {
                throw new InvalidOperationException(
                    $"Embedding dimension mismatch. Configured Dimension={_configuration.Dimension}, " +
                    $"but model output '{LastHiddenStateOutputName}' has hidden_size={dims[2]}. " +
                    "Fix the Dimension configuration to match the model.");
            }
        }
    }

    /// <summary>
    /// Generates an embedding vector for a single text input.
    /// </summary>
    /// <param name="text">The input text to embed.</param>
    /// <param name="cancellationToken">Cancellation token to cancel the operation.</param>
    /// <returns>A normalized embedding vector as an array of floats.</returns>
    public async Task<float[]> EmbedAsync(string text, CancellationToken cancellationToken = default)
    {
        var embeddings = await EmbedBatchAsync(new[] { text }, cancellationToken);
        return embeddings[0];
    }

    /// <summary>
    /// Generates embedding vectors for multiple text inputs in batch.
    /// </summary>
    /// <param name="texts">The collection of input texts to embed.</param>
    /// <param name="cancellationToken">Cancellation token to cancel the operation.</param>
    /// <returns>An array of normalized embedding vectors, one for each input text.</returns>
    /// <exception cref="ArgumentNullException">Thrown when texts is null.</exception>
    public async Task<float[][]> EmbedBatchAsync(IEnumerable<string> texts, CancellationToken cancellationToken = default)
    {
        var textList = texts?.ToList() ?? throw new ArgumentNullException(nameof(texts));
        if (textList.Count == 0)
        {
            return Array.Empty<float[]>();
        }

        var results = new List<float[]>(capacity: textList.Count);

        for (var i = 0; i < textList.Count; i += _configuration.BatchSize)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var batch = textList.Skip(i).Take(_configuration.BatchSize).ToList();
            var batchResults = await ProcessBatchAsync(batch, cancellationToken);
            results.AddRange(batchResults);
        }

        return results.ToArray();
    }

    /// <summary>
    /// Releases all resources used by the <see cref="OnnxEmbeddingService"/>.
    /// </summary>
    public void Dispose()
    {
        _session?.Dispose();
    }

    private Task<List<float[]>> ProcessBatchAsync(List<string> batch, CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            var results = new List<float[]>(capacity: batch.Count);

            foreach (var text in batch)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var encoding = _tokenizer.Encode(text ?? string.Empty);

                var inputIds = new DenseTensor<long>(new[] { 1, encoding.InputIds.Length });
                var attentionMask = new DenseTensor<long>(new[] { 1, encoding.AttentionMask.Length });

                for (var i = 0; i < encoding.InputIds.Length; i++)
                {
                    inputIds[0, i] = encoding.InputIds[i];
                    attentionMask[0, i] = encoding.AttentionMask[i];
                }

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
                    NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask)
                };

                using var output = _session.Run(inputs);

                var lastHiddenState = output.FirstOrDefault(o => string.Equals(o.Name, LastHiddenStateOutputName, StringComparison.Ordinal))
                    ?.AsTensor<float>()
                    ?? output.First().AsTensor<float>();

                var vector = MeanPooling(lastHiddenState, encoding.AttentionMask);
                results.Add(vector);
            }

            return results;
        }, cancellationToken);
    }

    private float[] MeanPooling(Tensor<float> embeddings, long[] attentionMask)
    {
        if (embeddings.Dimensions.Length != 3)
        {
            throw new InvalidOperationException($"Unexpected embedding tensor rank {embeddings.Dimensions.Length}. Expected [1, seq_len, hidden].");
        }

        var hiddenSize = embeddings.Dimensions[2];
        if (_configuration.Dimension > 0 && _configuration.Dimension != hiddenSize)
        {
            throw new InvalidOperationException(
                $"Embedding dimension mismatch at runtime. Configured Dimension={_configuration.Dimension}, " +
                $"but model produced hidden_size={hiddenSize}.");
        }

        var result = new float[hiddenSize];
        double maskSum = 0;

        var seqLen = embeddings.Dimensions[1];
        var tokenCount = Math.Min(seqLen, attentionMask.Length);

        for (var token = 0; token < tokenCount; token++)
        {
            if (attentionMask[token] == 0)
            {
                continue;
            }

            maskSum += 1;

            for (var dim = 0; dim < hiddenSize; dim++)
            {
                result[dim] += embeddings[0, token, dim];
            }
        }

        if (maskSum == 0)
        {
            maskSum = 1;
        }

        for (var i = 0; i < hiddenSize; i++)
        {
            result[i] = (float)(result[i] / maskSum);
        }

        var norm = Math.Sqrt(result.Sum(x => x * x));
        if (norm == 0)
        {
            return result;
        }

        for (var i = 0; i < hiddenSize; i++)
        {
            result[i] /= (float)norm;
        }

        return result;
    }

    private sealed record SessionCreateResult(
        InferenceSession Session,
        string Provider,
        bool UsedGpu,
        bool GpuFallbackToCpu,
        Exception? GpuException);

    private static SessionCreateResult CreateSessionPreferGpu(string modelPath)
    {
        using var gpuOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };

        var appendedProvider = TryAppendBestGpuExecutionProvider(gpuOptions);

        try
        {
            var session = new InferenceSession(modelPath, gpuOptions);
            if (!string.IsNullOrWhiteSpace(appendedProvider))
            {
                return new SessionCreateResult(
                    Session: session,
                    Provider: appendedProvider!,
                    UsedGpu: true,
                    GpuFallbackToCpu: false,
                    GpuException: null);
            }

            return new SessionCreateResult(
                Session: session,
                Provider: "CPU",
                UsedGpu: false,
                GpuFallbackToCpu: false,
                GpuException: null);
        }
        catch (Exception gpuEx) when (!string.IsNullOrWhiteSpace(appendedProvider))
        {
            try
            {
                using var cpuOptions = new SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
                };

                var session = new InferenceSession(modelPath, cpuOptions);
                return new SessionCreateResult(
                    Session: session,
                    Provider: "CPU",
                    UsedGpu: false,
                    GpuFallbackToCpu: true,
                    GpuException: gpuEx);
            }
            catch (Exception cpuEx)
            {
                throw new AggregateException(
                    $"Failed to create ONNX Runtime session using GPU provider '{appendedProvider}', and CPU fallback also failed.",
                    gpuEx,
                    cpuEx);
            }
        }
    }

    private static string? TryAppendBestGpuExecutionProvider(SessionOptions options)
    {
        if (TryInvokeSessionOptionsMethod(options, "AppendExecutionProvider_CUDA", DefaultDeviceId))
        {
            return "CUDA";
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) &&
            TryInvokeSessionOptionsMethod(options, "AppendExecutionProvider_DML", DefaultDeviceId))
        {
            return "DirectML";
        }

        return null;
    }

    private static bool TryInvokeSessionOptionsMethod(SessionOptions options, string methodName, params object[] args)
    {
        try
        {
            var argTypes = args.Select(a => a.GetType()).ToArray();
            var method = typeof(SessionOptions).GetMethod(
                methodName,
                BindingFlags.Instance | BindingFlags.Public,
                binder: null,
                types: argTypes,
                modifiers: null);

            if (method is null)
            {
                return false;
            }

            method.Invoke(options, args);
            return true;
        }
        catch (TargetInvocationException)
        {
            return false;
        }
        catch
        {
            return false;
        }
    }
}

