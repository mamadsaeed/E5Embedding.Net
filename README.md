# E5Embedding.Net

A high-performance .NET library for generating text embeddings using E5 models with ONNX Runtime. This library provides GPU acceleration support (CUDA/DirectML) with automatic CPU fallback, making it ideal for production environments.

## Features

- 🚀 **High Performance**: Optimized ONNX Runtime inference with GPU acceleration
- 🎯 **E5 Model Support**: Built specifically for E5 embedding models
- 🔧 **Flexible Tokenization**: Supports both SentencePiece and BERT-style tokenizers
- 💻 **GPU Acceleration**: Automatic GPU detection with CUDA and DirectML support
- 🔄 **Automatic Fallback**: Seamless fallback to CPU if GPU initialization fails
- 📦 **Easy Integration**: Simple API for embedding single texts or batches
- 🎨 **Production Ready**: Comprehensive error handling and logging support

## Installation

Install the package via NuGet:

```bash
dotnet add package E5Embedding.Net
```

Or via Package Manager:

```powershell
Install-Package E5Embedding.Net
```

## Quick Start

### Basic Usage

```csharp
using E5Embedding.Net;
using Microsoft.Extensions.Logging;

// Configure the embedding service
var config = new E5EmbeddingConfiguration
{
    OnnxModelPath = "path/to/model.onnx",
    SentencePieceModelFile = "path/to/sentencepiece.bpe.model",
    TokenizerConfigFile = "path/to/tokenizer_config.json",
    TokenizerJsonFile = "path/to/tokenizer.json",
    MaxSequenceLength = 512,
    Dimension = 1024,
    BatchSize = 16
};

// Create the service (optional logger)
var logger = LoggerFactory.Create(builder => builder.AddConsole()).CreateLogger<OnnxEmbeddingService>();
var embeddingService = new OnnxEmbeddingService(config, logger);

// Generate embedding for a single text
var embedding = await embeddingService.EmbedAsync("Your text here");

// Generate embeddings for multiple texts
var texts = new[] { "Text 1", "Text 2", "Text 3" };
var embeddings = await embeddingService.EmbedBatchAsync(texts);

// Don't forget to dispose
embeddingService.Dispose();
```

### Dependency Injection

```csharp
using E5Embedding.Net;
using Microsoft.Extensions.DependencyInjection;

// In your Startup.cs or Program.cs
services.AddSingleton<E5EmbeddingConfiguration>(sp =>
{
    var configuration = sp.GetRequiredService<IConfiguration>();
    return new E5EmbeddingConfiguration
    {
        OnnxModelPath = configuration["E5:OnnxModelPath"],
        SentencePieceModelFile = configuration["E5:SentencePieceModelFile"],
        TokenizerConfigFile = configuration["E5:TokenizerConfigFile"],
        TokenizerJsonFile = configuration["E5:TokenizerJsonFile"],
        MaxSequenceLength = configuration.GetValue<int>("E5:MaxSequenceLength"),
        Dimension = configuration.GetValue<int>("E5:Dimension"),
        BatchSize = configuration.GetValue<int>("E5:BatchSize", 16)
    };
});

services.AddSingleton<IEmbeddingService>(sp =>
{
    var config = sp.GetRequiredService<E5EmbeddingConfiguration>();
    var logger = sp.GetService<ILogger<OnnxEmbeddingService>>();
    return new OnnxEmbeddingService(config, logger);
});
```

## Configuration

### E5EmbeddingConfiguration Properties

| Property | Type | Description | Default |
|----------|------|-------------|---------|
| `OnnxModelPath` | `string` | Path to the ONNX model file | Required |
| `SentencePieceModelFile` | `string` | Path to SentencePiece model file | `"sentencepiece.bpe.model"` |
| `TokenizerConfigFile` | `string` | Path to tokenizer config JSON | `"tokenizer_config.json"` |
| `TokenizerJsonFile` | `string` | Path to tokenizer JSON file | `"tokenizer.json"` |
| `MaxSequenceLength` | `int` | Maximum sequence length for tokenization | Required |
| `Dimension` | `int` | Expected embedding dimension | `1024` |
| `BatchSize` | `int` | Batch size for processing multiple texts | `16` |

## GPU Acceleration

The library automatically detects and uses GPU acceleration when available:

- **CUDA**: Automatically used on systems with NVIDIA GPUs and CUDA support
- **DirectML**: Used on Windows systems with compatible GPUs
- **CPU Fallback**: Automatically falls back to CPU if GPU initialization fails

GPU usage is logged when the service is initialized. Check your logs to see which provider is being used.

## Tokenizers

### SentencePieceTokenizer

Used by default for E5 models. Supports BPE (Byte Pair Encoding) tokenization.

```csharp
using E5Embedding.Net.Tokenization;

var tokenizer = new SentencePieceTokenizer(
    sentencePieceModelFile: "path/to/sentencepiece.bpe.model",
    tokenizerConfigFile: "path/to/tokenizer_config.json",
    tokenizerJsonFile: "path/to/tokenizer.json",
    maxSequenceLength: 512
);

var encoding = tokenizer.Encode("Your text here");
```

### BertTokenizer

BERT-style WordPiece tokenizer for models that require it.

```csharp
using E5Embedding.Net.Tokenization;

var tokenizer = new BertTokenizer(
    tokenizerConfigFile: "path/to/tokenizer_config.json",
    tokenizerJsonFile: "path/to/tokenizer.json",
    maxSequenceLength: 512
);

var encoding = tokenizer.Encode("Your text here");
var pairEncoding = tokenizer.EncodePair("Query text", "Passage text");
```

## Requirements

- .NET 8.0 or later
- ONNX Runtime (included via NuGet package)
- E5 model files (ONNX format)
- Tokenizer files (SentencePiece model, config, and JSON)

## Model Files

You need the following files from your E5 model:

1. **model.onnx** - The ONNX model file
2. **sentencepiece.bpe.model** - SentencePiece tokenizer model (for SentencePieceTokenizer)
3. **tokenizer_config.json** - Tokenizer configuration
4. **tokenizer.json** - Tokenizer vocabulary and metadata

## Performance Tips

1. **Batch Processing**: Use `EmbedBatchAsync` for multiple texts to improve throughput
2. **Batch Size**: Adjust `BatchSize` based on your memory and performance requirements
3. **GPU**: Ensure GPU drivers are installed for best performance
4. **Dispose**: Always dispose the service when done to free resources

## Error Handling

The library provides comprehensive error handling:

- `ArgumentNullException`: When required parameters are null
- `FileNotFoundException`: When model or tokenizer files are missing
- `InvalidOperationException`: When configuration is invalid or dimensions don't match
- `AggregateException`: When both GPU and CPU initialization fail

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/mamadsaeed/E5Embedding.Net).