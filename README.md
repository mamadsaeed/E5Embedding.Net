# E5Embedding.Net

> High-performance .NET library for generating **text embeddings using E5 models** with **ONNX Runtime**, supporting **CUDA**, **DirectML**, and automatic **CPU fallback**.

E5Embedding.Net provides a simple and production-ready API for integrating modern embedding models into .NET applications.

Designed for:
- Semantic Search
- Retrieval Augmented Generation (RAG)
- Vector Databases
- Document Similarity
- Recommendation Systems
- AI-powered Search

---

## Features

🚀 **High Performance**
- Optimized ONNX Runtime inference
- Efficient batch processing
- Async embedding generation

🎯 **E5 Model Support**
- Built specifically for E5 embedding models
- Supports retrieval-style embeddings (`query:` / `passage:`)

💻 **GPU Acceleration**
- NVIDIA CUDA support
- Windows DirectML support
- Automatic CPU fallback

🔧 **Flexible Tokenization**
- SentencePiece tokenizer support
- BERT WordPiece tokenizer support

📦 **Easy Integration**
- Simple API
- Dependency Injection support
- Microsoft.Extensions.Logging integration

🛡️ **Production Ready**
- Resource management
- Validation
- Error handling
- Logging support

---

## Installation

Install via .NET CLI:
```bash
dotnet add package E5Embedding.Net
```

Or via Package Manager Console:
```powershell
Install-Package E5Embedding.Net
```

---

## Architecture

The processing flow of the pipeline:

```
Text Input
    |
    v
Tokenizer
    |
    v
Token IDs + Attention Mask
    |
    v
ONNX Runtime
    |
    v
Embedding Vector
```

---

## Quick Start

```csharp
using E5Embedding.Net;

var config = new E5EmbeddingConfiguration
{
    OnnxModelPath = "./E5/model.onnx",
    SentencePieceModelFile = "./E5/sentencepiece.bpe.model",
    TokenizerConfigFile = "./E5/tokenizer_config.json",
    TokenizerJsonFile = "./E5/tokenizer.json",

    MaxSequenceLength = 512,
    Dimension = 1024,
    BatchSize = 16
};

using var embeddingService = new OnnxEmbeddingService(config);

var embedding = await embeddingService.EmbedAsync("This is a sample text.");

Console.WriteLine($"Embedding size: {embedding.Length}");
```

### Batch Embeddings

For multiple documents, use batch processing:

```csharp
var documents = new[]
{
    "Document one",
    "Document two",
    "Document three"
};

var embeddings = await embeddingService.EmbedBatchAsync(documents);
```
> **Note:** Batch processing improves throughput by reducing inference overhead.

---

## Retrieval Example

E5 models are optimized for retrieval scenarios using prefixes:
- **Query:** `query: <your search query>`
- **Passage:** `passage: <your document content>`

```csharp
var queryEmbedding = await service.EmbedAsync(
    "query: What is machine learning?"
);

var passageEmbedding = await service.EmbedAsync(
    "passage: Machine learning is a branch of AI..."
);
```

---

## Dependency Injection

Example registration:

```csharp
services.AddSingleton<E5EmbeddingConfiguration>(sp =>
{
    return new E5EmbeddingConfiguration
    {
        OnnxModelPath = "./model.onnx",
        MaxSequenceLength = 512,
        Dimension = 1024,
        BatchSize = 16
    };
});

services.AddSingleton<IEmbeddingService>(sp =>
{
    var config = sp.GetRequiredService<E5EmbeddingConfiguration>();
    var logger = sp.GetService<ILogger<OnnxEmbeddingService>>();

    return new OnnxEmbeddingService(config, logger);
});
```

---

## Configuration

### `E5EmbeddingConfiguration`

| Property | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `OnnxModelPath` | `string` | ONNX model location | *Required* |
| `SentencePieceModelFile` | `string` | SentencePiece model file | `sentencepiece.bpe.model` |
| `TokenizerConfigFile` | `string` | Tokenizer configuration | `tokenizer_config.json` |
| `TokenizerJsonFile` | `string` | Tokenizer metadata | `tokenizer.json` |
| `MaxSequenceLength` | `int` | Maximum tokens | *Required* |
| `Dimension` | `int` | Embedding dimension | `1024` |
| `BatchSize` | `int` | Batch processing size | `16` |

---

## GPU Acceleration

E5Embedding.Net automatically selects the best available execution provider in the following order:
1. **CUDA**
2. **DirectML**
3. **CPU**

No additional configuration is required. The selected provider is reported through logging.

---

## Tokenizers

### SentencePieceTokenizer
Recommended for E5 models.

```csharp
var tokenizer = new SentencePieceTokenizer(
    "sentencepiece.bpe.model",
    "tokenizer_config.json",
    "tokenizer.json",
    512
);

var encoding = tokenizer.Encode("Hello world");
```

### BertTokenizer
Supports BERT-style WordPiece tokenization.

```csharp
var tokenizer = new BertTokenizer(
    "tokenizer_config.json",
    "tokenizer.json",
    512
);

var result = tokenizer.Encode("Example text");
```

---

## Supported Models

Currently tested with [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large/tree/main/onnx).

Supported ONNX variants:
- `model.onnx`
- `model.onnx_data`
- `model_O4.onnx`
- `model_qint8_avx512_vnni.onnx`

### Model Files
Required files:
- `model.onnx`
- `model.onnx_data`
- `sentencepiece.bpe.model`
- `tokenizer.json`
- `tokenizer_config.json`

---

## Requirements

- .NET 8.0+
- ONNX Runtime
- E5 ONNX model files & Tokenizer files

**Supported Platforms:**
- Windows / Linux
- GPU: NVIDIA CUDA / DirectML compatible GPUs

---

## Performance Tips

1. **Reuse the Service:**
   Create one instance and reuse it as a singleton. The ONNX session is expensive to initialize.
   ```csharp
   services.AddSingleton<IEmbeddingService, OnnxEmbeddingService>();
   ```
2. **Use Batch Processing:**
   Prefer `EmbedBatchAsync()` for multiple texts.
3. **Dispose Resources:**
   Always dispose of the service when finished:
   ```csharp
   using var service = new OnnxEmbeddingService(config);
   ```

---

## Error Handling

Common exceptions:

| Exception | Description |
| :--- | :--- |
| `ArgumentNullException` | Missing required arguments |
| `FileNotFoundException` | Model or tokenizer files missing |
| `InvalidOperationException` | Invalid configuration |
| `AggregateException` | GPU and CPU initialization failure |

---

## Roadmap

- [ ] More E5 model variants
- [ ] Native AOT support
- [ ] Memory pooling optimization
- [ ] Additional quantized models
- [ ] Streaming embedding API
- [ ] Built-in similarity utilities

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Support

For issues, discussions, and contributions, visit the [GitHub Repository](https://github.com/mamadsaeed/E5Embedding.Net).