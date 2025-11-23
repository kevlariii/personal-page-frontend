In my blog post ["Code-Aware Tokenization: ModernBERT"](https://www.mohammedsbaihi.com/blogs/modernbert.html) I have explored the code awareness in tokenization techniques used in ModernBERT comparted to the traditional BERT tokenizer. Where we confirmed the authors claims about the superiority of ModernBERT in handling code snippets effectively, at least in terms of tokenization.

But tokenization isn't the only aspect where ModernBERT evolves from BERT. ModernBERT came out in December 2024, almost 7 years after BERT's release in 2018. In this time, the field of NLP has seen significant advancements, especially with the rise of decoder-only models like GPT. Therefore in this post, we will compare ModernBERT and BERT across several dimensions including architecture, training paradigms and training data among others. Note that we will focus on the BASE versions of both models, but we'll highlight differences in LARGE versions where relevant.

## 1. Architecture
Both BERT and ModernBERT are based on the [Transformer architecture](https://arxiv.org/abs/1706.03762). They both utilize the encoder part of the Transformer, which is designed for Representation Learning (i.e., the goal is to learn rich representations of the input text) as opposed to decoder architectures used in models like GPT, which are optimized for text generation. However, ModernBERT incorporates several architectural improvements over BERT. These improvements are especially inspired by advancements in decoder models. We'll go over them in this section.

### 1.1 Embeddging Layer
To represent input embeddings, BERT uses a combination of token embeddings, segment embeddings and position embeddings.
These embeddings are summed together and are learned during training. Token embeddings are the base embeddings for each token in the vocabulary. Segment embeddings are used to differentiate between two sentences in the Next Sentence Prediction (NSP) task, which was one of the two pre-training objectives of BERT (alongside Masked Language Modeling (MLM)). Position embeddings are used to inject positional information into the model since during training (and inference as well in encoder models) tokens are processed in parallel and thus the model has no inherent sense of token order. 

Note that while the original BERT paper doesn't state whether they use learned (absolute) position embeddings or sinusoidal position encodings as in the original Transformer paper, but we can confirm that they use learned position embeddings by inspecting the BERT implementation in the HuggingFace `transformers` library.

```python
from transformers import BertModel
bert = BertModel.from_pretrained("bert-base-cased")
print(f"Type of Position Embedding:  {bert.config.position_embedding_type}")
print(f"Maximum Position:  {bert.config.max_position_embeddings}")
```

```bash
Type of Position Embedding:  absolute
Maximum Position:  512
```

We see that BERT uses absolute learned position embeddings with a maximum sequence length of 512 tokens. Which is the reason why BERT has a maximum input length of 512 tokens. This limit is especially hard capped by the architecture.

ModernBERT, on the other hand, only uses learned token embeddings, it doesn't use segment embeddings since it doesn't use the NSP objective during pre-training (we'll discuss this more in the Training Objectives section). And it doesn't use position embeddings either. Instead it uses [Rotary Position Embeddings (RoPE)](https://arxiv.org/abs/2104.09864) to inject positional information into the model. RoPE has been shown to be more effective than absolute position embeddings in capturing relative positions between tokens, which is crucial for understanding context in language. Additionally, RoPE allows ModernBERT to handle longer sequences more effectively than BERT, as it doesn't have a hard cap on input length like BERT does. While it still has a limit of 8192 tokens, this limit is soft and not architecturally enforced. This limit is especially achieved by tuning RoPE scaling parameters during training. Nevertheless, you can go beyone this limit during inference but with potential degradation in performance.

One common aspect here is that both models have a hidden dimension of 768 in their BASE versions and 1024 in their LARGE versions.

### 1.2 Encoder Layers
Both BERT and ModernBERT use the Transformer encoder layers. Each layer consists of the following building blocks (not in order):
- Self-Attention Mechanism
- MLP (Feed-Forward Neural Network)
- Normalization
- Residual Connections

But each model has its own tweaks to these building blocks. And we will discuss each one below. But one main difference is that BERT has 12 encoder layers in its BASE version and 24 in its LARGE version, while ModernBERT has 22 encoder layers in its BASE version and 28 in its LARGE version. Which also explain the difference in the number of parameters: 

| Model        | BASE  | LARGE  |
|--------------|-------|--------|
| BERT         | 110M  | 340M   |
| ModernBERT   | 149M  | 395M   |


#### 1.2.1 Attention Mechanism
BERT uses the standard Bidirectional Multi-Head Self-Attention mechanism with the traditional scaled dot-product attention (SDPA) formula as described in the original Transformer paper. I do not intend to go into the details of how self-attention works here, but you can refer to the [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) blog post for a detailed explanation.

ModernBERT also uses the same mechanism but with three main differences:
1. Alternating Attention:
    ModernBERT alternates between Global (Full) Attention and Local Attention. Global Attention is used every 3rd layer. See Figure below for a visual representation of these attention patterns.

    This means that in layers where Local Attention is used, each token can only attend to a fixed window $w$ of previous tokens ($w=128$ in BASE version). This significantly reduces the computational complexity of the attention mechanism from $O(L^2)$ in every layer in BERT to $O(L \cdot w)$ in Local Attention layers in ModernBERT, where $L$ is the sequence length. While the Global Attention layers still have a complexity of $O(L^2)$, They only make up 1/3 of the total layers, thus reducing the overall complexity of the model. 

2. RoPE:
    As I mentioned earlier, ModernBERT uses RoPE to inject positional information into the model. This happens in every attention layer by modifiying the query and key vectors before computing the attention scores.

    i.e $Q \leftarrow RoPE(Q)$ and $K \leftarrow RoPE(K)$

    You can refer to the [Rotary Position Embeddings paper](https://arxiv.org/abs/2104.09864) for more details on how RoPE works.

    The authors also affirm that they use different RoPE scaling factors:
    - Local layers: $\theta ≈ 10,000$
    - Global layers: $\theta ≈ 10,000$ for the first 1.7T tokens at a 1024 sequence length, then $\theta ≈ 160,000$ for the remaining 300B tokens to extend the effective context length to 8,192 tokens.


3. Flash Attention:
    ModernBERT uses a mixture of Flash Attention 3 for Global layers and Flash Attention 2 for Local layers. [FlashAttention](https://arxiv.org/abs/2205.14135) is a memory-efficient implementation of self-attention that avoids materializing the full $Q K^T$ attention matrix in GPU memory. Instead, it computes attention in tiled blocks, streaming data through fast on-chip SRAM, which drastically reduces memory access and speeds up training and inference while enabling much longer context lengths.

However, both models use the same number of attention heads: 12 in BASE versions and 16 in LARGE versions.

#### 1.2.2 MLP
BERT uses a standard two layer Feed-Forward Neural Network (FFN) with a GELU activation function in between. The hidden dimension of the FFN is 4 times the model's hidden dimension (i.e 4x expansion). 

```bash
(intermediate): BertIntermediate(
  (dense): Linear(768 → 3072)
  (intermediate_act_fn): GELU
)

(output): BertOutput(
  (dense): Linear(3072 → 768)
  (LayerNorm)
  (dropout p=0.1)
)
```
Parameter count:
- first layer: 768×3072 + 3072 (bias)= 2,359,296 + 3,072 = 2,362,368
- second layer: 3072×768 + 768 (bias)= 2,359,296 + 768 = 2,360,064
- Total: 4,72M parameters per encoder block.

ModernBERT, on the other hand, uses a modern gated MLP architecture.
````python
from transformers import ModernBertModel
modernbert = ModernBertModel.from_pretrained("answerdotai/ModernBERT-base")
modernbert.base_model.layers[0].mlp
````

```bash
ModernBertMLP(
  (Wi): Linear(in_features=768, out_features=2304, bias=False)
  (act): GELUActivation()
  (drop): Dropout(p=0.0, inplace=False)
  (Wo): Linear(in_features=1152, out_features=768, bias=False)
)
```

At first glance, the dimension might seem confusing, since the first layer expands the hidden dimension from 768 to 2304 but the second layer expects an input of dimension 1152. This implies that ModernBERT uses a [GeGLU](https://arxiv.org/pdf/2002.05202)-like mechanism. Here is what exactly happens:

1. The input hidden state $h$ of dimension 768 is passed through the first linear layer $W_i$ to produce an output of dimension 2304 (i.e 3x expansion).
2. This output is then split into two equal parts of dimension 1152 each: $a$ and $b$.
3. The activation function (GELU) is applied to the second part $b \leftarrow GELU(b)$.
4. The result is then element-wise multiplied with the first part $a$: $h = a \odot GELU(b)$. with $h$ now having a dimension of 1152.
5. Finally, this result is passed through the second linear layer $W_o$ to project it back to the original hidden dimension of 768.

Parameter count (no bias terms):
- first layer: 768×2304 = 1,769,472
- second layer: 1152×768 = 884,736
- Total: 2,65M parameters per encoder block.

That's 44% fewer parameters than BERT's MLP per encoder block, which is a significant reduction.

| Feature               | BERT MLP                | ModernBERT MLP          |
|-----------------------|-------------------------|-------------------------|
| FFN type              | Standard 2-layer MLP    | Gated MLP (GLU / GeGLU-style) |
| Expansion              | 768 → 3072              | 768 → 2304 (split into 1152 + 1152) |
| Effective hidden       | 3072                    | 1152 (after gating)     |
| Activation             | GELU                    | GELU (on gate branch)  |
| Gating                |  None                 | Multiplicative gate  |
| Bias terms            | Yes                  | No                   |
| Dropout               | 0.1                     | 0.0                     |
| Params per block      | ~4.72M                  | ~2.65M                  |
| Norm placement        | Post-norm               | Pre-norm                |

#### 1.2.3 Normalization
Both models use Layer Normalization. However, BERT applies LayerNorm after the addition of the residual connection (Post-LN), while ModernBERT applies LayerNorm before the attention and MLP blocks (Pre-LN). Pre-LN has been shown to improve training stability and convergence speed in deeper Transformer models.





