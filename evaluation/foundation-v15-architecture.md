# UniPilot Foundation v1.5 Current Architecture

This records the unchanged 19.5M Foundation baseline before any Phase 26 decision.

```text
token ids (B, T)
   |
   +--> tied token embedding (4096 x 384) --+
   +--> learned absolute position (512 x 384)+--> dropout 0.1
                                                |
             +----------------------------------+
             |  repeat 10 decoder blocks
             |
             +--> LN(eps=1e-5) --> joint QKV --> 6 heads x 64
             |                    --> QK^T/sqrt(64) --> causal mask
             |                    --> softmax(keys) --> dropout
             |                    --> values --> output projection --> dropout
             |                                      |
             +---------------- x + attention --------+
             |
             +--> LN(eps=1e-5) --> Linear 384->1536 --> GELU
                                  --> Linear 1536->384 --> dropout
                                                        |
             +---------------- x + FFN ------------------+
                                                |
                                      final LayerNorm
                                                |
                              tied bias-free LM head (384->4096)
                                                |
                                      next-token logits
```

## Exact specification

- type: `decoder-only autoregressive Transformer`
- layers: `10`
- hidden_dimension: `384`
- heads: `6`
- head_dimension: `64`
- ffn_dimension: `1536`
- activation: `GELU`
- positional_encoding: `learned absolute embedding`
- normalization: `LayerNorm`
- norm_placement: `Pre-LN`
- norm_epsilon: `1e-05`
- attention: `manual multi-head causal scaled dot-product self-attention`
- qkv_layout: `joint QKV projection then 3-way chunk`
- qkv_bias: `True`
- output_projection_bias: `True`
- embedding_scaling: `none`
- residual_connections: `['x + attention(norm1(x))', 'x + ffn(norm2(x))']`
- dropout: `0.1`
- attention_probability_dropout: `0.1`
- attention_output_dropout: `0.1`
- ffn_output_dropout: `0.1`
- lm_head: `linear d_model -> vocab, no bias`
- weight_tying: `True`
- initialization: `all Linear/Embedding weights Normal(mean=0,std=0.02); linear biases zero`
- residual_initialization: `no special residual scaling`
- attention_scaling: `QK^T / sqrt(head_dim) = QK^T / 8`
- softmax_dimension: `last/key dimension`
- parameters: `19514880`

## Parameter count

- token_embedding: `1,572,864`
- position_embedding: `196,608`
- attention: `5,913,600`
- normalization: `16,128`
- feed_forward: `11,815,680`
- unique_total: `19,514,880`
