import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from story_llm.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.attn_dropout = config.attn_dropout

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.out_proj.SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size()
        # project and split into heads: (B, T, C) -> (B, nh, T, dh)
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        # merge heads: (B, nh, T, dh) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.pre_norm = config.pre_norm
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.self_attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ffn, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.d_ffn, config.d_model, bias=config.bias),
            nn.Dropout(config.ffn_dropout),
        )
        self.ffn[2].SCALE_INIT = 1

    def forward(self, x):
        if self.pre_norm:
            # Pre-norm: normalise input before each sublayer (GPT-3 / modern style)
            x = x + self.self_attn(self.ln_1(x))
            x = x + self.ffn(self.ln_2(x))
        else:
            # Post-norm: normalise after the residual add (GPT-2 / original transformer)
            x = self.ln_1(x + self.self_attn(x))
            x = self.ln_2(x + self.ffn(x))
        return x


class StoryModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.dropout = nn.Dropout(config.ffn_dropout)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std = 0.02 / math.sqrt(2 * self.config.n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.size()
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=x.device).unsqueeze(0))
        x = self.dropout(tok_emb + pos_emb)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.head(x)
        
