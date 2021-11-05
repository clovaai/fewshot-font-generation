"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import split_dim
from .blocks import w_norm_dispatch, ConvBlock, norm_dispatch


class Attention(nn.Module):
    def __init__(self, C_in_q, C_in_kv, C_qk, C_v, w_norm='none', scale=False, n_heads=1,
                 down_kv=False, rel_pos_size=None):
        """
        Args:
            C_in_q: query source (encoder feature x)
            C_in_kv: key/value source (decoder feature y)
            C_qk: inner query/key dim, which should be same
            C_v: inner value dim, which same as output dim

            down_kv: Area attention for lightweight self-attention
                w/ mean pooling.
            rel_pos_size: height & width for relative positional embedding.
                If None or 0 is given, do not use relative positional embedding.
        """
        super().__init__()
        self.n_heads = n_heads
        self.down_kv = down_kv

        w_norm = w_norm_dispatch(w_norm)
        self.q_proj = w_norm(nn.Conv1d(C_in_q, C_qk, 1))
        self.k_proj = w_norm(nn.Conv1d(C_in_kv, C_qk, 1))
        self.v_proj = w_norm(nn.Conv1d(C_in_kv, C_v, 1))
        self.out = w_norm(nn.Conv2d(C_v, C_v, 1))

        if scale:
            self.scale = 1. / (C_qk ** 0.5)

        if rel_pos_size:
            C_h_qk = C_qk // n_heads
            self.rel_pos = RelativePositionalEmbedding2d(
                C_h_qk, rel_pos_size, rel_pos_size, down_kv=down_kv
            )

    def forward(self, x, y):
        """ Attend from x (decoder) to y (encoder)

        Args:
            x: decoder feature
            y: encoder feature
        """
        B, C, H, W = x.shape
        flat_x = x.flatten(start_dim=2) # [B, C, H*W]

        if not self.down_kv:
            flat_y = y.flatten(start_dim=2)
        else:
            y_down = F.avg_pool2d(y, 2)
            flat_y = y_down.flatten(2) # [B, C, H*W/4]

        query = self.q_proj(flat_x) # [B, C_qk, H*W]
        key = self.k_proj(flat_y) # [B, C_qk, H*W]
        value = self.v_proj(flat_y) # [B, C, H*W]

        query = split_dim(query, 1, self.n_heads)
        key = split_dim(key, 1, self.n_heads)
        value = split_dim(value, 1, self.n_heads)

        attn_score = torch.einsum('bhcq,bhck->bhqk', query, key) # [B, n_heads, H*W, H*W]
        if hasattr(self, 'rel_pos'):
            attn_score += self.rel_pos(query)
        if hasattr(self, 'scale'):
            attn_score *= self.scale

        attn_w = F.softmax(attn_score, dim=-1)
        attn_out = torch.einsum('bhqk,bhck->bhcq', attn_w, value).reshape(B, C, H, W)
        out = self.out(attn_out)

        return out


class AttentionFFNBlock(nn.Module):
    """ Transformer-like attention + ffn block """
    def __init__(self, C_in_q, C_in_kv, C_qk, C_v, size, scale=True, norm='ln',
                 dropout=0.1, activ='relu', n_heads=1, ffn_mult=4, area=False, rel_pos=False):
        super().__init__()
        self.C_out = C_v
        if rel_pos:
            rel_pos = size
        self.attn = Attention(
            C_in_q, C_in_kv, C_qk, C_v,
            scale=scale, n_heads=n_heads, down_kv=area, rel_pos_size=rel_pos
        )
        self.dropout = nn.Dropout2d(dropout)
        self.ffn = nn.Sequential(
            ConvBlock(C_v, C_v*ffn_mult, 1, 1, 0, activ='none'),
            nn.Dropout2d(dropout),
            ConvBlock(C_v*ffn_mult, C_v, 1, 1, 0, activ=activ)
        )
        if norm == 'ln':
            self.norm = nn.LayerNorm([C_v, size, size])
        else:
            norm = norm_dispatch(norm)
            self.norm = norm(C_v)

    def forward(self, x, y):
        skip = x
        x = self.norm(x)
        x = self.attn(x, y)
        x = self.dropout(x)

        x = self.ffn(x)
        x += skip

        return x


class SAFFNBlock(AttentionFFNBlock):
    def __init__(self, C, size, C_qk_ratio=0.25, scale=True, norm='ln', dropout=0.1, activ='relu',
                 n_heads=1, ffn_mult=4, area=False, rel_pos=False):
        C_in_q = C
        C_in_kv = C
        C_qk = int(C * C_qk_ratio)
        C_v = C

        super().__init__(
            C_in_q, C_in_kv, C_qk, C_v, size, scale, norm, dropout, activ, n_heads,
            ffn_mult, area, rel_pos
        )

        self.C_in = C

    def forward(self, x):
        return super().forward(x, x)


class GlobalContext(nn.Module):
    """ Global-context """
    def __init__(self, C, bottleneck_ratio=0.25, w_norm='none'):
        super().__init__()
        C_bottleneck = int(C * bottleneck_ratio)
        w_norm = w_norm_dispatch(w_norm)
        self.k_proj = w_norm(nn.Conv2d(C, 1, 1))
        self.transform = nn.Sequential(
            w_norm(nn.Linear(C, C_bottleneck)),
            nn.LayerNorm(C_bottleneck),
            nn.ReLU(),
            w_norm(nn.Linear(C_bottleneck, C))
        )

    def forward(self, x):
        # x: [B, C, H, W]
        context_logits = self.k_proj(x)  # [B, 1, H, W]
        context_weights = F.softmax(context_logits.flatten(1), dim=1)  # [B, HW]
        context = torch.einsum('bci,bi->bc', x.flatten(2), context_weights)
        out = self.transform(context)

        return out[..., None, None]


class GCBlock(nn.Module):
    """ Global-context block """
    def __init__(self, C, bottleneck_ratio=0.25, w_norm='none'):
        super().__init__()
        self.gc = GlobalContext(C, bottleneck_ratio, w_norm)

    def forward(self, x):
        gc = self.gc(x)
        return x + gc


class RelativePositionalEmbedding2d(nn.Module):
    """ Learned relative positional embedding
    return Q * (R_x + R_y) for input Q and learned embedding R
    """
    def __init__(self, emb_dim, H, W, down_kv=False):
        super().__init__()
        self.H = H
        self.W = W
        self.down_kv = down_kv

        self.h_emb = nn.Embedding(H*2-1, emb_dim)
        self.w_emb = nn.Embedding(W*2-1, emb_dim)

        rel_y, rel_x = self.rel_grid()
        self.register_buffer('rel_y', rel_y)
        self.register_buffer('rel_x', rel_x)

    def rel_grid(self):
        # rel_y in [-(H-1), (H-1)]
        # rel_x in [-(W-1), (W-1)]
        y, x = torch.meshgrid(torch.arange(self.H), torch.arange(self.W))

        # rel_y[i, j] = j_y - i_y
        # rel_x[i, j] = j_x - i_x
        rel_y = y.reshape(1, -1) - y.reshape(-1, 1)
        rel_x = x.reshape(1, -1) - x.reshape(-1, 1)

        if self.down_kv:
            def down(x):
                n_q, n_k = x.shape
                x = x.view(n_q, 1, int(n_k**0.5), int(n_k**0.5))
                return (F.avg_pool2d(x.float(), 2) - 0.5).flatten(1).long()

            rel_y = down(rel_y)
            rel_x = down(rel_x)

        # shifting negative values to semi-positive values (>=0)
        rel_y += (self.H-1)
        rel_x += (self.W-1)

        return rel_y, rel_x

    def forward(self, query):
        """
        Args:
            query: [B, n_heads, C_qk, H*W]

        return:
            [B, n_heads, H*W, H*W]
        """
        r_x = self.w_emb(self.rel_x)  # [H*W, H*W, C_qk]
        r_y = self.h_emb(self.rel_y)  # [H*W, H*W, C_qk]

        S_rel = torch.einsum('bhci,ijc->bhij', query, r_x + r_y)
        return S_rel

