# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block, global_pool_nlc, LayerScale
from timm.models.vision_transformer import VisionTransformer as _BaseVit
from timm.layers import use_fused_attn, Mlp, DropPath

import torch.nn.functional as F


def collate_with_patch_mask(batch, patch=16):
    imgs, labels = zip(*batch)

    # make H & W divisible by patch and pad to batch-max
    imgs = [F.pad(i, (0, (patch-i.shape[-1] % patch) % patch,
                       0, (patch-i.shape[-2] % patch) % patch)) for i in imgs]
    Hs, Ws = [i.shape[-2] for i in imgs], [i.shape[-1] for i in imgs]
    Hmax, Wmax = max(Hs), max(Ws)
    padded, masks = [], []
    for im, H, W in zip(imgs, Hs, Ws):
        padded.append(F.pad(im, (0, Wmax-W, 0, Hmax-H)))

        Hp, Wp = H//patch, W//patch
        Hmp, Wmp = Hmax//patch, Wmax//patch
        m = torch.zeros(Hmp*Wmp, dtype=torch.bool)   # False = keep
        m[Hp*Wp:] = True                             # True  = ignore
        masks.append(m)

    return torch.stack(padded), torch.stack(masks), torch.tensor(labels)


class MaskedAttention(nn.Module):
    """
    Exactly the same API / behaviour as timm.layers.Attention **plus**
    an optional boolean/float `attn_mask` whose shape can be

        (B, 1, N, N)   or   (B, N, N)   or   (B, h, N, N)

    True / non-zero positions are ignored.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                      attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim) \
                         .permute(2, 0, 3, 1, 4)          # 3 × B × h × N × d
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # -------- fused (Flash / SDPA) path ----------
        if self.fused_attn:
            # PyTorch ≥2.0 SDPA accepts bool or float mask
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,                     # <<<<<<<<<<
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        # -------- classic path ----------
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)               # B × h × N × N
            if attn_mask is not None:
                # broadcast if mask is (B×1×N×N) or (B×N×N)
                if attn_mask.dim() == 3:
                    attn_mask = attn_mask[:, None, :, :]  # → B×1×N×N
                attn = attn.masked_fill(attn_mask, float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v                                  # B × h × N × d

        x = x.transpose(1, 2).reshape(B, N, C)            # B × N × C
        x = self.proj_drop(self.proj(x))
        return x


class MaskedBlock(nn.Module):
    """
    Drop-in replacement for `Block` that understands `attn_mask`.
    Call signature:  y = block(x, attn_mask=None)
    Everything else (norms, ML P, drops, LayerScale) is unchanged.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MaskedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    # ---- forward: identical to original, plus the mask -----------------
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attn_mask is not None:  # attn_mask (B,1,1,N)
            pad_mask = (attn_mask.squeeze(1).squeeze(1) != 0)  # back to bool
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0)

        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(_BaseVit):
    """
    Vision Transformer with optional padding-token mask support.
    Pass `pad_mask` (B, N) where True marks dummy patches.
    No mask ⇒ identical behaviour to original timm ViT.
    """
    # ----------------------------- constructor (unchanged) -----------------------------
    def __init__(self, global_pool, head_layers=1, **kwargs):
        super().__init__(**kwargs)
        self.global_pool = global_pool
        num_classes = kwargs['num_classes']

        layers = []
        for _ in range(head_layers - 1):
            layers.extend([nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU()])
        layers.append(nn.Linear(self.embed_dim, num_classes))
        self.head = nn.Sequential(*layers) if head_layers > 1 else nn.Linear(
            self.embed_dim, num_classes
        )

    # ----------------------------- helpers -----------------------------
    @staticmethod
    def _masked_mean(x, mask):
        """x: B×N×D, mask: B×N (True=PAD)  → mean over non-pad tokens"""
        keep = (~mask).unsqueeze(-1)          # B×N×1
        denom = keep.sum(1, keepdim=True).clamp(min=1)
        return (x * keep).sum(1) / denom

    @staticmethod
    def _masked_max(x, mask):
        x = x.masked_fill(mask.unsqueeze(-1), float("-inf"))
        return x.max(1).values                # B×D

    def _make_attn_mask(self, pad_mask: torch.Tensor):
        """
        pad_mask : (B, Npatch) with True = padded token.
        returns  : (B, 1, 1, Nseq) suitable for SDPA’s `attn_mask`.
                  (only masks the *keys*, keeps query dimension free)
        """
        if self.num_prefix_tokens:
            pad_mask = F.pad(pad_mask, (self.num_prefix_tokens, 0), value=False)

        # convert bool → float additive
        mask = pad_mask.to(dtype=torch.float32) * -1e9  # 0 or -1e9
        return mask[:, None, None, :]

    def unfreeze_patch_embed(self):
        for p in self.patch_embed.parameters():
            p.requires_grad = True

    def freeze_encoder(self, num_blocks=None):
        if num_blocks is None:
            for p in self.blocks.parameters():
                p.requires_grad = False
        else:
            for p in self.blocks[:num_blocks].parameters():
                p.requires_grad = False
        for p in self.patch_embed.proj.parameters():
            p.requires_grad = False

    # ----------------------------- forward_features: add pad_mask arg ------------------
    def forward_features(self, x: torch.Tensor,
                         pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        # ———  run encoder ———
        attn_mask = self._make_attn_mask(pad_mask) if pad_mask is not None else None
        for block in self.blocks:
            x = block(x, attn_mask) if attn_mask is not None else block(x)

        x = self.norm(x)
        return x

    # ----------------------------- pool override (mask-aware if needed) ----------------
    def _pool(self, x, pad_mask=None):
        if self.attn_pool is not None:            # 'map' pooling path (ignore mask)
            return self.attn_pool(x)

        if pad_mask is None or self.global_pool in ("", "token"):
            return global_pool_nlc(
                x, pool_type=self.global_pool, num_prefix_tokens=self.num_prefix_tokens
            )

        # ---------- avg / avgmax / max with masking ----------
        spatial = x[:, self.num_prefix_tokens:]          # drop CLS etc.
        if self.global_pool.startswith("avg"):
            pooled = self._masked_mean(spatial, pad_mask)
        else:  # 'max'
            pooled = self._masked_max(spatial, pad_mask)

        # re-attach prefix tokens (CLS, etc.) if any
        if self.num_prefix_tokens:
            x = torch.cat([x[:, : self.num_prefix_tokens], pooled.unsqueeze(1)], 1)
        else:
            x = pooled.unsqueeze(1)                      # keep 3-D shape
        return x

    # ----------------------------- forward_head (pass mask) ----------------------------
    def forward_head(self, x, pad_mask=None, pre_logits: bool = False):
        x = self._pool(x, pad_mask)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    # ----------------------------- main forward ----------------------------------------
    def forward(self,
                x: torch.Tensor,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.forward_features(x, pad_mask)
        x = self.forward_head(x)
        return x


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_fn=MaskedBlock, dynamic_img_size=True,
        dynamic_img_pad=True, **kwargs)
    return model


def vit_medium_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_fn=MaskedBlock, dynamic_img_size=True,
        dynamic_img_pad=True, **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_fn=MaskedBlock, dynamic_img_size=True,
        dynamic_img_pad=True, **kwargs)
    return model
