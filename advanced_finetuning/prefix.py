# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Based on https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
# with some refactor
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.layers import use_fused_attn
from torch.jit import Final
import torch.nn.functional as F
from timm.layers import DropPath, Mlp, LayerScale
from typing import Optional


class PrefixEncoder(nn.Module):
    def __init__(self, num_prefix_tokens, embed_dim, num_layers):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.embed_dim = embed_dim
        self.prefix_tokens = nn.Parameter(torch.zeros(1, num_prefix_tokens, embed_dim))
        self.transform = nn.Sequential(
            nn.Tanh(),
            nn.Linear(embed_dim, num_layers * 2 * embed_dim),
        )

    def forward(self, batch_size):
        return self.transform(self.prefix_tokens.expand((batch_size, self.num_prefix_tokens, self.embed_dim)))


class AttentionWithPrefix(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer=nn.LayerNorm,
            pool: str = 'token'  # 'token' (default) or 'mean'
    ) -> None:
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
        self.pool = pool

    def forward(self, x: torch.Tensor, prefix_k, prefix_v) -> torch.Tensor:
        """
        x: (B, N, C)
        If self.prefix_encoder is provided, compute prefix tokens and prepend them to keys and values.
        """
        B, N, C = x.shape

        # Standard QKV computation: (B, N, 3*dim) -> reshape to (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: (B, num_heads, N, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        num_prefix_tokens = prefix_k.shape[1]
        prefix_k = prefix_k.view(B, num_prefix_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        prefix_v = prefix_v.view(B, num_prefix_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # Prepend the prefix tokens along the sequence dimension (dim=2)
        k = torch.cat([prefix_k, k], dim=2)
        v = torch.cat([prefix_v, v], dim=2)

        # Attention computation
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        # x: (B, num_heads, N, head_dim) → reshape back to (B, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockWithPrefix(nn.Module):
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
            attn: AttentionWithPrefix = None) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        if attn is not None:
            self.attn = attn
        else:
            self.attn = AttentionWithPrefix(
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

    def forward(self, x: torch.Tensor, prefix_k, prefix_v) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), prefix_k, prefix_v)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def create_prefix_tuning_model(model: VisionTransformer, pool: str = 'token'):
    """
    Replace each block in model.blocks with a BlockWithPrefix, copying over all parameters
    from the original block (including attention layer parameters). A shared PrefixEncoder
    is created and attached to the model.

    Args:
        model: The VisionTransformer instance.
        pool: Either 'token' (to use all prefix tokens) or 'mean' (to average them).

    Returns:
        The modified model with prefix tuning integrated.
    """
    new_blocks = []
    dim = model.embed_dim  # embedding dimension
    for old_block in model.blocks:
        # Extract hyperparameters from the original block.
        num_heads = old_block.attn.num_heads
        # Assume mlp_ratio is stored on the block; otherwise, use a default (e.g. 4.0).
        mlp_ratio = getattr(old_block, "mlp_ratio", 4.0)
        qkv_bias = (old_block.attn.qkv.bias is not None)
        qk_norm = not isinstance(old_block.attn.q_norm, nn.Identity)
        attn_drop = old_block.attn.attn_drop.p if hasattr(old_block.attn.attn_drop, 'p') else 0.0
        proj_drop = old_block.attn.proj_drop.p if hasattr(old_block.attn.proj_drop, 'p') else 0.0
        init_values = getattr(old_block, "init_values", None)
        drop_path = old_block.drop_path1.p if hasattr(old_block.drop_path1, 'p') else 0.0
        act_layer = nn.GELU  # or extract from old_block.mlp if available
        # norm_layer is provided as a constructor.
        norm_layer = nn.LayerNorm

        # Create a new AttentionWithPrefix instance with the same hyperparameters.
        new_attn = AttentionWithPrefix(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            pool=pool
        )
        # Copy the attention parameters.
        new_attn.qkv.load_state_dict(old_block.attn.qkv.state_dict())
        if qk_norm:
            new_attn.q_norm.load_state_dict(old_block.attn.q_norm.state_dict())
            new_attn.k_norm.load_state_dict(old_block.attn.k_norm.state_dict())
        new_attn.proj.load_state_dict(old_block.attn.proj.state_dict())
        if hasattr(old_block.attn.attn_drop, 'p'):
            new_attn.attn_drop.p = old_block.attn.attn_drop.p
        if hasattr(old_block.attn.proj_drop, 'p'):
            new_attn.proj_drop.p = old_block.attn.proj_drop.p

        # Now create a new BlockWithPrefix instance.
        new_block = BlockWithPrefix(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=old_block.mlp.__class__,
            attn=new_attn
        )

        # Copy normalization layers.
        new_block.norm1.load_state_dict(old_block.norm1.state_dict())
        new_block.norm2.load_state_dict(old_block.norm2.state_dict())
        # Copy the LayerScale if it exists (if not Identity, load its state).
        if not isinstance(old_block.ls1, nn.Identity):
            new_block.ls1.load_state_dict(old_block.ls1.state_dict())
        if not isinstance(old_block.ls2, nn.Identity):
            new_block.ls2.load_state_dict(old_block.ls2.state_dict())
        # Similarly, copy drop paths if applicable.
        if not isinstance(old_block.drop_path1, nn.Identity):
            new_block.drop_path1.load_state_dict(old_block.drop_path1.state_dict())
        if not isinstance(old_block.drop_path2, nn.Identity):
            new_block.drop_path2.load_state_dict(old_block.drop_path2.state_dict())
        # Copy the MLP state.
        new_block.mlp.load_state_dict(old_block.mlp.state_dict())

        new_blocks.append(new_block)

    # Replace the blocks in the model with our new blocks.
    model.blocks = nn.ModuleList(new_blocks)
    # Attach the shared prefix encoder to the model for use in forward passes.
    return model

