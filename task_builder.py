import copy
import os
import argparse
from dataclasses import dataclass
from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

# If you keep CEViT, import it
from models_ofdm_ce import CEViT

@dataclass
class BuildConfig:
    pruned_vit_path: str
    task: str = "sensing"   # sensing | radio | positioning | ofdm
    head_size: int = 2
    num_blocks: int | None = None
    save_dir: str = "."

class ViTTaskBuilder:
    TASK_SPECS = {
        "sensing":     {"in_chans": 3, "out_dim": 6},
        "radio":       {"in_chans": 1, "out_dim": 20},
        "positioning": {"in_chans": 4, "out_dim": 3},
        "ofdm":        {"in_chans": 2, "out_dim": None},  # handled by CEViT
    }

    def __init__(self, cfg: BuildConfig):
        self.cfg = cfg
        self._validate_task()
        self.p_ratio = self._derive_p_ratio()

    def _validate_task(self):
        if self.cfg.task not in self.TASK_SPECS:
            raise ValueError(f"Unknown task: {self.cfg.task}")
        if self.cfg.head_size < 1:
            raise ValueError("head_size must be >= 1")

    def _load_pruned(self):
        pruned = torch.load(self.cfg.pruned_vit_path, map_location="cpu", weights_only=False)
        return pruned

    def _derive_p_ratio(self) -> str:
        name = os.path.basename(self.cfg.pruned_vit_path)
        tail = name.partition(".")[0]
        return ''.join(ch for ch in tail[-3:-1] if ch.isdigit()) or "xx"

    def _task_io(self):
        spec = self.TASK_SPECS[self.cfg.task]
        return spec["in_chans"], spec["out_dim"]

    def _make_head(self, embed_dim: int, out_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        hidden = embed_dim
        for _ in range(max(0, self.cfg.head_size - 1)):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1)]
        layers.append(nn.Linear(hidden, out_dim))
        return nn.Sequential(*layers)

    def build(self) -> nn.Module:
        pruned = self._load_pruned()
        in_ch, out_dim = self._task_io()

        # Extract encoder attributes from pruned model
        embed_dim   = getattr(pruned, "embed_dim", 512)
        depth       = len(pruned.blocks)
        num_heads   = getattr(pruned.blocks[0].attn, "num_heads", 8) if depth > 0 else 8
        patch_size  = getattr(getattr(pruned, "patch_embed", None), "patch_size", 16)
        img_size    = getattr(pruned, "img_size", 224)

        # Optionally keep only the first K blocks
        if self.cfg.num_blocks is not None:
            k = self.cfg.num_blocks
            if k < 1 or k > depth:
                raise ValueError(f"num_blocks must be in [1, {depth}]")
            blocks = nn.Sequential(*[copy.deepcopy(b) for b in pruned.blocks[:k]])
        else:
            blocks = nn.Sequential(*[copy.deepcopy(b) for b in pruned.blocks])

        # Build the base model for the task
        if self.cfg.task == "ofdm":
            model = CEViT(
                patch_size=patch_size,
                embed_dim=embed_dim,
                in_chans=in_ch,
                depth=len(blocks),
                num_heads=num_heads,
                decoder_embed_dim=embed_dim // 2,
                decoder_depth=1,
                decoder_num_heads=num_heads * 2,
                mlp_ratio=4,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
        else:
            model = VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                depth=len(blocks),
                num_heads=num_heads,
                mlp_ratio=4,
                in_chans=in_ch,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                num_classes=embed_dim,  # interim; we replace with task head
            )

        # ——— Carry over encoder blocks from the pruned model ———
        model.blocks = blocks

        # Ensure all LayerNorm eps are consistent
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.eps = 1e-6

        # Task head (skip for ofdm/autoencoder-style)
        if self.cfg.task != "ofdm":
            model.head = self._make_head(embed_dim, out_dim)

        return model

    def save(self, model: nn.Module) -> str:
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        p = self.p_ratio
        tag = f"_with_{self.cfg.num_blocks}_block" if self.cfg.num_blocks else ""
        fname = f"pruned_ViT{tag}_for_{self.cfg.task}_task_pruned_{p}%.pth"
        path = os.path.join(self.cfg.save_dir, fname)

        # We need to save the whole model beacause we cannot track the accurate architecutre after performing structured pruning:
        torch.save(model, path)
        return path


def main(args):
    cfg = BuildConfig(
        pruned_vit_path=args.pruned_vit,
        task=args.task,
        head_size=args.head_size,
        num_blocks=args.num_blocks,
        save_dir=args.save_dir,
    )
    builder = ViTTaskBuilder(cfg)
    model = builder.build()
    print(f"Model built for task: {cfg.task} with head size: {cfg.head_size}\n")
    print(model)
    save_path = builder.save(model)
    print(f"\nWeights saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build task-specific model from a pruned ViT")
    parser.add_argument("--pruned_vit", type=str, required=True,
                        help="Path to the pruned ViT .pt/.pth")
    parser.add_argument("--task", type=str, default="sensing",
                        help="sensing | radio | positioning | ofdm")
    parser.add_argument("--head_size", type=int, default=2,
                        help="Number of Linear blocks in the task head (>=1)")
    parser.add_argument("--num_blocks", type=int, default=None,
                        help="Keep only the first K Transformer blocks")
    parser.add_argument("--save_dir", type=str, default=".",
                        help="Where to save the new weights")
    main(parser.parse_args())
