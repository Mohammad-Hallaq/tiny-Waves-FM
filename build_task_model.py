import argparse
import os
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, Block, PatchEmbed
from functools import partial
from models_ofdm_ce import CEViT

def main(args):

    # Load pruned model
    pruned_vit = torch.load(args.pruned_vit, weights_only=False)
    
    #Extract encoder components
    blocks = nn.Sequential(*list(pruned_vit.blocks))  # Convert ModuleList to Sequential
    norm = pruned_vit.norm  # Normalization layer

    if args.task == 'sensing':
        input_channels = 3
        output_dim = 6
    elif args.task == 'radio':
        input_channels = 1
        output_dim = 20
    elif args.task == 'positioning':
        input_channels = 4
        output_dim = 3
    elif args.task == 'ofdm':
        input_channels = 2

    if args.task == 'ofdm':
        vit_model = CEViT(
        patch_size=16, embed_dim=512, in_chans=input_channels, depth=12, num_heads=8,
        decoder_embed_dim=256, decoder_depth=1, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))

    else:
        # Create a new Vision Transformer model using the extracted encoder
        vit_model = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=512,
            depth=12,
            num_heads=8,
            mlp_ratio=4,
            in_chans=input_channels,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=512  
        )

    # Replace the relevant components
    vit_model.blocks = blocks
    vit_model.norm = norm

    for module in vit_model.modules():
        if isinstance(module, nn.LayerNorm):
            module.eps = 1e-06

    if args.num_blocks:
        vit_model.blocks = nn.Sequential(*list(pruned_vit.blocks[0:args.num_blocks]))

    if args.task != 'ofdm':
        # Build dynamic task head
        head_layers = []
        embed_dim = 512

        for _ in range(args.head_size - 1):
            head_layers += [nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1)]
            
        head_layers.append(nn.Linear(embed_dim, output_dim))
        vit_model.head = nn.Sequential(*head_layers)

    print(f"Model built for task: {args.task} with head size: {args.head_size}\n")
    print(vit_model)

    p_ratio = args.pruned_vit.partition('Vit_pruned_')[-1][:2]

    # Save pruned model
    os.makedirs(args.save_dir, exist_ok=True)

    if args.num_blocks:
        model_path = os.path.join(args.save_dir, f'pruned_ViT_with_{args.num_blocks}_block_for_{args.task}_task_pruned_{p_ratio}%.pth')
    else:
        model_path = os.path.join(args.save_dir, f'pruned_ViT_for_{args.task}_task_pruned_{p_ratio}%.pth')

    torch.save(vit_model, model_path)
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune the ViT encoder to a specific pruning ratio.")
    parser.add_argument('--pruned_vit', type=str, default='.',
                        help='Path to the pruned ViT')
    parser.add_argument('--task', type=str, default='sensing',
                        help='The task that the model will perform. The tasks are: Human Activity Sensing, Radio Signal Identification, 5G Positioning')
    parser.add_argument('--head_size', type=int, default=2,
                        help='The size of the linyar-layer task head.')
    parser.add_argument('--num_blocks', type=int, default=None,
                        help='The number of attention of blocks to keep from the autoencoder.')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory to save the new model.')
    
    args = parser.parse_args()
    main(args)
