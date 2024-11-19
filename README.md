## Masked Spectrogram Modeling with Vision Transformers: A PyTorch Implementation

<p align="center">
  <img src=assets/fig_mae_setup.png width="480">
</p>


This is a PyTorch implementation of the paper [Building 6G Radio Foundation Models with Transformer Architectures](https://arxiv.org/abs/2411.09996):
```
@article{aboulfotouh2024building6gradiofoundation,
      title={Building 6G Radio Foundation Models with Transformer Architectures}, 
      author={Ahmed Aboulfotouh and Ashkan Eshaghbeigi and Hatem Abou-Zeid},
      year={2024},
      journal={arXiv:2411.09996},
}
```
* This repo is a modification of the amazing [mae repo](https://github.com/facebookresearch/mae).
### Catalog

- [x] Pre-trained checkpoints + fine-tuning code
- [x] Pre-training code

### Visualization demo

Run our interactive visualization demo using [Colab notebook]() (no GPU needed):
<p align="center">
  <img src="assets/fig_reconstructed_images.png" width="600">
</p>

### Fine-tuning with pre-trained checkpoints

The following table provides the pre-trained checkpoints used in the paper, converted from TF/TPU to PT/GPU:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-S</th>
<th valign="bottom">ViT-M</th>
<th valign="bottom">ViT-L</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth">download</a></td>
<td align="center"><a href="">download</a></td>
</tr>
<tr><td align="left">number of parameters</td>
<td align="center"><tt>30M</tt></td>
<td align="center"><tt>80M</tt></td>
<td align="center"><tt>300M</tt></td>
</tr>
</tbody></table>

The fine-tuning instruction is in [FINETUNE.md](FINETUNE.md).

By fine-tuning these pre-trained models, we rank #1 in these classification tasks (detailed in the paper):
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-S</th>
<th valign="bottom">ViT-M</th>
<th valign="bottom">ViT-L</th>
<th valign="bottom">ViT-H<sub>448</sub></th>
<td valign="bottom" style="color:#C0C0C0">prev best</td>
<!-- TABLE BODY -->
<tr><td align="left">ImageNet-1K (no external data)</td>
<td align="center">83.6</td>
<td align="center">85.9</td>
<td align="center">86.9</td>
<td align="center"><b>87.8</b></td>
<td align="center" style="color:#C0C0C0">87.1</td>
</tr>
<tr><td align="left">ImageNet-Corruption (error rate) </td>
<td align="center">51.7</td>
<td align="center">41.8</td>
<td align="center"><b>33.8</b></td>
<td align="center">36.8</td>
<td align="center" style="color:#C0C0C0">42.5</td>
</tr>
</tr>
</tbody></table>

### Pre-training

The pre-training instruction is in [PRETRAIN.md](PRETRAIN.md).

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
