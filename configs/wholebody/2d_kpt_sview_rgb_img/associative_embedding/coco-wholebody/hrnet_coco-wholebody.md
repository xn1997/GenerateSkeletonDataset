<!-- [ALGORITHM] -->

<details>
<summary align="right">Associative Embedding (NIPS'2017)</summary>

```bibtex
@inproceedings{newell2017associative,
  title={Associative embedding: End-to-end learning for joint detection and grouping},
  author={Newell, Alejandro and Huang, Zhiao and Deng, Jia},
  booktitle={Advances in neural information processing systems},
  pages={2277--2287},
  year={2017}
}
```

<!-- [ALGORITHM] -->

<details>
<summary align="right">HRNet (CVPR'2019)</summary>

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
}
```

</details>

<!-- [DATASET] -->

<!-- [DATASET] -->

<details>
<summary align="right">COCO-WholeBody (ECCV'2020)</summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody v1.0 val  without multi-scale test

| Arch  | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR  | Hand AP | Hand AR | Whole AP | Whole AR | ckpt | log |
| :---- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----: | :-----: | :------: |:-------: |:------: | :------: |
| [HRNet-w32+](/configs/wholebody/2d_kpt_sview_rgb_img/associative_embedding/coco-wholebody/hrnet_w32_coco_wholebody_512x512.py)  | 512x512 | 0.551 | 0.650 | 0.271 | 0.451 | 0.564 | 0.618 | 0.159 | 0.238 | 0.342 | 0.453 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_wholebody_512x512_plus-f1f1185c_20210517.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_wholebody_512x512_plus_20210517.log.json) |
| [HRNet-w48+](/configs/wholebody/2d_kpt_sview_rgb_img/associative_embedding/coco-wholebody/hrnet_w48_coco_wholebody_512x512.py)  | 512x512 | 0.592 | 0.686 | 0.443 | 0.595 | 0.619 | 0.674 | 0.347 | 0.438 | 0.422 | 0.532 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_wholebody_512x512_plus-4de8a695_20210517.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_wholebody_512x512_plus_20210517.log.json) |

Note: `+` means the model is first pre-trained on original COCO dataset, and then fine-tuned on COCO-WholeBody dataset. We find this will lead to better performance.
