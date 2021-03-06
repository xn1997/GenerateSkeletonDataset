## 个人说明

本仓库用于自制骨架数据集,用于行为识别

### 使用

#### 生成单个视频骨架数据`make_skeleton_csv.py`

在程序中修改待处理视频的父目录列表

`video_dir_list = ["/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/摔倒, ]"`

运行

```shell
python XZY/make_skeleton_csv.py \
    --det_config demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_coco.py \
    --det_checkpoint http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth \
    --pose_config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288_udp.py \
    --pose_checkpoint https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w48_coco_384x288_udp-0f89c63e_20210223.pth \
    --video-path demo/resources/demo.mp4 \ # 这个参数随意设置,没有意义这里
    --out-video-root vis_results # 画出骨架的视频保存路径
```



**注意:**
这里直接使用的det+pose,尝试了det+tracking+pose但效果不好,频繁在几帧检测不到
理论上使用SOT可以避免很多问题,但mmpose的SOT效果巨差,如果有时间建议使用SOT制作单人数据集

#### 骨架数据组合

将所有视频的csv骨架数据保存到一个pkl文件,用于gcn训练

在`csv_dir_list`中添加csv文件所在路径,然后执行

```shell
python XZY/csv_to_pkl.py
```

就会在当前文件夹下生成包含所有动作骨架数据的`pkl文件`


#### 有问题的视频记录

ZCCT31_clop_image.avi

<div align="center">
    <img src="resources/mmpose-logo.png" width="400"/>
</div>
#### `7-19网络数据集骨架检测脚本.py`

面向普通数据的处理脚本

1. 自动检索二级目录下的所有视频
2. 将视频的检测结果及骨架数据保存至vis_results文件夹下的同名文件夹下（目录结构和视频结果一模一样）
3. csv文件结构如下
   ![image-20210730154633095](https://raw.githubusercontent.com/xn1997/picgo/master/image-20210730154633095.png)
   保存了每帧每个人的坐标（人是进行了跟踪的）

#### `csv_to_pkl.py`

将所有的csv文件合并到一个pkl中,方便`Project-ActionRecognize`仓库训练图卷积模型.

#### `9-29提取单帧骨架.py`

只用来展示骨架检测效果,并不保存骨架信息.

### 使用流程图

```mermaid
graph TD
运行`make_skeleton_csv.py`提取图片得到骨架信息csv文件-->
使用`csv_to_pkl.py`将所有的csv文件转换为一个pkl文件,用于图卷积的训练
```



## Introduction

English | [简体中文](README_CN.md)

[![Documentation](https://readthedocs.org/projects/mmpose/badge/?version=latest)](https://mmpose.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmpose/workflows/build/badge.svg)](https://github.com/open-mmlab/mmpose/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmpose/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmpose)
[![PyPI](https://badge.fury.io/py/mmpose.svg)](https://pypi.org/project/mmpose/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)

MMPose is an open-source toolbox for pose estimation based on PyTorch.
It is a part of the [OpenMMLab project](https://github.com/open-mmlab).

The master branch works with **PyTorch 1.3+**.

<div align="center">
    <img src="demo/resources/demo_coco.gif" width="600px" alt><br>
    COCO 17-keypoint pose estimation
</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/9464825/95552839-00a61080-0a40-11eb-818c-b8dad7307217.gif" width="600px" alt><br>

133-keypoint whole-body pose estimation ([full HD version](https://www.youtube.com/watch?v=pIJpQg8mXUU))

</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/11788150/114201893-4446ec00-9989-11eb-808b-5718c47c7b23.gif" width="600px" alt><br>
    2D animal_pose estimation
</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/15977946/118820606-02df2000-b8e9-11eb-9984-b9228101e780.gif" width="600px" alt><br>
    3D human pose estimation
</div>

### Major Features

- **Support diverse tasks**

  We support a wide spectrum of mainstream pose analysis tasks in current research community, including 2d multi-person human pose estimation, 2d hand pose estimation, 2d face landmark detection, 133 keypoint whole-body human pose estimation, 3d human mesh recovery, fashion landmark detection and animal pose estimation.
  See [demo.md](demo/README.md) for more information.

- **Higher efficiency and higher accuracy**

  MMPose implements multiple state-of-the-art (SOTA) deep learning models, including both top-down & bottom-up approaches. We achieve faster training speed and higher accuracy than other popular codebases, such as [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).
  See [benchmark.md](docs/benchmark.md) for more information.

- **Support for various datasets**

  The toolbox directly supports multiple popular and representative datasets, COCO, AIC, MPII, MPII-TRB, OCHuman etc.
  See [data_preparation.md](docs/data_preparation.md) for more information.

- **Well designed, tested and documented**

  We decompose MMPose into different components and one can easily construct a customized
  pose estimation framework by combining different modules.
  We provide detailed documentation and API reference, as well as unittests.

## [Model Zoo](https://mmpose.readthedocs.io/en/latest/modelzoo.html)

Supported algorithms:

<details open>
<summary>(click to collapse)</summary>

- [x] [DeepPose](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#div-align-center-deeppose-cvpr-2014-div) (CVPR'2014)
- [x] [CPM](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-cpm-cvpr-2016-div) (CVPR'2016)
- [x] [Hourglass](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-hourglass-eccv-2016-div) (ECCV'2016)
- [x] [MSPN](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-mspn-arxiv-2019-div) (ArXiv'2019)
- [x] [RSN](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-rsn-eccv-2020-div) (ECCV'2020)
- [x] [SimpleBaseline2D](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#div-align-center-simplebaseline2d-eccv-2018-div) (ECCV'2018)
- [x] [HRNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-hrnet-cvpr-2019-div) (CVPR'2019)
- [x] [HRNetv2](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-hrnetv2-tpami-2019-div) (TPAMI'2019)
- [x] [SCNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-scnet-cvpr-2020-div) (CVPR'2020)
- [x] [Associative Embedding](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#div-align-center-associative-embedding-nips-2017-div) (NeurIPS'2017)
- [x] [HigherHRNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-higherhrnet-cvpr-2020-div) (CVPR'2020)
- [x] [HMR](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#div-align-center-hmr-cvpr-2018-div) (CVPR'2018)
- [x] [SimpleBaseline3D](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#div-align-center-simplebaseline3d-iccv-2017-div) (ICCV'2017)
- [x] [InterNet](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#div-align-center-internet-eccv-2020-div) (ECCV'2020)
- [x] [VideoPose3D](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#div-align-center-videopose3d-cvpr-2019-div) (CVPR'2019)

</details>

Supported techniques:

<details open>
<summary>(click to collapse)</summary>

- [x] [Wingloss](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#div-align-center-wingloss-cvpr-2018-div) (CVPR'2018)
- [x] [DarkPose](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#div-align-center-darkpose-cvpr-2020-div) (CVPR'2020)
- [x] [UDP](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#div-align-center-udp-cvpr-2020-div) (CVPR'2020)
- [x] [FP16](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#div-align-center-fp16-arxiv-2017-div) (ArXiv'2017)
- [x] [Albumentations](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#div-align-center-albumentations-information-2020-div) (Information'2020)

</details>

Supported [datasets](https://mmpose.readthedocs.io/en/latest/datasets.html):

<details open>
<summary>(click to collapse)</summary>

- [x] [COCO](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-coco-eccv-2014-div) \[[homepage](http://cocodataset.org/)\] (ECCV'2014)
- [x] [COCO-WholeBody](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-coco-wholebody-eccv-2020-div) \[[homepage](https://github.com/jin-s13/COCO-WholeBody/)\] (ECCV'2020)
- [x] [MPII](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-mpii-cvpr-2014-div) \[[homepage](http://human-pose.mpi-inf.mpg.de/)\] (CVPR'2014)
- [x] [MPII-TRB](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-mpii-trb-iccv-2019-div) \[[homepage](https://github.com/kennymckormick/Triplet-Representation-of-human-Body)\] (ICCV'2019)
- [x] [AI Challenger](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-ai-challenger-arxiv-2017-div) \[[homepage](https://github.com/AIChallenger/AI_Challenger_2017)\] (ArXiv'2017)
- [x] [OCHuman](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-ochuman-cvpr-2019-div) \[[homepage](https://github.com/liruilong940607/OCHumanApi)\] (CVPR'2019)
- [x] [CrowdPose](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-crowdpose-cvpr-2019-div) \[[homepage](https://github.com/Jeff-sjtu/CrowdPose)\] (CVPR'2019)
- [x] [PoseTrack18](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-posetrack18-cvpr-2018-div) \[[homepage](https://posetrack.net/users/download.php)\] (CVPR'2018)
- [x] [MHP](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-mhp-acm-mm-2018-div) \[[homepage](https://lv-mhp.github.io/dataset)\] (ACM MM'2018)
- [x] [sub-JHMDB](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-jhmdb-iccv-2013-div) \[[homepage](http://jhmdb.is.tue.mpg.de/dataset)\] (ICCV'2013)
- [x] [Human3.6M](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-human3-6m-tpami-2014-div) \[[homepage](http://vision.imar.ro/human3.6m/description.php)\] (TPAMI'2014)
- [x] [300W](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-300w-imavis-2016-div) \[[homepage](https://ibug.doc.ic.ac.uk/resources/300-W/)\] (IMAVIS'2016)
- [x] [WFLW](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-wflw-cvpr-2018-div) \[[homepage](https://wywu.github.io/projects/LAB/WFLW.html)\] (CVPR'2018)
- [x] [AFLW](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-aflw-iccvw-2011-div) \[[homepage](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)\] (ICCVW'2011)
- [x] [COFW](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-cofw-iccv-2013-div) \[[homepage](http://www.vision.caltech.edu/xpburgos/ICCV13/)\] (ICCV'2013)
- [x] [OneHand10K](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-onehand10k-tcsvt-2019-div) \[[homepage](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html)\] (TCSVT'2019)
- [x] [FreiHand](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-freihand-iccv-2019-div) \[[homepage](https://lmb.informatik.uni-freiburg.de/projects/freihand/)\] (ICCV'2019)
- [x] [RHD](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-rhd-iccv-2017-div) \[[homepage](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)\] (ICCV'2017)
- [x] [CMU Panoptic HandDB](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-cmu-panoptic-handdb-cvpr-2017-div) \[[homepage](http://domedb.perception.cs.cmu.edu/handdb.html)\] (CVPR'2017)
- [x] [InterHand2.6M](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-interhand2-6m-eccv-2020-div) \[[homepage](https://mks0601.github.io/InterHand2.6M/)\] (ECCV'2020)
- [x] [DeepFashion](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-deepfashion-cvpr-2016-div) \[[homepage](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html)\] (CVPR'2016)
- [x] [Animal-Pose](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-animal-pose-iccv-2019-div) \[[homepage](https://sites.google.com/view/animal-pose/)\] (ICCV'2019)
- [x] [Horse-10](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-horse-10-wacv-2021-div) \[[homepage](http://www.mackenziemathislab.org/horse10)\] (WACV'2021)
- [x] [MacaquePose](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-macaquepose-biorxiv-2020-div) \[[homepage](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html)\] (bioRxiv'2020)
- [x] [Vinegar Fly](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-vinegar-fly-nature-methods-2019-div) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Nature Methods'2019)
- [x] [Desert Locust](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-desert-locust-elife-2019-div) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Elife'2019)
- [x] [Grévy’s Zebra](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-grevys-zebra-elife-2019-div) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Elife'2019)
- [x] [ATRW](https://mmpose.readthedocs.io/en/latest/papers/datasets.html#div-align-center-atrw-acm-mm-2020-div) \[[homepage](https://cvwc2019.github.io/challenge.html)\] (ACM MM'2020)

</details>

Supported backbones:

<details>
<summary>(click to expand)</summary>

- [x] [AlexNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-alexnet-neurips-2012-div) (NeurIPS'2012)
- [x] [VGG](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-vgg-iclr-2015-div) (ICLR'2015)
- [x] [ResNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-resnet-cvpr-2016-div) (CVPR'2016)
- [x] [ResNetV1D](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-resnetv1d-cvpr-2019-div) (CVPR'2019)
- [x] [ResNeSt](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-resnest-arxiv-2020-div) (ArXiv'2020)
- [x] [ResNext](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-resnext-cvpr-2017-div) (CVPR'2017)
- [x] [SEResNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-seresnet-cvpr-2018-div) (CVPR'2018)
- [x] [ShufflenetV1](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-shufflenetv1-cvpr-2018-div) (CVPR'2018)
- [x] [ShufflenetV2](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-shufflenetv2-eccv-2018-div) (ECCV'2018)
- [x] [MobilenetV2](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-mobilenetv2-cvpr-2018-div) (CVPR'2018)

</details>

Results and models are available in the *README.md* of each method's config directory.
A summary can be found in the [**model zoo**](https://mmpose.readthedocs.io/en/latest/modelzoo.html) page.
We will keep up with the latest progress of the community, and support more popular algorithms and frameworks.

If you have any feature requests, please feel free to leave a comment in [Issues](https://github.com/open-mmlab/mmpose/issues/9).

## Benchmark

We demonstrate the superiority of our MMPose framework in terms of speed and accuracy on the standard COCO keypoint detection benchmark.

| Model | Input size| MMPose (s/iter) | [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) (s/iter) | MMPose (mAP) | [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) (mAP) |
| :--- | :---------------: | :---------------: |:--------------------: | :----------------------------: | :-----------------: |
| resnet_50  | 256x192  | **0.28** | 0.64 | **0.718** | 0.704 |
| resnet_50  | 384x288  | **0.81** | 1.24 | **0.731** | 0.722 |
| resnet_101 | 256x192  | **0.36** | 0.84 | **0.726** | 0.714 |
| resnet_101 | 384x288  | **0.79** | 1.53 | **0.748** | 0.736 |
| resnet_152 | 256x192  | **0.49** | 1.00 | **0.735** | 0.720 |
| resnet_152 | 384x288  | **0.96** | 1.65 | **0.750** | 0.743 |
| hrnet_w32  | 256x192  | **0.54** | 1.31 | **0.746** | 0.744 |
| hrnet_w32  | 384x288  | **0.76** | 2.00 | **0.760** | 0.758 |
| hrnet_w48  | 256x192  | **0.66** | 1.55 | **0.756** | 0.751 |
| hrnet_w48  | 384x288  | **1.23** | 2.20 | **0.767** | 0.763 |

More details about the benchmark are available on [benchmark.md](docs/benchmark.md).

## Installation

Please refer to [install.md](docs/install.md) for installation.

## Data Preparation

Please refer to [data_preparation.md](docs/data_preparation.md) for a general knowledge of data preparation.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMPose.
There are also tutorials:

- [learn about configs](docs/tutorials/0_config.md)
- [finetune model](docs/tutorials/1_finetune.md)
- [add new dataset](docs/tutorials/2_new_dataset.md)
- [customize data pipelines](docs/tutorials/3_data_pipeline.md)
- [add new modules](docs/tutorials/4_new_modules.md)
- [export a model to ONNX](docs/tutorials/5_export_model.md)
- [customize runtime settings](docs/tutorials/6_customize_runtime.md)

## FAQ

Please refer to [FAQ](docs/faq.md) for frequently asked questions.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```

## Contributing

We appreciate all contributions to improve MMPose. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMPose is an open source project that is contributed by researchers and engineers from various colleges and companies.
We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new models.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): A Comprehensive Toolbox for Text Detection, Recognition and Understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab's next-generation toolbox for generative models.
