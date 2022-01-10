# 作者：Xiang Zhaoyi
# 日期：2021/6/3 上午11:10
# 工具：PyCharm
"""
输入：单张图片
功能：1. 画出骨架图 2. 原图标出指定的文字

文件夹下图片批量检测
注意:
    1. 不会保存检测结果,仅供骨架提取效果展示
"""
import os
# os.chdir('../')
from copy import deepcopy
from XZY.libs import gol

gol._init()
import numpy as np
import functools
from argparse import ArgumentParser

import cv2

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmdet.apis import show_result_pyplot
from XZY.libs.config import coco_idx_keypoint_map
from XZY.libs.save_csv_lib import save_csv_for_per_video, show_keypoint

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_mmdet_results(mmdet_results, cat_id=1):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 1 for human)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id - 1]

    person_results = []
    for bbox in bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results


def parser_init():
    parser = ArgumentParser()
    parser.add_argument('--det_config', default='demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py', help='Config file for detection')
    parser.add_argument('--det_checkpoint',
                        default='http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
                        help='Checkpoint file for detection')
    parser.add_argument('--pose_config', default='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py', help='Config file for pose')
    parser.add_argument('--pose_checkpoint', default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth',
                        help='Checkpoint file for pose')
    parser.add_argument('--video-path', default='None', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='vis_results',
        help='Root of the output video file. '
             'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.1,  # default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.0, help='Keypoint score threshold')  # default=0.3

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()
    return args


first_load_model = True  # 确保只加载一次模型


def main(args, is2021_data=False):
    """Visualize the demo images.
    Using mmdet to detect the human.
    is2021_data: True: 针对2021采集的数据做了特定的噪声处理

    """

    image_path = args.video_path  # 待处理的图片
    image_save_path = os.path.join(args.out_video_root,
                                   f'vis_{os.path.basename(args.video_path)}')
    args.bbox_thr = 0.3 # 0.1
    args.kpt_thr = 0.2 # 0.3
    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    global first_load_model, det_model, pose_model
    if first_load_model:
        det_model = init_detector(
            args.det_config, args.det_checkpoint, device=args.device.lower())
        # build the pose model from a config file and a checkpoint file
        pose_model = init_pose_model(
            args.pose_config, args.pose_checkpoint, device=args.device.lower())
        first_load_model = False

    dataset = pose_model.cfg.data['test']['type']

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    img = cv2.imread(image_path)

    # test a single image, the resulting box is (x1, y1, x2, y2)
    mmdet_results = inference_detector(det_model, img)
    # keep the person class bounding boxes.
    person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

    # test a single image, with a list of bboxes.
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        img,
        person_results,
        bbox_thr=args.bbox_thr,
        format='xyxy',
        dataset=dataset,
        return_heatmap=return_heatmap,
        outputs=output_layer_names)
    # show the results
    # 修改关节点，使其符合论文中关节点画法：去掉眼、耳朵，增加脖子记为5
    dataset = 'Paper'
    pose_results_back = deepcopy(pose_results)
    pose_results = []
    for person_id in range(len(pose_results_back)):
        person_info = pose_results_back[person_id]
        person_info['keypoints'][4][0:2] = (person_info['keypoints'][5][0:2] + person_info['keypoints'][6][0:2]) / 2
        person_info['keypoints'][1][2] = 0
        person_info['keypoints'][2][2] = 0
        person_info['keypoints'][3][2] = 0
        person_info['bbox'] = np.array([0, 0, 0, 0, 0])
        if person_id in [3,4]:
            continue
        pose_results.append(person_info)

    vis_img = vis_pose_result(
        pose_model,
        img,
        pose_results,
        radius=2,
        thickness=2,
        dataset=dataset,
        kpt_score_thr=args.kpt_thr,
        show=False)
    vis_binary_img = np.zeros(img.shape, np.uint8)
    vis_binary_img = vis_pose_result(
        pose_model,
        vis_binary_img,
        pose_results,
        radius=2,
        thickness=2,
        dataset=dataset,
        kpt_score_thr=args.kpt_thr,
        show=False)
    # 为每个人添加相同的文字
    for person_id in range(len(pose_results)):
        action_name = "Normal Standing"
        person_info = pose_results[person_id]
        head_pos = tuple(person_info['keypoints'][0][0:2])
        vis_img = cv2.putText(vis_img, action_name, head_pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 255), thickness=2)

    if args.show:
        cv2.imshow('Image', vis_img)
        cv2.waitKey(1)

    if save_out_video:
        cv2.imwrite(image_save_path, vis_img)
        cv2.imwrite(os.path.splitext(image_save_path)[0] + "binary.jpg", vis_binary_img)


################################################
# 运行代码
'''
python XZY/make_skeleton_csv.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --video-path demo/resources/demo.mp4 \
    --out-video-root vis_results
    
python XZY/9-29提取单帧骨架.py \
    --det_config demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_coco.py \
    --det_checkpoint http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth \
    --pose_config configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288_udp.py \
    --pose_checkpoint https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w48_coco_384x288_udp-0f89c63e_20210223.pth \
    --video-path demo/resources/demo.mp4 \
    --out-video-root vis_results
'''
################################################

if __name__ == '__main__':
    args = parser_init()
    image_dir = "/home/xzy/G/DeepLearning/Gitee/mmpose/tests/data/coco/"  # 所有图片所在的目录
    image_dir = "/media/xzy/My Passport/实验视频/单人行为类别/向前摔倒/"
    for image_name in os.listdir(image_dir):
        if (not image_name.endswith('.jpg')) and \
                (not image_name.endswith('.png')):
            continue
        video_path = os.path.join(image_dir, image_name)
        print(f'deal with video : {video_path}')
        gol.set_value("video_path", value=video_path)
        gol.set_value("change_video", value=True)
        args.video_path = video_path
        main(args=args)
