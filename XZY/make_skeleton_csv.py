# 作者：Xiang Zhaoyi
# 日期：2021/6/3 上午11:10
# 工具：PyCharm
"""
功能：批量检测视频中的骨架，并保存骨架信息到csv文件，且保存对应的视频
"""
import os
# os.chdir('../')
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


def main(args, is2021_data=False):
    """Visualize the demo images.
    Using mmdet to detect the human.
    is2021_data: True: 针对2021采集的数据做了特定的噪声处理

    """

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f'Faild to load video file {args.video_path}'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if is2021_data:
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),  # //2 是因为后面处理时//2
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2))
        else:
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),  # //2 是因为后面处理时//2
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'AVC1')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    while (cap.isOpened()):
        flag, img = cap.read()
        if not flag:
            break

        # region    ##############################################
        # 切除一部分图片 + 屏蔽左上角人的影响 + 放大图片
        if is2021_data:
            img = img[:, 0:img.shape[1] // 2, :]
            mask_pos = (40, 150, 3)
            mask_pos2 = (0, 390, 27, 440, 3)
            img = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            if "正常" not in args.video_path:
                points1 = np.array([[480, 0], [570, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]], np.int32)
                points2 = np.array([[0, 310], [332, 60], [500, 60], [500, 0], [0, 0]], np.int32)
                fill_color = (255,255,255)
                cv2.fillPoly(img, [points1], fill_color)
                cv2.fillPoly(img, [points2], fill_color)
        # endregion ##############################################

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)
        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        # region    ##############################################
        # 按照xmax排序，保留最右侧的 + 放大检测框20%，确保可以完整的框住整个人，防止丢掉部分关节

        if "正常" in args.video_path and is2021_data:
            for i in range(len(person_results) - 1, -1, -1):  # 删除右上一小部分有影响的box
                box = person_results[i]
                xmin, ymin, xmax, ymax, _ = box['bbox']
                # cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
                # if (cx > 390 and cx < 440 and cy > 0 and cy < 27):
                if ymax < 50:
                    del person_results[i]

        # region 根据视频而做的处理，以保证结果中只有一个人
        def cmp(x1, x2):
            # if x1['bbox'][3] + x1['bbox'][2] > x2['bbox'][3] + x2['bbox'][2]:
            h, w = img.shape[0:2]
            if x1['bbox'][3]/h + x1['bbox'][2]/w > x2['bbox'][3]/h + x2['bbox'][2]/w:
                return -1  # 大的在前
            else:
                return 1

        if len(person_results) > 0:
            person_results = sorted(person_results, key=functools.cmp_to_key(cmp))
            person_results = [person_results[0]]
        # endregion

        scale = 1.2
        for box in person_results:
            xmin, ymin, xmax, ymax, _ = box['bbox']
            cx, cy, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
            new_w, new_h = w * scale, h * scale
            xmin, ymin, xmax, ymax = max(cx - new_w / 2, 0), max(cy - new_h / 2, 0), min(cx + new_w / 2, img.shape[1] - 1), min(cy + new_h / 2, img.shape[0] - 1)
            box['bbox'] = np.array([xmin, ymin, xmax, ymax, _])
        # endregion ##############################################

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
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            dataset=dataset,
            kpt_score_thr=args.kpt_thr,
            show=False)

        # region    ##############################################
        # 保存keypoint到csv
        if len(pose_results) > 0:
            save_csv_for_per_video(pose_results[0]['keypoints'])
            # vis_img = show_keypoint(vis_img,pose_results[0]['keypoints'])
        # endregion ##############################################

        if args.show:
            cv2.imshow('Image', vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_out_video:
        videoWriter.release()
    cv2.destroyAllWindows()


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
    
python XZY/make_skeleton_csv.py \
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
    video_dir_list = [
        # "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/摔倒",
        # "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/正常站立",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/back_against",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/climb",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/fall_behind",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/fall_toward",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/hand_out",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/head_and_hand_out",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/normal_standing",
    ]
    for video_dir in video_dir_list:
        for video_name in os.listdir(video_dir):
            if (not video_name.endswith('.mp4')) and \
                    (not video_name.endswith('.avi')):
                continue
            # if '摔倒检测1' in video_name:
            #     continue
            video_path = os.path.join(video_dir, video_name)
            print(f'deal with video : {video_path}')
            gol.set_value("video_path", value=video_path)
            gol.set_value("change_video", value=True)
            args.video_path = video_path
            main(args=args)
