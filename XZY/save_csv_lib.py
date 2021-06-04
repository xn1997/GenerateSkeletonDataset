# 作者：Xiang Zhaoyi
# 日期：2021/6/2 下午8:03
# 工具：PyCharm
from XZY import gol
from XZY.config import csv_idx_keypoint_map, coco_idx_keypoint_map
import pandas as pd
import cv2
import os

frame_count = 0  # 记录当前获得的是第几帧的检测结果


def show_keypoint(img, keypoints):
    """
    在img中画出keypoint
    """
    if len(keypoints) < 1:
        return img

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(17):
        x, y, score = keypoints[i]
        keypoint_str = coco_idx_keypoint_map['EN'][i]
        # cv2.circle(img, (int(x), int(y)), 4, colors[i], thickness=-1)
        cv2.putText(img, keypoint_str, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[i], 1)
    return img


def save_csv_for_per_video(keypoints):
    global frame_count
    frame_count += 1

    video_path = gol.get_value("video_path")
    csv_path = video_path.split('.')[0] + '.csv'
    if gol.get_value("change_video"):
        # 如果换了视频，就置当前帧数
        gol.set_value("change_video", False)
        frame_count = 1

    idx = [f'第{frame_count}帧第1个人']
    col = []
    for x in csv_idx_keypoint_map['CN'].values():
        col.append(x + 'x')
        col.append(x + 'y')
        col.append(x + 'c')
    col.append('activity')
    if frame_count == 1:
        # 创建空的csv文件
        pd.DataFrame(columns=col, dtype=float).to_csv(csv_path)
    csv_info = pd.read_csv(csv_path, index_col=0)  # 读取已有csv内的信息，跳过多余的第一行
    result_skeleton = pd.DataFrame(index=idx, columns=col, dtype=float)

    # region Description 改掉这一块，即可，其他保存不用变
    for i in range(18):
        if i != 17:
            x, y, score = keypoints[i]
            keypoint_str = coco_idx_keypoint_map['CN'][i]
        else:  # 根据左右肩计算出颈的结点
            x, y, score = (keypoints[5] + keypoints[6]) / 2
            keypoint_str = csv_idx_keypoint_map['CN'][1]
    # endregion

        #####################################################
        # 保存到csv文件
        #####################################################
        result_skeleton[keypoint_str + 'x'] = x
        result_skeleton[keypoint_str + 'y'] = y
        result_skeleton[keypoint_str + 'c'] = score
    if '正常' in csv_path:
        result_skeleton['activity'] = 'normal_standing'  # 标记当前的动作,手动设置
    else:
        result_skeleton['activity'] = 'fall'
    csv_info = csv_info.append(result_skeleton)
    csv_info.to_csv(csv_path)


if __name__ == '__main__':
    result = pd.DataFrame(index=['x', 'y', 'c'], columns=csv_idx_keypoint_map['CN'].values(), dtype=float)
    result['头'] = [5, 6, 7]
    print(result)
    print(result['头']['x'])

    a = pd.DataFrame(index=['v'], columns=csv_idx_keypoint_map['CN'].values(), dtype=float)
    result = result.append(a)
    result = result.append(a)
    result['头']['v'] = [20]
    print(result)

    empty = pd.DataFrame(columns=csv_idx_keypoint_map['CN'].values(), dtype=float)
    print(empty.append(a))
