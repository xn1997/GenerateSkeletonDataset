import numpy as np

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker
import torch

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2):
        self.min_confidence = 0.3
        # yolov3中检测结果置信度阈值，筛选置信度小于0.3的detection。

        self.nms_max_overlap = 1.0
        # 非极大抑制阈值，设置为1代表不进行抑制

        # 用于提取图片的embedding,返回的是一个batch图片对应的特征
        self.extractor = Extractor("origin_model",
                                   model_path,
                                   use_cuda=True)

        max_cosine_distance = max_dist
        # 用在级联匹配的地方，如果大于改阈值，就直接忽略
        nn_budget = 100
        # 预算，每个类别最多的样本个数，如果超过，删除旧的

        # 第一个参数可选'cosine' or 'euclidean'
        metric = NearestNeighborDistanceMetric("cosine",
                                               max_cosine_distance,
                                               nn_budget)
        self.tracker = Tracker(metric, n_init=1)  # 检测到即视为跟踪到，不使用not confirmed状态

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        "---------提取reid特征---------跳入"
        features = self._get_features(bbox_xywh, ori_img)
        # 从原图中crop bbox对应图片并计算得到embedding
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)

        detections = [
            Detection(bbox_tlwh[i], conf, features[i])
            for i, conf in enumerate(confidences) if conf > self.min_confidence
        ]  # 筛选小于min_confidence的目标，并构造一个Detection对象构成的列表
        # Detection是一个存储图中一个bbox结果
        # 需要：1. bbox(tlwh形式) 2. 对应置信度 3. 对应embedding

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # 使用非极大抑制
        # 默认nms_thres=1的时候开启也没有用，实际上并没有进行非极大抑制
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        "---------跟踪匹配---------跳入"
        # update tracker
        # tracker给出一个预测结果，然后将detection传入，进行卡尔曼滤波操作
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        # 存储结果以及可视化
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return np.array(outputs)

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        bbox_xywh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_xywh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_xywh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
            "根据目标检测结果，截出所有的行人---此时并未进行resize"
        if im_crops:
            # 在这里调用并提取embedding
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y
