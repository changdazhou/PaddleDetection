# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""
This code is based on https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/tracker/matching.py
"""

try:
    import lap
except:
    print(
        'Warning: Unable to use JDE/FairMOT/ByteTrack, please install lap, for example: `pip install lap`, see https://github.com/gatagat/lap'
    )
    pass

import scipy
import numpy as np
from scipy.spatial.distance import cdist
from ..motion import kalman_filter
from ..motion import KalmanFilter
from ..motion.kalman_filter import OCKalmanFilter
import warnings
warnings.filterwarnings("ignore")

__all__ = [
    'merge_matches',
    'linear_assignment',
    'bbox_ious',
    'iou_distance',
    'embedding_distance',
    'fuse_motion',
]


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix(
        (np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix(
        (np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def linear_assignment(cost_matrix, thresh):
    try:
        import lap
    except Exception as e:
        raise RuntimeError(
            'Unable to use JDE/FairMOT/ByteTrack, please install lap, for example: `pip install lap`, see https://github.com/gatagat/lap'
        )
    if cost_matrix.size == 0:
        return np.empty(
            (0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(
                range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def soft_biou_batch(bboxes1, bboxes2):
    """
    Computes soft BIoU between two bboxes in the form [x1,y1,x2,y2]
    BIoU is introduced in https://arxiv.org/pdf/2211.14317
    Soft BIoU is introduced as part of BoostTrack++
    # Author : Vukasin Stanojevic
    # Email  : vukasin.stanojevic@pmf.edu.rs
    """
    bboxes1=np.ascontiguousarray(bboxes1, dtype=np.float32)
    bboxes2=np.ascontiguousarray(bboxes2, dtype=np.float32)
    N = bboxes1.shape[0]
    K = bboxes2.shape[0]
    ious = np.zeros((N, K), dtype=bboxes2.dtype)
    if N * K == 0:
        return ious
    

    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    k1 = 0.25
    k2 = 0.5
    b2conf = bboxes2[..., 4]
    b1x1 = bboxes1[..., 0] - (bboxes1[..., 2]-bboxes1[..., 0]) * (1-b2conf)*k1
    b2x1 = bboxes2[..., 0] - (bboxes2[..., 2]-bboxes2[..., 0]) * (1-b2conf)*k2
    xx1 = np.maximum(b1x1, b2x1)

    b1y1 = bboxes1[..., 1] - (bboxes1[..., 3]-bboxes1[..., 1]) * (1-b2conf)*k1
    b2y1 = bboxes2[..., 1] - (bboxes2[..., 3]-bboxes2[..., 1]) * (1-b2conf)*k2
    yy1 = np.maximum(b1y1, b2y1)

    b1x2 = bboxes1[..., 2] + (bboxes1[..., 2]-bboxes1[..., 0]) * (1-b2conf)*k1
    b2x2 = bboxes2[..., 2] + (bboxes2[..., 2]-bboxes2[..., 0]) * (1-b2conf)*k2
    xx2 = np.minimum(b1x2, b2x2)

    b1y2 = bboxes1[..., 3] + (bboxes1[..., 3]-bboxes1[..., 1]) * (1-b2conf)*k1
    b2y2 = bboxes2[..., 3] + (bboxes2[..., 3]-bboxes2[..., 1]) * (1-b2conf)*k2
    yy2 = np.minimum(b1y2, b2y2)

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h

    o = wh / (
        (b1x2 - b1x1) * (b1y2 - b1y1)
        + (b2x2 - b2x1) * (b2y2 - b2y1)
        - wh
    )

    return o


def bbox_ious(atlbrs, btlbrs):
    boxes = np.ascontiguousarray(atlbrs, dtype=np.float32)
    query_boxes = np.ascontiguousarray(btlbrs, dtype=np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    ious = np.zeros((N, K), dtype=boxes.dtype)
    if N * K == 0:
        return ious

    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + 1))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]) + 1)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]) + 1)
                if ih > 0:
                    ua = float((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[
                        n, 3] - boxes[n, 1] + 1) + box_area - iw * ih)
                    ious[n, k] = iw * ih / ua
    return ious


def iou_distance(atracks, btracks,iou=False,metric="biou"):
    """
    Compute cost based on IoU between two list[STrack].
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [np.hstack((track.tlbr, track.score)) for track in atracks]
        btlbrs = [np.hstack((track.tlbr, track.score)) for track in btracks]
    # _biou = soft_biou_batch(atlbrs, btlbrs)
    if metric == "iou":
        _ious = bbox_ious(atlbrs, btlbrs)
    else:
        _ious = soft_biou_batch(atlbrs, btlbrs)
    if iou:
        return _ious
    cost_matrix = 1 - _ious

    return cost_matrix

def speed_direction(box1, box2):
    center_x1 ,center_y1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    center_x2 ,center_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    dx = center_x2 - center_x1
    dy = center_y2 - center_y1
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return np.array([dx,dy])


def cosine_similarity(x, y):
    # 将列表转换为NumPy数组
    x = np.array(x)
    y = np.array(y)
    
    # 计算点积
    dot_product = np.dot(x, y)
    
    # 计算向量的范数
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    
    # 计算余弦相似度
    similarity = dot_product / (norm_x * norm_y)
    
    # print(similarity)
    
    return similarity

def speed_angle_distance(tracks, detections,iou_threshold=0.3):
    cost_matrix = np.ones((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    
    trk_velocitys = [track.velocity for track in tracks]
    similarity = np.zeros((len(tracks), len(detections)),dtype=np.float32)
    mask = np.zeros((len(tracks), len(detections)),dtype=np.float32)
    for i in range(len(tracks)):
        track_velocity = trk_velocitys[i]
        for j in range(len(detections)): 
            det_velocity=speed_direction(tracks[i].tlwh, detections[j].tlwh)
            cosine_dist = cosine_similarity(track_velocity, det_velocity)
            similarity[i][j] = cosine_dist
    iou=iou_distance(tracks, detections,iou=True)
    # import pdb; pdb.set_trace()
    
    # print(f"iou: {iou}")
    # print(f"similiarity: {similarity}")
    # print(f"cost matrix: {cost_matrix}")
    
    # cost_matrix = np.multiply(iou, similarity)
    
    cost_matrix = 1 - similarity
    cost_matrix[np.where(iou<iou_threshold)] = np.inf
    # print(f"after multiplying with iou: {cost_matrix}")
    # cost_matrix[mask==0] = np.inf
    # print(cost_matrix)
    # print(cost_matrix)
    return cost_matrix



def embedding_distance(tracks, detections, metric='euclidean'):
    """
    Compute cost based on features between two list[STrack].
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray(
        [track.curr_feat for track in detections], dtype=np.float32)
    track_features = np.asarray(
        [track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features,
                                        metric))  # Nomalized features
    return cost_matrix


def fuse_motion(kf,
                cost_matrix,
                tracks,
                detections,
                only_position=False,
                lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    if isinstance(kf,OCKalmanFilter):
        measurements = np.asarray([det.to_xysr() for det in detections])
    if isinstance(kf,KalmanFilter):
        measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean,
            track.covariance,
            measurements,
            only_position,
            metric='maha')
        # import pdb; pdb.set_trace()
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_
                                                         ) * gating_distance
    return cost_matrix
