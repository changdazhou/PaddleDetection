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
This code is based on https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/tracker/multitracker.py
"""
import copy
import numpy as np
from collections import defaultdict

# from ..matching import jde_matching as matching
from ..matching import jdeoc_matching as matching
from ..motion import KalmanFilter
from .base_jde_tracker import TrackState, STrack
from .jde_tracker import JDETracker
from ..motion.kalman_filter import OCKalmanFilter
from .base_jde_tracker import joint_stracks, sub_stracks, remove_duplicate_stracks

from ppdet.core.workspace import register, serializable
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['JDEOCTracker']




@register
@serializable
class JDEOCTracker(JDETracker):
    __shared__ = ['num_classes']
    """
    JDE tracker, support single class and multi classes

    Args:
        use_byte (bool): Whether use ByteTracker, default False
        num_classes (int): the number of classes
        det_thresh (float): threshold of detection score
        track_buffer (int): buffer for tracker
        min_box_area (int): min box area to filter out low quality boxes
        vertical_ratio (float): w/h, the vertical ratio of the bbox to filter
            bad results. If set <= 0 means no need to filter bboxes，usually set
            1.6 for pedestrian tracking.
        tracked_thresh (float): linear assignment threshold of tracked 
            stracks and detections
        r_tracked_thresh (float): linear assignment threshold of 
            tracked stracks and unmatched detections
        unconfirmed_thresh (float): linear assignment threshold of 
            unconfirmed stracks and unmatched detections
        conf_thres (float): confidence threshold for tracking, also used in
            ByteTracker as higher confidence threshold
        match_thres (float): linear assignment threshold of tracked 
            stracks and detections in ByteTracker
        low_conf_thres (float): lower confidence threshold for tracking in
            ByteTracker
        input_size (list): input feature map size to reid model, [h, w] format,
            [64, 192] as default.
        motion (str): motion model, KalmanFilter as default
        metric_type (str): either "euclidean" or "cosine", the distance metric 
            used for measurement to track association.
    """

    def __init__(self,
                 use_byte=True,
                 num_classes=1,
                 det_thresh=0.3,
                 track_buffer=30,
                 min_box_area=0,
                 vertical_ratio=0,
                 tracked_thresh=0.7,
                 r_tracked_thresh=0.5,
                 unconfirmed_thresh=0.7,
                 conf_thres=0,
                 match_thres=0.8,
                 low_conf_thres=0.2,
                 input_size=[64, 192],
                 motion='KalmanFilter',
                 delta_t=3,
                 metric_type='euclidean'):
        self.use_byte = use_byte
        self.num_classes = num_classes
        self.det_thresh = det_thresh if not use_byte else conf_thres + 0.1
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.vertical_ratio = vertical_ratio

        self.tracked_thresh = tracked_thresh
        self.r_tracked_thresh = r_tracked_thresh
        self.unconfirmed_thresh = unconfirmed_thresh
        self.conf_thres = conf_thres
        self.match_thres = match_thres
        self.low_conf_thres = low_conf_thres

        self.input_size = input_size
        if motion == 'KalmanFilter':
            self.motion = KalmanFilter()
        elif motion == 'OCKalmanFilter':
            self.motion = OCKalmanFilter(delta_t=delta_t)
        self.metric_type = metric_type

        self.frame_id = 0
        self.tracked_tracks_dict = defaultdict(list)  # dict(list[STrack])
        self.lost_tracks_dict = defaultdict(list)  # dict(list[STrack])
        self.removed_tracks_dict = defaultdict(list)  # dict(list[STrack])

        self.max_time_lost = 0
        # max_time_lost will be calculated: int(frame_rate / 30.0 * track_buffer)

    def update(self, pred_dets, pred_embs=None):
        """
        Processes the image frame and finds bounding box(detections).
        Associates the detection with corresponding tracklets and also handles
            lost, removed, refound and active tracklets.

        Args:
            pred_dets (np.array): Detection results of the image, the shape is
                [N, 6], means 'cls_id, score, x0, y0, x1, y1'.
            pred_embs (np.array): Embedding results of the image, the shape is
                [N, 128] or [N, 512].

        Return:
            output_stracks_dict (dict(list)): The list contains information
                regarding the online_tracklets for the received image tensor.
        """
        self.frame_id += 1
        if self.frame_id == 1:
            STrack.init_count(self.num_classes)
        activated_tracks_dict = defaultdict(list)
        refined_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        pred_dets_dict = defaultdict(list)
        pred_embs_dict = defaultdict(list)

        # unify single and multi classes detection and embedding results
        for cls_id in range(self.num_classes):
            cls_idx = (pred_dets[:, 0:1] == cls_id).squeeze(-1)
            pred_dets_dict[cls_id] = pred_dets[cls_idx]
            if pred_embs is not None:
                pred_embs_dict[cls_id] = pred_embs[cls_idx]
            else:
                pred_embs_dict[cls_id] = None

        for cls_id in range(self.num_classes):
            """ Step 1: Get detections by class"""
            pred_dets_cls = pred_dets_dict[cls_id]
            pred_embs_cls = pred_embs_dict[cls_id]
            remain_inds = (pred_dets_cls[:, 1:2] > self.conf_thres).squeeze(-1)
            if remain_inds.sum() > 0:
                pred_dets_cls = pred_dets_cls[remain_inds]
                if pred_embs_cls is None:
                    # in original ByteTrack
                    detections = [
                        STrack(
                            STrack.tlbr_to_tlwh(tlbrs[2:6]),
                            tlbrs[1],
                            cls_id,
                            30,
                            temp_feat=None) for tlbrs in pred_dets_cls
                    ]
                else:
                    pred_embs_cls = pred_embs_cls[remain_inds]
                    detections = [
                        STrack(
                            STrack.tlbr_to_tlwh(tlbrs[2:6]), tlbrs[1], cls_id,
                            30, temp_feat) for (tlbrs, temp_feat) in
                        zip(pred_dets_cls, pred_embs_cls)
                    ]
            else:
                detections = []

            ''' Add newly detected tracklets to tracked_stracks'''
            unconfirmed_dict = defaultdict(list) # 非激活的轨迹列表
            tracked_tracks_dict = defaultdict(list) # 激活的轨迹列表
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    # previous tracks which are not active in the current frame are added in unconfirmed list
                    unconfirmed_dict[cls_id].append(track)
                else:
                    # Active tracks are added to the local list 'tracked_stracks'
                    tracked_tracks_dict[cls_id].append(track)
            """ Step 2: First association, with embedding"""
            # building tracking pool for the current frame
            track_pool_dict = defaultdict(list)
            track_pool_dict[cls_id] = joint_stracks(
                tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])


            # Predict the current location with KalmanFilter
            #TODO：更新的KF需要改
            # 根据轨迹池更新轨迹的预测位置
            STrack.multi_predict(track_pool_dict[cls_id], self.motion)

            if pred_embs_cls is None:
                # in original ByteTrack
                dists = matching.iou_distance(track_pool_dict[cls_id],
                                              detections)
                matches, u_track, u_detection = matching.linear_assignment(
                    dists, thresh=self.match_thres)  # not self.tracked_thresh
            else:
                dists = matching.embedding_distance(
                    track_pool_dict[cls_id],
                    detections,
                    metric=self.metric_type)
                #TODO：更新的KF需要改
                dists = matching.fuse_motion(
                    self.motion, dists, track_pool_dict[cls_id], detections)
                matches, u_track, u_detection = matching.linear_assignment(
                    dists, thresh=self.tracked_thresh)

            for i_tracked, idet in matches:
                # i_tracked is the id of the track and idet is the detection
                track = track_pool_dict[cls_id][i_tracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    # If the track is active, add the detection to the track
                    track.update(detections[idet], self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    # We have obtained a detection from a track which is not active,
                    # hence put the track in refind_stracks list
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)

            # None of the steps below happen if there are no undetected tracks.
            """ Step 3: Second association, with IOU"""
            if self.use_byte:
                inds_low = pred_dets_dict[cls_id][:, 1:2] > self.low_conf_thres
                inds_high = pred_dets_dict[cls_id][:, 1:2] < self.conf_thres
                inds_second = np.logical_and(inds_low, inds_high).squeeze(-1)
                pred_dets_cls_second = pred_dets_dict[cls_id][inds_second]

                # association the untrack to the low score detections
                if len(pred_dets_cls_second) > 0:
                    if pred_embs_dict[cls_id] is None:
                        # in original ByteTrack
                        detections_second = [
                            STrack(
                                STrack.tlbr_to_tlwh(tlbrs[2:6]),
                                tlbrs[1],
                                cls_id,
                                30,
                                temp_feat=None)
                            for tlbrs in pred_dets_cls_second
                        ]
                    else:
                        pred_embs_cls_second = pred_embs_dict[cls_id][
                            inds_second]
                        detections_second = [
                            STrack(
                                STrack.tlbr_to_tlwh(tlbrs[2:6]), tlbrs[1],
                                cls_id, 30, temp_feat) for (tlbrs, temp_feat) in
                            zip(pred_dets_cls_second, pred_embs_cls_second)
                        ]
                else:
                    detections_second = []
                r_tracked_stracks = [
                    track_pool_dict[cls_id][i] for i in u_track
                    if track_pool_dict[cls_id][i].state == TrackState.Tracked
                ]
                dists = matching.iou_distance(r_tracked_stracks,
                                              detections_second)
                matches, u_track, u_detection_second = matching.linear_assignment(
                    dists, thresh=0.4)  # not r_tracked_thresh
            else:
                detections = [detections[i] for i in u_detection]
                r_tracked_stracks = []
                for i in u_track:
                    if track_pool_dict[cls_id][i].state == TrackState.Tracked:
                        r_tracked_stracks.append(track_pool_dict[cls_id][i])
                dists = matching.iou_distance(r_tracked_stracks, detections)

                matches, u_track, u_detection = matching.linear_assignment(
                    dists, thresh=self.r_tracked_thresh)
                

            for i_tracked, idet in matches:
                track = r_tracked_stracks[i_tracked]
                det = detections[
                    idet] if not self.use_byte else detections_second[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks_dict[cls_id].append(track)
                    
            """ Step 4: Third association, with speed direction"""
            use_speed = True if len(u_track)+len(u_detection)<20 else False
            # use_speed = False
            if use_speed:
                detections = [detections[i] for i in u_detection]
                # if detections:
                #     print([det.score for det in detections])
                r_tracked_stracks = [r_tracked_stracks[i] for i in u_track]
                # r_tracked_stracks = []
                # for i in u_track:
                #     # if track_pool_dict[cls_id][i].state == TrackState.Tracked:
                #     r_tracked_stracks.append(track_pool_dict[cls_id][i])
                # if r_tracked_stracks:
                #     print([trk.score for trk in r_tracked_stracks])
                dists = matching.speed_angle_distance(r_tracked_stracks, detections,iou_threshold=0.05)
                
                u_track = [i for i in range(len(r_tracked_stracks))]
                u_detection = [i for i in range(len(detections))]

                matches, u_track, u_detection = matching.linear_assignment(
                    dists, thresh=0.087)   
                
                for i_tracked, idet in matches:
                    track = r_tracked_stracks[i_tracked]
                    det = detections[
                        idet]
                    if track.state == TrackState.Tracked:
                        track.update(det, self.frame_id)
                        activated_tracks_dict[cls_id].append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
                        refined_tracks_dict[cls_id].append(track)
                if False:
                    inds_low = pred_dets_dict[cls_id][:, 1:2] > self.low_conf_thres
                    inds_high = pred_dets_dict[cls_id][:, 1:2] < self.conf_thres
                    inds_second = np.logical_and(inds_low, inds_high).squeeze(-1)
                    pred_dets_cls_second = pred_dets_dict[cls_id][inds_second]

                    # association the untrack to the low score detections
                    if len(pred_dets_cls_second) > 0:
                        if pred_embs_dict[cls_id] is None:
                            # in original ByteTrack
                            detections_second = [
                                STrack(
                                    STrack.tlbr_to_tlwh(tlbrs[2:6]),
                                    tlbrs[1],
                                    cls_id,
                                    30,
                                    temp_feat=None)
                                for tlbrs in pred_dets_cls_second
                            ]
                        else:
                            pred_embs_cls_second = pred_embs_dict[cls_id][
                                inds_second]
                            detections_second = [
                                STrack(
                                    STrack.tlbr_to_tlwh(tlbrs[2:6]), tlbrs[1],
                                    cls_id, 30, temp_feat) for (tlbrs, temp_feat) in
                                zip(pred_dets_cls_second, pred_embs_cls_second)
                            ]
                    else:
                        detections_second = []
                    r_tracked_stracks = [r_tracked_stracks[i] for i in u_track]
                    dists = matching.speed_angle_distance(r_tracked_stracks,
                                                detections_second,iou_threshold=0.1)
                    matches, u_track, u_detection_second = matching.linear_assignment(
                        dists, thresh=0.1)  # not r_tracked_thresh
                    
                    for i_tracked, idet in matches:
                        track = r_tracked_stracks[i_tracked]
                        det = detections_second[idet]
                        if track.state == TrackState.Tracked:
                            track.update(det, self.frame_id)
                            activated_tracks_dict[cls_id].append(track)
                        else:
                            track.re_activate(det, self.frame_id, new_id=False)
                            refined_tracks_dict[cls_id].append(track)

            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)
            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            detections = [detections[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed_dict[cls_id], detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(
                dists, thresh=self.unconfirmed_thresh)
            for i_tracked, idet in matches:
                unconfirmed_dict[cls_id][i_tracked].update(detections[idet],
                                                           self.frame_id)
                activated_tracks_dict[cls_id].append(unconfirmed_dict[cls_id][
                    i_tracked])
            for it in u_unconfirmed:
                track = unconfirmed_dict[cls_id][it]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)
            """ Step 5: Init new stracks"""
            for inew in u_detection:
                track = detections[inew]
                if track.score < self.det_thresh:
                    continue
                track.activate(self.motion, self.frame_id)
                activated_tracks_dict[cls_id].append(track)
            """ Step 6: Update state"""
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)

            self.tracked_tracks_dict[cls_id] = [
                t for t in self.tracked_tracks_dict[cls_id]
                if t.state == TrackState.Tracked
            ]
            self.tracked_tracks_dict[cls_id] = joint_stracks(
                self.tracked_tracks_dict[cls_id], activated_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id] = joint_stracks(
                self.tracked_tracks_dict[cls_id], refined_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_stracks(
                self.lost_tracks_dict[cls_id], self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_stracks(
                self.lost_tracks_dict[cls_id], self.removed_tracks_dict[cls_id])
            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[
                cls_id] = remove_duplicate_stracks(
                    self.tracked_tracks_dict[cls_id],
                    self.lost_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [
                track for track in self.tracked_tracks_dict[cls_id]
                if track.is_activated
            ]

            logger.debug('===========Frame {}=========='.format(self.frame_id))
            logger.debug('Activated: {}'.format(
                [track.track_id for track in activated_tracks_dict[cls_id]]))
            logger.debug('Refind: {}'.format(
                [track.track_id for track in refined_tracks_dict[cls_id]]))
            logger.debug('Lost: {}'.format(
                [track.track_id for track in lost_tracks_dict[cls_id]]))
            logger.debug('Removed: {}'.format(
                [track.track_id for track in removed_tracks_dict[cls_id]]))

        return output_tracks_dict
    
    
    
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """
# This code is based on https://github.com/noahcao/OC_SORT/blob/master/trackers/ocsort_tracker/ocsort.py
# """

# import pdb
# import numpy as np
# from ..matching import jde_matching as matching
# from collections import deque, OrderedDict
# from .base_jde_tracker import TrackState, STrack
# from .base_jde_tracker import joint_stracks, sub_stracks, remove_duplicate_stracks
# from collections import defaultdict
# from ..matching.ocsort_matching import associate, linear_assignment, iou_batch, associate_only_iou
# from ppdet.core.workspace import register, serializable
# from .ocsort_tracker import *

# ########################

# import numpy as np
# from collections import defaultdict

# from ..matching import jde_matching as matching
# from ..motion import KalmanFilter
# from .base_jde_tracker import TrackState, STrack
# from .jde_tracker import JDETracker
# from .base_jde_tracker import joint_stracks, sub_stracks, remove_duplicate_stracks

# from ppdet.core.workspace import register, serializable
# from ppdet.utils.logger import setup_logger
# logger = setup_logger(__name__)

# __all__ = ['JDEOCTracker']

# class JDEOCKalmanFilter(object):
#     """
#     This class represents the internal state of individual tracked objects observed as bbox.

#     Args:
#         bbox (np.array): bbox in [x1,y1,x2,y2,score] format.
#         delta_t (int): delta_t of previous observation
#     """
#     count = 0

#     def __init__(self, bbox, delta_t=3):

#         self.kf = OCSORTKalmanFilter(dim_x=7, dim_z=4)
#         self.kf.F = np.array([[1., 0, 0, 0, 1., 0, 0], [0, 1., 0, 0, 0, 1., 0],
#                               [0, 0, 1., 0, 0, 0, 1], [0, 0, 0, 1., 0, 0, 0],
#                               [0, 0, 0, 0, 1., 0, 0], [0, 0, 0, 0, 0, 1., 0],
#                               [0, 0, 0, 0, 0, 0, 1.]])
#         self.kf.H = np.array([[1., 0, 0, 0, 0, 0, 0], [0, 1., 0, 0, 0, 0, 0],
#                               [0, 0, 1., 0, 0, 0, 0], [0, 0, 0, 1., 0, 0, 0]])
#         self.kf.R[2:, 2:] *= 10.
#         self.kf.P[4:, 4:] *= 1000.
#         # give high uncertainty to the unobservable initial velocities
#         self.kf.P *= 10.
#         self.kf.Q[-1, -1] *= 0.01
#         self.kf.Q[4:, 4:] *= 0.01

#         self.score = bbox[4]
#         self.kf.x[:4] = convert_bbox_to_z(bbox)
#         self.time_since_update = 0
#         self.id = KalmanBoxTracker.count
#         KalmanBoxTracker.count += 1
#         self.history = []
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0
#         """
#         NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
#         function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
#         fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
#         """
#         self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
#         self.observations = dict()
#         self.history_observations = []
#         self.velocity = None
#         self.delta_t = delta_t

#     def update(self, bbox, angle_cost=False):
#         """
#         Updates the state vector with observed bbox.
#         """
#         if bbox is not None:
#             if angle_cost and self.last_observation.sum(
#             ) >= 0:  # no previous observation
#                 previous_box = None
#                 for i in range(self.delta_t):
#                     dt = self.delta_t - i
#                     if self.age - dt in self.observations:
#                         previous_box = self.observations[self.age - dt]
#                         break
#                 if previous_box is None:
#                     previous_box = self.last_observation
#                 """
#                   Estimate the track speed direction with observations \Delta t steps away
#                 """
#                 self.velocity = speed_direction(previous_box, bbox)
#             """
#               Insert new observations. This is a ugly way to maintain both self.observations
#               and self.history_observations. Bear it for the moment.
#             """
#             self.last_observation = bbox
#             self.observations[self.age] = bbox
#             self.history_observations.append(bbox)

#             self.time_since_update = 0
#             self.history = []
#             self.hits += 1
#             self.hit_streak += 1
#             self.kf.update(convert_bbox_to_z(bbox))
#         else:
#             self.kf.update(bbox)

#     def predict(self):
#         """
#         Advances the state vector and returns the predicted bounding box estimate.
#         """
#         if ((self.kf.x[6] + self.kf.x[2]) <= 0):
#             self.kf.x[6] *= 0.0

#         self.kf.predict()
#         self.age += 1
#         if (self.time_since_update > 0):
#             self.hit_streak = 0
#         self.time_since_update += 1
#         self.history.append(convert_x_to_bbox(self.kf.x, score=self.score))
#         return self.history[-1]

#     def get_state(self):
#         return convert_x_to_bbox(self.kf.x, score=self.score)

# @register
# @serializable
# class JDEOCTracker(JDETracker):
#     __shared__ = ['num_classes']
#     """
#     JDE tracker, support single class and multi classes

#     Args:
#         use_byte (bool): Whether use ByteTracker, default False
#         num_classes (int): the number of classes
#         det_thresh (float): threshold of detection score
#         track_buffer (int): buffer for tracker
#         min_box_area (int): min box area to filter out low quality boxes
#         vertical_ratio (float): w/h, the vertical ratio of the bbox to filter
#             bad results. If set <= 0 means no need to filter bboxes，usually set
#             1.6 for pedestrian tracking.
#         tracked_thresh (float): linear assignment threshold of tracked 
#             stracks and detections
#         r_tracked_thresh (float): linear assignment threshold of 
#             tracked stracks and unmatched detections
#         unconfirmed_thresh (float): linear assignment threshold of 
#             unconfirmed stracks and unmatched detections
#         conf_thres (float): confidence threshold for tracking, also used in
#             ByteTracker as higher confidence threshold
#         match_thres (float): linear assignment threshold of tracked 
#             stracks and detections in ByteTracker
#         low_conf_thres (float): lower confidence threshold for tracking in
#             ByteTracker
#         input_size (list): input feature map size to reid model, [h, w] format,
#             [64, 192] as default.
#         motion (str): motion model, KalmanFilter as default
#         metric_type (str): either "euclidean" or "cosine", the distance metric 
#             used for measurement to track association.
#     """

#     def __init__(self,
#                  use_byte=False,
#                  num_classes=1,
#                  det_thresh=0.3,
#                  track_buffer=30,
#                  min_box_area=0,
#                  vertical_ratio=0,
#                  tracked_thresh=0.7,
#                  r_tracked_thresh=0.5,
#                  unconfirmed_thresh=0.7,
#                  conf_thres=0,
#                  match_thres=0.8,
#                  low_conf_thres=0.2,
#                  input_size=[64, 192],
#                  motion='KalmanFilter',
#                  metric_type='euclidean'):
#         self.use_byte = use_byte
#         self.num_classes = num_classes
#         self.det_thresh = det_thresh if not use_byte else conf_thres + 0.1
#         self.track_buffer = track_buffer
#         self.min_box_area = min_box_area
#         self.vertical_ratio = vertical_ratio

#         self.tracked_thresh = tracked_thresh
#         self.r_tracked_thresh = r_tracked_thresh
#         self.unconfirmed_thresh = unconfirmed_thresh
#         self.conf_thres = conf_thres
#         self.match_thres = match_thres
#         self.low_conf_thres = low_conf_thres

#         self.input_size = input_size
#         if motion == 'KalmanFilter':
#             self.motion = KalmanFilter()
#         self.metric_type = metric_type

#         self.frame_id = 0
#         self.tracked_tracks_dict = defaultdict(list)  # dict(list[STrack])
#         self.lost_tracks_dict = defaultdict(list)  # dict(list[STrack])
#         self.removed_tracks_dict = defaultdict(list)  # dict(list[STrack])

#         self.max_time_lost = 0
#         # max_time_lost will be calculated: int(frame_rate / 30.0 * track_buffer)
        
#     def update_features(self, feat):
#         # L2 normalizing, this function has no use for BYTETracker
#         feat /= np.linalg.norm(feat)
#         self.curr_feat = feat
#         if self.smooth_feat is None:
#             self.smooth_feat = feat
#         else:
#             self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha
#                                                                 ) * feat
#         self.features.append(feat)
#         self.smooth_feat /= np.linalg.norm(self.smooth_feat)

#     def update(self, pred_dets, pred_embs=None):
#         """
#         Args:
#             pred_dets (np.array): Detection results of the image, the shape is
#                 [N, 6], means 'cls_id, score, x0, y0, x1, y1'.
#             pred_embs (np.array): Embedding results of the image, the shape is
#                 [N, 128] or [N, 512], default as None.

#         Return:
#             tracking boxes (np.array): [M, 6], means 'x0, y0, x1, y1, score, id'.
#         """
#         if pred_dets is None:
#             return np.empty((0, 6))

#         self.frame_count += 1
        
#         activated_tracks_dict = defaultdict(list)
#         refined_tracks_dict = defaultdict(list)
#         lost_tracks_dict = defaultdict(list)
#         removed_tracks_dict = defaultdict(list)
#         output_tracks_dict = defaultdict(list)

#         pred_dets_dict = defaultdict(list)
#         pred_embs_dict = defaultdict(list)
        
#         # unify single and multi classes detection and embedding results
#         for cls_id in range(self.num_classes):
#             cls_idx = (pred_dets[:, 0:1] == cls_id).squeeze(-1)
#             pred_dets_dict[cls_id] = pred_dets[cls_idx]
#             if pred_embs is not None:
#                 pred_embs_dict[cls_id] = pred_embs[cls_idx]
#             else:
#                 pred_embs_dict[cls_id] = None
                
#         for cls_id in range(self.num_classes):
#             # bboxes = pred_dets[:, 2:]
#             # scores = pred_dets[:, 1:2]
#             # dets = np.concatenate((bboxes, scores), axis=1)
#             # scores = scores.squeeze(-1)

#             # inds_low = scores > 0.1
#             # inds_high = scores < self.det_thresh
#             # inds_second = np.logical_and(inds_low, inds_high)
#             # # self.det_thresh > score > 0.1, for second matching
#             # dets_second = dets[inds_second]  # detections for second matching
#             # remain_inds = scores > self.det_thresh
#             # dets = dets[remain_inds]
#             pred_dets_cls = pred_dets_dict[cls_id]
#             pred_embs_cls = pred_embs_dict[cls_id]
#             remain_inds = (pred_dets_cls[:, 1:2] > self.conf_thres).squeeze(-1)
#             if remain_inds.sum() > 0:
#                 pred_dets_cls = pred_dets_cls[remain_inds]
#                 if pred_embs_cls is None:
#                     # without embedding, use dets for matching
#                     detections = [
#                         STrack(
#                             STrack.tlbr_to_tlwh(tlbrs[2:6]),
#                             tlbrs[1],
#                             cls_id,
#                             30,
#                             temp_feat=None) for tlbrs in pred_dets_cls
#                     ]
#                     pdb.set_trace()
#                 else:
#                     # with embedding, use embedding for matching
#                     pred_embs_cls = pred_embs_cls[remain_inds]
#                     detections = [
#                         STrack(
#                             STrack.tlbr_to_tlwh(tlbrs[2:6]), tlbrs[1], cls_id,
#                             30, temp_feat) for (tlbrs, temp_feat) in
#                         zip(pred_dets_cls, pred_embs_cls)
#                     ]
#                     pdb.set_trace()
#             else:
#                 detections = []
#             ''' Add newly detected tracklets to tracked_stracks'''
#             unconfirmed_dict = defaultdict(list) # 非激活的轨迹列表
#             tracked_tracks_dict = defaultdict(list) # 激活的轨迹列表
#             # 从已跟踪的轨迹池中加载轨迹信息
#             for track in self.tracked_tracks_dict[cls_id]:
#                 if not track.is_activated:
#                     # previous tracks which are not active in the current frame are added in unconfirmed list
#                     unconfirmed_dict[cls_id].append(track)
#                 else:
#                     # Active tracks are added to the local list 'tracked_stracks'
#                     tracked_tracks_dict[cls_id].append(track)
#             """ Step 2: First association, with embedding"""
#             # building tracking pool for the current frame
#             track_pool_dict = defaultdict(list)
#             # 首先用激活轨迹与失配轨迹进行匹配
#             track_pool_dict[cls_id] = joint_stracks(
#                 tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])
            
            
#             # # get predicted locations from existing trackers. N*5
#             # # 创建本次匹配的轨迹池。长度为当前已跟踪轨迹数，5维[x1, y1, x2, y2, score]
#             # trks = np.zeros((len(self.trackers), 5))
#             # to_del = []
#             # ret = []
#             # # 遍历本次轨迹池，使用KF的结果更新预测位置，用于后续匹配
#             # for t, trk in enumerate(trks):
#             #     pos = self.trackers[t].predict()[0]
#             #     # 更新到本次轨迹池
#             #     trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#             #     # 记录预测失败轨迹，用于后续删除操作
#             #     if np.any(np.isnan(pos)):
#             #         to_del.append(t)
#             # # 删除预测失败的轨迹
#             # trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#             # for t in reversed(to_del):
#             #     self.trackers.pop(t)
#             #TODO：更新的KF需要改
#             # 根据轨迹池更新轨迹的预测位置
#             STrack.multi_predict(track_pool_dict[cls_id], self.motion)
            
#             # matched, unmatched_dets, unmatched_trks = associate_only_iou(
#             # dets, trks, self.iou_threshold)
#             if pred_embs_cls is None:
#                 # in original ByteTrack
#                 dists = matching.iou_distance(track_pool_dict[cls_id],
#                                               detections)
#                 matches, u_track, u_detection = matching.linear_assignment(
#                     dists, thresh=self.match_thres)  # not self.tracked_thresh
#             else:
#                 dists = matching.embedding_distance(
#                     track_pool_dict[cls_id],
#                     detections,
#                     metric=self.metric_type)
#                 dists = matching.fuse_motion(
#                     self.motion, dists, track_pool_dict[cls_id], detections)
#                 matches, u_track, u_detection = matching.linear_assignment(
#                     dists, thresh=self.tracked_thresh) 
            
            
        
#         pdb.set_trace()

       
            
#         # trks 保存着KF预测之后的结果 = STrack.multi_predict

#         last_boxes = np.array([trk.last_observation for trk in self.trackers])
#         """
#             First round of association
#         """
#         matched, unmatched_dets, unmatched_trks = associate_only_iou(
#             dets, trks, self.iou_threshold)

#         for m in matched:
#             self.trackers[m[1]].update(
#                 dets[m[0], :], angle_cost=self.use_angle_cost)
#         """
#             Second round of associaton by OCR
#         """
#         # BYTE association
#         if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[
#                 0] > 0:
#             u_trks = trks[unmatched_trks]
#             iou_left = iou_batch(
#                 dets_second,
#                 u_trks)  # iou between low score detections and unmatched tracks
#             iou_left = np.array(iou_left)
#             if iou_left.max() > self.iou_threshold:
#                 """
#                     NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
#                     get a higher performance especially on MOT17/MOT20 datasets. But we keep it
#                     uniform here for simplicity
#                 """
#                 matched_indices = linear_assignment(-iou_left)
#                 to_remove_trk_indices = []
#                 for m in matched_indices:
#                     det_ind, trk_ind = m[0], unmatched_trks[m[1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold:
#                         continue
#                     self.trackers[trk_ind].update(
#                         dets_second[det_ind, :], angle_cost=self.use_angle_cost)
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_trks = np.setdiff1d(unmatched_trks,
#                                               np.array(to_remove_trk_indices))

#         if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
#             left_dets = dets[unmatched_dets]
#             left_trks = last_boxes[unmatched_trks]
#             iou_left = iou_batch(left_dets, left_trks)
#             iou_left = np.array(iou_left)
#             if iou_left.max() > self.iou_threshold:
#                 """
#                     NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
#                     get a higher performance especially on MOT17/MOT20 datasets. But we keep it
#                     uniform here for simplicity
#                 """
#                 rematched_indices = linear_assignment(-iou_left)
#                 to_remove_det_indices = []
#                 to_remove_trk_indices = []
#                 for m in rematched_indices:
#                     det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[
#                         1]]
#                     if iou_left[m[0], m[1]] < self.iou_threshold:
#                         continue
#                     self.trackers[trk_ind].update(
#                         dets[det_ind, :], angle_cost=self.use_angle_cost)
#                     to_remove_det_indices.append(det_ind)
#                     to_remove_trk_indices.append(trk_ind)
#                 unmatched_dets = np.setdiff1d(unmatched_dets,
#                                               np.array(to_remove_det_indices))
#                 unmatched_trks = np.setdiff1d(unmatched_trks,
#                                               np.array(to_remove_trk_indices))

#         for m in unmatched_trks:
#             self.trackers[m].update(None)

#         # create and initialise new trackers for unmatched detections
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(dets[i, :], delta_t=self.delta_t)
#             self.trackers.append(trk)

#         i = len(self.trackers)
#         for trk in reversed(self.trackers):
#             if trk.last_observation.sum() < 0:
#                 d = trk.get_state()[0]
#             else:
#                 d = trk.last_observation  # tlbr + score
#             if (trk.time_since_update < 1) and (
#                     trk.hit_streak >= self.min_hits or
#                     self.frame_count <= self.min_hits):
#                 # +1 as MOT benchmark requires positive
#                 ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
#             i -= 1
#             # remove dead tracklet
#             if (trk.time_since_update > self.max_age):
#                 self.trackers.pop(i)
#         if (len(ret) > 0):
#             return np.concatenate(ret)
#         return np.empty((0, 6))
