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

import numpy as np
from collections import defaultdict
from collections import deque, OrderedDict
from ..matching import jdeoc_matching as matching
from ..motion import KalmanFilter
from ..motion.kalman_filter import OCKalmanFilter
from ppdet.core.workspace import register, serializable
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
import warnings
warnings.filterwarnings("ignore")

__all__ = [
    'TrackState',
    'BaseTrack',
    'STrack',
    'joint_stracks',
    'sub_stracks',
    'remove_duplicate_stracks',
]


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


@register
@serializable
class BaseTrack(object):
    _count_dict = defaultdict(int)  # support single class and multi classes

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feat = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id(cls_id):
        BaseTrack._count_dict[cls_id] += 1
        return BaseTrack._count_dict[cls_id]

    # @even: reset track id
    @staticmethod
    def init_count(num_classes):
        """
        Initiate _count for all object classes
        :param num_classes:
        """
        for cls_id in range(num_classes):
            BaseTrack._count_dict[cls_id] = 0

    @staticmethod
    def reset_track_count(cls_id):
        BaseTrack._count_dict[cls_id] = 0

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


@register
@serializable
class STrack(BaseTrack):
    def __init__(self, tlwh, score, cls_id, buff_size=30, temp_feat=None):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.cls_id = cls_id
        self.track_len = 0
        self._history = OrderedDict()
        self.velocitie = np.zeros(2)

        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.use_reid = True if temp_feat is not None else False
        if self.use_reid:
            self.smooth_feat = None
            self.update_features(temp_feat)
            self.features = deque([], maxlen=buff_size)
            self.alpha = 0.9

    def update_features(self, feat):
        # L2 normalizing, this function has no use for BYTETracker
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha
                                                                ) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state,
                                                                self.covariance)

    @staticmethod
    def multi_predict(tracks, kalman_filter):
        if len(tracks) > 0:
            multi_mean = np.asarray([track.mean.copy() for track in tracks])
            multi_covariance = np.asarray(
                [track.covariance for track in tracks])
            for i, st in enumerate(tracks):
                if st.state != TrackState.Tracked:
                    if isinstance(kalman_filter,OCKalmanFilter):
                        if multi_mean[i][6] + multi_mean[i][2] < 0:
                            multi_mean[i][6] = 0
                    elif isinstance(kalman_filter,KalmanFilter):
                        multi_mean[i][7] = 0
                    else:
                        raise ValueError("Kalman filter must be either a "
                                         "KalmanFilter instance or a subclass "
                                         "of it.")
            multi_mean, multi_covariance = kalman_filter.multi_predict(
                multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def reset_track_id(self):
        self.reset_track_count(self.cls_id)

    def activate(self, kalman_filter, frame_id):
        """Start a new track"""
        self.kalman_filter = kalman_filter
        # update track id for the object class
        self.track_id = self.next_id(self.cls_id)
        if isinstance(self.kalman_filter,OCKalmanFilter):
            self.mean, self.covariance = self.kalman_filter.initiate(
                self.tlwh_to_xysr(self._tlwh))
        elif isinstance(self.kalman_filter,KalmanFilter):
            self.mean, self.covariance = self.kalman_filter.initiate(
                self.tlwh_to_xyah(self._tlwh))

        self.track_len = 0
        self.state = TrackState.Tracked  # set flag 'tracked'

        if frame_id == 1:  # to record the first frame's detection result
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id
        self.update_history()

    def re_activate(self, new_track, frame_id, new_id=False, interpolation='linear'):
        time_gap = frame_id - self.frame_id
        if interpolation == 'linear' and time_gap > 20:
            ## 线性插值
            box_1 = self.tlwh
            x1,y1,w1,h1 = box_1
            box_2 = new_track.tlwh
            x2,y2,w2,h2 = box_2
            dx = (x2-x1)/time_gap
            dy = (y2-y1)/time_gap
            dw = (w2-w1)/time_gap
            dh = (h2-h1)/time_gap

            
            for i in range(time_gap):
                x1+=dx
                y1+=dy
                w1+=dw
                h1+=dh
                new_box = np.array((x1,y1,w1,h1),dtype=np.float32)
                # print(new_box)
                if isinstance(self.kalman_filter,OCKalmanFilter):
                    self.mean, self.covariance = self.kalman_filter.update(
                        self.mean, self.covariance, self.tlwh_to_xysr(new_box))
                elif isinstance(self.kalman_filter,KalmanFilter):
                    self.mean, self.covariance = self.kalman_filter.update(
                        self.mean, self.covariance, self.tlwh_to_xyah(new_box))
        elif interpolation == 'gaussian':
            tau = 10
            fid_list = list(self._history.keys())
            # print(frame_id)
            t = [self._history[fid][0] for fid in self._history.keys()]
            l = [self._history[fid][1] for fid in self._history.keys()]
            w = [self._history[fid][2] for fid in self._history.keys()]
            h = [self._history[fid][3] for fid in self._history.keys()]
            fid_list.append(frame_id)
            t.append(new_track.tlwh[0])
            l.append(new_track.tlwh[1])
            w.append(new_track.tlwh[2])
            h.append(new_track.tlwh[3])
            fid_list = np.array(fid_list).reshape(-1, 1)
            # t = np.array(t).reshape(-1, 1)
            # l = np.array(l).reshape(1, -1)
            # w = np.array(w).reshape(1, -1)
            # h = np.array(h).reshape(1, -1)
            len_scale = np.clip(tau * np.log(tau ** 3 / len(fid_list)+1), tau ** -1, tau ** 2)
            time_gap = frame_id - self.frame_id -1
            pred_frame_id = np.arange(self.frame_id+1, frame_id).reshape(-1, 1)
            gpr = GPR(RBF(len_scale, 'fixed'))
            # print(fid_list)
            # print(pred_frame_id)
            gpr.fit(fid_list, t)
            tt = gpr.predict(pred_frame_id)
            # print(t)
            # print(tt)
            gpr.fit(fid_list, l)
            ll = gpr.predict(pred_frame_id)
            gpr.fit(fid_list, w)
            ww = gpr.predict(pred_frame_id)
            gpr.fit(fid_list, h)
            hh = gpr.predict(pred_frame_id)
            print("*"*100)
            print(self.tlwh)
            print(self.mean)
            print(new_track.tlwh)
            print("*"*100)
            # print(time_gap)
            for i in range(time_gap):
                # print(i)
                new_tlwh = np.array((tt[i],ll[i],ww[i],hh[i]),dtype=np.float32)
                print(new_tlwh)
                if isinstance(self.kalman_filter,OCKalmanFilter):
                    self.mean, self.covariance = self.kalman_filter.update(
                        self.mean, self.covariance, self.tlwh_to_xysr(new_tlwh))
                elif isinstance(self.kalman_filter,KalmanFilter):
                    self.mean, self.covariance = self.kalman_filter.update(
                        self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
                # print(new_box)
            # print(new_track.tlwh)
            # exit()
                
            
            
            if isinstance(self.kalman_filter,OCKalmanFilter):
                self.mean, self.covariance = self.kalman_filter.update(
                    self.mean, self.covariance, self.tlwh_to_xysr(new_track.tlwh))
            elif isinstance(self.kalman_filter,KalmanFilter):
                self.mean, self.covariance = self.kalman_filter.update(
                    self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        else:
            if isinstance(self.kalman_filter,OCKalmanFilter):
                self.mean, self.covariance = self.kalman_filter.update(
                    self.mean, self.covariance, self.tlwh_to_xysr(new_track.tlwh))
            elif isinstance(self.kalman_filter,KalmanFilter):
                self.mean, self.covariance = self.kalman_filter.update(
                    self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        if self.use_reid:
            self.update_features(new_track.curr_feat)
        self.track_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:  # update track id for the object class
            self.track_id = self.next_id(self.cls_id)

    def update(self, new_track, frame_id, update_feature=True):
        self.frame_id = frame_id
        self.track_len += 1

        new_tlwh = new_track.tlwh
        self.update_velocitie(new_tlwh)
        if isinstance(self.kalman_filter,OCKalmanFilter):
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xysr(new_tlwh))
        elif isinstance(self.kalman_filter,KalmanFilter):
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked  # set flag 'tracked'
        self.is_activated = True  # set flag 'activated'
        self.update_history()

        self.score = new_track.score
        if update_feature and self.use_reid:
            self.update_features(new_track.curr_feat)
            
    def update_history(self):
        self._history[self.frame_id]=(self.tlwh)
        
        
    def update_velocitie(self,new_box):
        self.velocity = matching.speed_direction(self.tlwh,new_box)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        if isinstance(self.kalman_filter,OCKalmanFilter):
            w = np.sqrt(ret[2] * ret[3])
            h = ret[2] / w + 1e-6
            ret[3] = h
            ret[2] = w 
            ret[:2] -= ret[2:] / 2
        elif isinstance(self.kalman_filter,KalmanFilter):
            ret[2] *= ret[3]
            ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    @staticmethod
    def tlwh_to_xysr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] *= ret[3]
        ret[3] = ret[2]/(ret[3]*ret[3]+1e-6)
        return ret
    
    def to_xysr(self):
        return self.tlwh_to_xysr(self.tlwh)
        

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_({}-{})_({}-{})'.format(self.cls_id, self.track_id,
                                           self.start_frame, self.end_frame)


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
