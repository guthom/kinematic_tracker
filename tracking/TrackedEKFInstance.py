import uuid
from datetime import datetime

from guthoms_helpers.common_stuff.DataBuffer import DataBuffer
import copy
from kinematic_tracker.tracking.TrackingInformation import TrackingInformation
from guthoms_helpers.signal_processing.KalmanFilter import KalmanState, KalmanFilter
import numpy as np

from typing import Optional, List

class TrackedEKFInstance(object):
    def __init__(self, trackingInfos: Optional[TrackingInformation] = None):
        self.uuid = uuid.uuid4()
        self.currentlyObserved = False

        self.lastUpdate = None
        self.trackingInfos: TrackingInformation = trackingInfos

        self.poseKalmans: List[KalmanFilter] = []

        self.InitKalmans(trackingInfos)

        #create databuffer with defined length 100 to store tracked history of instance
        self.history: DataBuffer = DataBuffer(100)

        if trackingInfos is not None:
            self.Update(trackingInfos)

    def InitKalmans(self, trackingInfos: TrackingInformation):

        poses = trackingInfos.kinematicChain.GetPoseList()
        for i in range(0, poses.__len__()):
            self.poseKalmans.append(KalmanFilter(2, R=np.array([[0.25], [0.25]]), X_0=poses[i].toNp()))

    def PredictBB(self):
        pass

    def PredictPoses(self):
        pass

    def Valid(self, invalidationMicroSeconds: int=1000) -> bool:
        secondsPast = (datetime.now() - self.lastUpdate).microseconds * 3e-3
        valid = secondsPast <= invalidationMicroSeconds
        return valid

    def GetHistory(self, amount: int = 1):
        return self.history.GetLastItems(amount)

    def IterateKalmans(self, trackingInfos: TrackingInformation):
        poses = trackingInfos.kinematicChain.GetPoseList()
        for i in range(0, poses.__len__()):
            state = self.poseKalmans[i].AddMeasurement(Y_m=poses[i].toNp())
            pose = type(trackingInfos.kinematicChain.poses[i])(state.X_state[0], state.X_state[0], 0)
            trackingInfos.kinematicChain.poses[i] = pose

    def Update(self, trackingInfos: TrackingInformation):
        self.trackingInfos = trackingInfos
        self.history.append(copy.deepcopy(self.trackingInfos))
        self.lastUpdate = datetime.now()
        self.IterateKalmans(trackingInfos)
