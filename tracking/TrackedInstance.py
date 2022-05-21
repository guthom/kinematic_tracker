import uuid
from datetime import datetime

from guthoms_helpers.common_stuff.DataBuffer import DataBuffer
import copy
from kinematic_tracker.tracking.TrackingInformation import TrackingInformation

from typing import Optional

class TrackedInstance(object):
    def __init__(self, trackingInfos: Optional[TrackingInformation] = None):
        self.uuid = uuid.uuid4()
        self.currentlyObserved = False

        self.lastUpdate = None
        self.trackingInfos: TrackingInformation = trackingInfos

        #create databuffer with defined length 100 to store tracked history of instance
        self.history: DataBuffer = DataBuffer(100)

        if trackingInfos is not None:
            self.Update(trackingInfos)

    def Valid(self, invalidationMicroSeconds: int=1000) -> bool:
        secondsPast = (datetime.now() - self.lastUpdate).microseconds * 3e-3
        valid = secondsPast <= invalidationMicroSeconds
        return valid

    def GetHistory(self, amount: int = 1):
        return self.history.GetLastItems(amount)

    def Update(self, trackingInfos: TrackingInformation):
        self.trackingInfos = trackingInfos
        self.history.append(copy.deepcopy(self.trackingInfos))
        self.lastUpdate = datetime.now()
