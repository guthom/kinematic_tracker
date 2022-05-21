from typing import Optional, Tuple
from kinematic_tracker.pose_models.KinematicChainBase import KinematicChainBase
from kinematic_tracker.tracking.TrackingDescriptor import TrackingDescriptor
from datetime import datetime

class TrackingInformation(object):

    def __init__(self, kinematicChain: KinematicChainBase, imageSize: Tuple[int, int], descriptor: Optional[TrackingDescriptor] = None):
        self.kinematicChain: KinematicChainBase = kinematicChain

        if descriptor is None:
            self.descriptor = TrackingDescriptor(kinematicChain, imageSize)
        else:
            self.descriptor: TrackingDescriptor = descriptor

        self.creatingTime = datetime.now()

