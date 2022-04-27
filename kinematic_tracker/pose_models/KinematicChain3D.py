from typing import Dict, List, Union
import numpy as np
from guthoms_helpers.base_types.Pose3D import Pose3D
from guthoms_helpers.base_types.BoundingBox3D import BoundingBox3D
from kinematic_tracker.pose_models import Joint

from kinematic_tracker.pose_models.KinematicChainBase import KinematicChainBase

import cv2

class KinematicChain3D(KinematicChainBase):

    def __init__(self, chainName: str, linkNames: List[str] = None, partChaines: List['KinematicChain3D']=None,
                 linkCount: int = None, poses: Pose3D=None, linkColors: List = None, closed:bool = False):

        super(KinematicChain3D, self).__init__(chainName=chainName, linkNames=linkNames, partChaines=partChaines,
                                               linkCount=linkCount, poses=poses, linkColors=linkColors, closed=closed)

    def ToNumpy(self) -> np.array:
        raise Exception("Not Implemented")

    def Draw(self, target: Union[np.array, None]) -> Union[np.array, None]:
        raise Exception("Not Implemented!")

    def GetBoundingBox(self, extend: bool = False) -> BoundingBox3D:
        raise Exception("Not Implemented!")

    def GetDescriptor(self) -> np.array:
        raise Exception("Not Implemented!")
