from abc import ABC, abstractmethod, abstractclassmethod
from typing import Union, Optional

from kinematic_tracker.pose_models.KinematicChain2D import KinematicChain2D
from kinematic_tracker.pose_models.KinematicChain3D import KinematicChain3D
from kinematic_tracker.pose_models.ModelMappings import ModelMappings
from guthoms_helpers.base_types.Pose2D import Pose2D
from guthoms_helpers.base_types.Pose3D import Pose3D
import numpy as np
from typing import List

class PoseModelBase(ABC):

    def __init__(self, kinematicChain: Union[KinematicChain2D, KinematicChain3D], mapping: ModelMappings):

        self.kinematicChain = kinematicChain
        self.modelMapping = mapping

        self.jointCount = kinematicChain.CountJoints()

    def ToNumpy(self) -> np.array:
        return self.kinematicChain.ToNumpy()

    def GetPoses(self, mapping: Optional[ModelMappings]=ModelMappings.CHAIN) -> Union[List[Pose2D], List[Pose3D]]:
        if mapping is None:
            mapping = self.modelMapping

        return self.ChainToPoses(mapping)

    def UpdatePoses(self, poses: Union[List[Pose2D], List[Pose3D]], mapping: Optional[ModelMappings]=None):
        if mapping is None:
            mapping = self.modelMapping

        self.kinematicChain.UpdatePoses(self.PosesToChain(poses, mapping))

    def Draw(self, image: Union[np.array, None]) -> np.array:
        return self.kinematicChain.Draw(image)

    @abstractmethod
    def PosesToChain(self, poses: Union[List[Pose2D], List[Pose3D]], mapping: ModelMappings) -> \
            Union[List[Pose2D], List[Pose3D]]:
        raise Exception("Not Implemented")

    @abstractmethod
    def ChainToPoses(self, mapping: ModelMappings) -> \
            Union[List[Pose2D], List[Pose3D]]:
        raise Exception("Not Implemented")

