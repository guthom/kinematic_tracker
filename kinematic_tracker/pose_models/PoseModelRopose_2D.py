from typing import Union, List
from kinematic_tracker.pose_models.PoseModelBase import PoseModelBase
from kinematic_tracker.pose_models.KinematicChain2D import KinematicChain2D
from guthoms_helpers.base_types.Pose2D import Pose2D
from guthoms_helpers.base_types.Pose3D import Pose3D
from guthoms_helpers.common_helpers.ColorHelper import ColorHelper
from kinematic_tracker.pose_models.ModelMappings import ModelMappings

class PoseModelRopose_2D(PoseModelBase):


    def __init__(self, mapping: ModelMappings=ModelMappings.CHAIN):

        linkNames = ["base_link",       #Keypoint 0
                     "shoulder_link",   #Keypoint 1
                     "upper_arm_link",  #Keypoint 2
                     "forearm_link",    #Keypoint 3
                     "wrist_1_link",    #Keypoint 4
                     "wrist_2_link",    #Keypoint 5
                     "wrist_3_link"]    #Keypoint 6

        kinematicChain = KinematicChain2D(chainName="robot", linkNames=linkNames,
                                          linkColors=ColorHelper.GetUniqueColors(7))

        super(PoseModelRopose_2D, self).__init__(kinematicChain, mapping=mapping)

    def PosesToChain(self, poses: List[Pose2D], mapping: ModelMappings) -> List[Pose2D]:

        if mapping is not ModelMappings.CHAIN:
            raise Exception("Mapping is not yet implemented!")

        return poses

    def ChainToPoses(self, mapping: ModelMappings) -> List[Pose2D]:
        if mapping is not ModelMappings.CHAIN:
            raise Exception("Mapping is not yet implemented!")

        return self.kinematicChain.GetPoseList()
