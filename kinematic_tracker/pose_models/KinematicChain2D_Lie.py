from typing import Dict, List, Optional
import numpy as np
from guthoms_helpers.base_types.Pose3D import Pose3D
from guthoms_helpers.base_types.Pose2D import Pose2D
from kinematic_tracker.pose_models import Joint
from guthoms_helpers.common_helpers.ColorHelper import ColorHelper
from guthoms_helpers.base_types.BoundingBox2D import BoundingBox2D
from guthoms_helpers.base_types.Rotation2D import Rotation2D
import math
from kinematic_tracker.pose_models.KinematicChainBase import KinematicChainBase
from kinematic_tracker.pose_models.KinematicChain2D import KinematicChain2D
from guthoms_helpers.base_types.Complex import Complex as MyComplex
from sophus import Se2, So2, se2, so2, Vector2, Vector3, Complex
from copy import deepcopy
import cv2

# basically just a wrapper for my classes and the the Sophus library https://github.com/strasdat/Sophus
# i started to implement a group representation in the guthoms_helpers package as well. however, its not finished yet

class KinematicChain2D_Lie(object):

    def __init__(self, baseChain: KinematicChain2D):

        self.baseChain: KinematicChain2D = baseChain

        self.group: List[Se2] = self.ConvertStandardChain(baseChain)

        pass

    @staticmethod
    def ConvertStandardChain(baseChain: KinematicChain2D) -> List[Se2]:

        ret = []
        for i in range(0, baseChain.poses.__len__()):
            pose = baseChain.poses[i]
            if i > 0:
                poseBefore = baseChain.poses[i-1]
                transDiff = pose.trans - poseBefore.trans
                trans: Vector2(transDiff[0], transDiff[1])

                rot = Rotation2D.From2Vectors(poseBefore.trans, pose.trans)
                myComp = rot.ToComplex()
                comp: MyComplex = MyComplex(myComp.r, myComp.i)
            else:
                comp: MyComplex = MyComplex(1.0, 0) #angle = 0.0
                trans: Vector2(pose.trans[0], pose.trans[1])

            groupElement = Se2(So2(Complex(comp.r, comp.i)), Vector2(pose.trans[0], pose.trans[1]))
            ret.append(groupElement)

        return ret

    def ToStandardChain(self) -> KinematicChain2D:

        poses: List[Pose2D] = []
        for i in range(0, self.group.__len__()):
            pose = self.group[i]

            #trans =


        ret = deepcopy(self.baseChain)

        ret.UpdatePoses(poses)
        return ret


