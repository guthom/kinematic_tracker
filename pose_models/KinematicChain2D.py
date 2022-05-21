from typing import Dict, List, Optional
import numpy as np
from guthoms_helpers.base_types.Pose3D import Pose3D
from guthoms_helpers.base_types.Pose2D import Pose2D
from kinematic_tracker.pose_models import Joint
from guthoms_helpers.common_helpers.ColorHelper import ColorHelper
from guthoms_helpers.base_types.BoundingBox2D import BoundingBox2D
import math
from kinematic_tracker.pose_models.KinematicChainBase import KinematicChainBase


import cv2

class KinematicChain2D(KinematicChainBase):

    def __init__(self, chainName: str, linkNames: List[str] = None, partChaines: List['KinematicChain2D']=None,
                 linkCount: int = None, poses: Pose2D=None, linkColors: List = None, closed: bool=False):

        super(KinematicChain2D, self).__init__(chainName=chainName, linkNames=linkNames, partChaines=partChaines,
                                               linkCount=linkCount, poses=poses, linkColors=linkColors, closed=closed)

    def Draw(self, rawImage: np.array) -> np.array:

        subchains = self.GetSubChains()

        #delte itself
        del subchains[0]

        for chain in subchains:
            chain.Draw(rawImage)

        poses = self.poses
        linkColors = self.linkColors

        # draw closed chain and enlarge lists with a additional entry
        if self.closedChain:
            poses.append(poses[0])
            linkColors.append(self.linkColors[-1])

        for i in range(0, poses.__len__()):
            if not poses[i].visible:
                continue

            if not math.isnan(poses[i][0]) and not math.isnan(poses[i][1]):
                # draw pose circles
                position = (int(poses[i][0]), int(poses[i][1]))
                rawImage = cv2.circle(rawImage, position, 5, linkColors[i], thickness=-1)

        for i in range(0, poses.__len__() - 1):
            if not poses[i].visible:
                continue
            if not math.isnan(poses[i][0]) and not math.isnan(poses[i][1]) and not \
                    math.isnan(poses[i + 1][0]) and not math.isnan(poses[i + 1][1]) \
                    and poses[i][0] != -1 and poses[i][1] != -1 \
                    and poses[i + 1][0] != -1 and poses[i + 1][1] != -1:
                length = ((poses[i][1] - poses[i + 1][1]) ** 2 + (poses[i][0] - poses[i + 1][0]) ** 2) ** 0.5

                angle = math.degrees(math.atan2(poses[i][1] - poses[i + 1][1], poses[i][0] - poses[i + 1][0]))

                meanX = (poses[i][1] + poses[i + 1][1]) / 2
                meanY = (poses[i][0] + poses[i + 1][0]) / 2

                polygon = cv2.ellipse2Poly((int(meanY), int(meanX)), (int(length / 2), 2), int(angle), 0,
                                           360, 1)

                cv2.fillConvexPoly(rawImage, polygon, linkColors[i])

        return rawImage

    def GetBoundingBox(self, extend: bool = True) -> BoundingBox2D:
        poseList = self.GetTranslationList()
        bb = BoundingBox2D.CreateBoundingBox(poseList, expandBox=extend)
        bb = bb.ClipMin(0.0)
        return bb

    def GetDescriptor(self) -> np.array:
        raise Exception("Not Implemented!")
