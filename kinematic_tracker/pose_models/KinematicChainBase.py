from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

from guthoms_helpers.base_types.PoseBase import PoseBase
from guthoms_helpers.base_types.VectorBase import VectorBase
from guthoms_helpers.base_types.BoundingBoxBase import BoundingBoxBase
from guthoms_helpers.base_types.OpticalFlow import OpticalFlow
from guthoms_helpers.common_helpers.ColorHelper import ColorHelper

from kinematic_tracker.pose_models.PoseModelHelper import PoseModelHelper

import cv2

class KinematicChainBase(ABC):

    def __init__(self, chainName: str, linkNames: List[str] = None, partChaines: List['KinematicChainBase']=None,
                 linkCount: int = None, poses: PoseBase=None, linkColors: List = None, closed: bool = False):

        self.chainName: str = chainName
        self.poses: List[PoseBase] = poses
        self.closedChain = closed

        if partChaines is not None:
            for partChain in partChaines:
                self.AddChain(partChain)

        self.linkCount = 0
        if linkCount is None:
            if linkNames is not None:
                self.linkCount: int = linkNames.__len__()
        else:
            self.linkCount = linkCount

        if linkNames is not None:
            self.linkNames: List[str] = linkNames

        if linkColors is None:
            if self.closedChain:
                self.linkColors = ColorHelper.GetUniqueColors(self.linkCount+1)
            else:
                self.linkColors = ColorHelper.GetUniqueColors(self.linkCount)

        else:
            self.linkColors: List[List[float, float, float]] = linkColors

            if self.closedChain and self.linkColors.__len__() < self.linkCount+1:
                #append the last color again to allow for closed chains
                self.linkColors.append(self.linkColors[-1])


        self.linkedChains: Dict[str, List[KinematicChainBase]] = dict()
        self.unlinkedChains: List[KinematicChainBase] = []

        self.InitChain()
        self.partCount = self.CountJoints()

    def __len__(self):
        return self.partCount

    @abstractmethod
    def Draw(self, target: Union[np.array, None]) -> Union[np.array, None]:
        raise Exception("Not Implemented!")

    @abstractmethod
    def GetDescriptor(self) -> np.array:
        raise Exception("Not Implemented!")

    @abstractmethod
    def GetBoundingBox(self, extend: bool = True) -> BoundingBoxBase:
        raise Exception("Not Implemented!")

    def InvalidatePoses(self):
        for chain in self.GetSubChains():
            chain.InvalidatePoses()

        for pose in self.poses:
            pose.visible = False

    def HasSubchains(self) -> bool:

        if self.GetSubChains().__len__() > 0:
            return True

        return False

    def MovePositions(self, diffs: List[VectorBase]) -> 'KinematicChainBase':
        for i in range(0, self.poses.__len__()):
            self.poses[i] += diffs[i]

        return self


    def ToNumpy(self) -> np.array:
        return PoseModelHelper.PoseListToNumpy(self.GetPoseList())

    def AddChain(self, kinematicChain: 'KinematicChainBase',  linkName: str = None):
        if linkName is not None:

            if linkName not in self.linkNames:
                raise Exception("Linkname is not known in base chain!")

            self.linkedChains[linkName].append(kinematicChain)
        else:
            self.unlinkedChains.append(kinematicChain)

        self.partCount = self.CountJoints()

    def InitChain(self):
        #init link chains
        for linkName in self.linkNames:
            self.linkedChains[linkName] = []

    def CountJoints(self) -> int:
        count = self.linkCount

        for joint in self.linkedChains:
            for chain in self.linkedChains[joint]:
                count += chain.CountJoints() - 1

        for chain in self.unlinkedChains:
            count += chain.CountJoints()

        return count

    def CalculateOpticalFlow(self, other: 'KinematicChainBase') -> List[OpticalFlow]:

        flows: List[OpticalFlow] = []

        for i in range(0, self.poses.__len__()):
            flow = self.poses[i].trans.CalculateFlow(other.poses[i].trans)
            flows.append(flow)

        return flows

    def UpdatePoses(self, poses: List[PoseBase]):
        if poses.__len__() != self.__len__():
            raise Exception("poses count (" + str(poses.__len__()) + ") of chain (" + self.chainName +
                            ") is not equal to the pose model (" + str(self.__len__()) + ")!")

        #fill base kinematic chain
        self.poses = poses[0:self.linkCount]

        #run up all connected chains according to the position in the chain
        currentPosition = self.linkCount
        for i in range(0, self.linkNames.__len__()):
            chains = self.linkedChains[self.linkNames[i]]
            for chain in chains:
                end = currentPosition + chain.__len__() - 1
                newPoses = []
                newPoses.append(self.poses[i])
                newPoses.extend(poses[currentPosition:end])
                chain.UpdatePoses(newPoses)
                currentPosition = end

        for chain in self.unlinkedChains:
            end = currentPosition + chain.__len__()
            chain.UpdatePoses(poses[currentPosition:end])
            currentPosition = end

    def GetSubChains(self) -> List['KinematicChainBase']:
        subChains = []

        subChains.append(self)

        for chain in self.unlinkedChains:
            subChains.extend(chain.GetSubChains())

        for key in self.linkedChains:
            for chain in self.linkedChains[key]:
                subChains.extend(chain.GetSubChains())

        return subChains

    def GetPoseList(self) -> List[PoseBase]:
        poseList = []
        # fill base kinematic chain
        poseList.extend(self.poses)

        # run up all connected chains according to the position in the chain
        for i in range(0, self.linkNames.__len__()):
            chains = self.linkedChains[self.linkNames[i]]
            for chain in chains:
                poseList.extend(chain.GetPoseList()[1:])

        for chain in self.unlinkedChains:
            poseList.extend(chain.GetPoseList())

        return poseList

    def GetTranslationList(self) ->List[VectorBase]:
        poseList = []
        # fill base kinematic chain
        for pose in self.poses:
            poseList.append(pose.trans)

        # run up all connected chains according to the position in the chain
        for i in range(0, self.linkNames.__len__()):
            chains = self.linkedChains[self.linkNames[i]]
            for chain in chains:
                poseList.extend(chain.GetTranslationList()[1:])

        for chain in self.unlinkedChains:
            poseList.extend(chain.GetTranslationList())

        return poseList