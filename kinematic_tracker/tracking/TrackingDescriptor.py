from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Tuple, Optional, List
from kinematic_tracker.tracking.DescriptorSet import DescriptorSet
from kinematic_tracker.pose_models.KinematicChainBase import KinematicChainBase
from guthoms_helpers.base_types.PoseBase import PoseBase
from guthoms_helpers.base_types.VectorBase import VectorBase
from guthoms_helpers.base_types.RotationBase import RotationBase
from guthoms_helpers.base_types.Rotation2D import Rotation2D
import numpy as np
import math
from typing import Tuple

class TrackingDescriptor(ABC):

    def __init__(self, kinematicChain: KinematicChainBase, imageSize: Tuple[int, int]):
        self.kinematicChain: KinematicChainBase = kinematicChain
        self.bbWeigth = 0.25

        self.imageSize = imageSize
        self.maxDistance = math.sqrt(math.pow(imageSize[0], 2) + math.pow(imageSize[1], 2))

        #RawDescriptorValues
        self.descriptorSets: List[DescriptorSet] = self.CalculatePoseDescriptor(self.kinematicChain)


    def CalculateSingleDescriptorFromPoses(self, pose1: PoseBase, pose2: PoseBase):
        relativeAngle = None
        relativeDistance = None

        if pose1.visible and pose2.visible:
            relativeAngle = pose1.rotation.From2Vectors(pose1.trans, pose2.trans)
            relativeDistance = pose1.trans.Distance(pose2.trans) / self.maxDistance

        return relativeAngle, relativeDistance


    def CalculatePoseDescriptor(self, kinematicChain: KinematicChainBase) -> List[DescriptorSet]:

        descriptors = []
        subChains = kinematicChain.GetSubChains()

        for chain in subChains:
            descriptor = DescriptorSet()
            #append bounding box
            descriptor.boundingBox = chain.GetBoundingBox()
            descriptor.boundingBox = descriptor.boundingBox.NormWithSize(self.imageSize)


            #special case when we have chains with just one link (like nose etc)
            if chain.poses.__len__() == 1:
                descriptor.relativeAngles.append(chain.poses[0].rotation.Empty())
                descriptor.relativeDistances.append(0.0)
            else:
                for i in range(0, chain.poses.__len__()):
                    pose1 = chain.poses[i]

                    for j in range(0, chain.poses.__len__()):
                        if j < i:
                            pose2 = chain.poses[j]

                            relativeAngle, relativeDistance = self.CalculateSingleDescriptorFromPoses(pose1, pose2)

                            descriptor.relativeAngles.append(relativeAngle)
                            descriptor.relativeDistances.append(relativeDistance)

            descriptors.append(descriptor)

        return descriptors

    def ComparePose(self, other: 'TrackingDescriptor') -> float:
        similarities = []

        if self.descriptorSets.__len__() != other.descriptorSets.__len__():
            raise Exception("Error, descriptor lengths not matching!!")

        for i in range(0, self.descriptorSets.__len__()):
            similarities.append(self.descriptorSets[i].ComparePose(other.descriptorSets[i]))

        return float(np.mean(similarities))

    def CompareBB(self, other: 'TrackingDescriptor') -> float:
        similarities = []

        if self.descriptorSets.__len__() != other.descriptorSets.__len__():
            return 0.0

        for i in range(0, self.descriptorSets.__len__()):
            similarities.append(self.descriptorSets[i].CompareBB(other.descriptorSets[i]))

        return float(np.mean(similarities))

    def CompareAll(self, other: 'TrackingDescriptor') -> float:
        pose = self.ComparePose(other)
        bb = self.CompareBB(other)

        return (1 - self.bbWeigth)*pose + self.bbWeigth*bb
        #return pose

    def ToVectorRepresentation(self) -> np.array:
        sets = []

        for set in self.descriptorSets:
            sets.extend(set.ToVectorRepresentation())

        return np.array(sets)

    def __str__(self):
        return str(self.descriptorSets)
