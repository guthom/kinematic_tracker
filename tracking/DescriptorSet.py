from guthoms_helpers.base_types.VectorBase import VectorBase
from guthoms_helpers.base_types.RotationBase import RotationBase
from guthoms_helpers.base_types.BoundingBoxBase import BoundingBoxBase
import numpy as np
import math
from typing import List, Optional, Tuple

class DescriptorSet(object):

    def __init__(self):

        #RawDescriptorValues
        self.relativeAngles = list()
        self.relativeDistances = list()
        self.boundingBox = None

    @staticmethod
    def CompareRotations(rotations1: List[RotationBase], rotations2: List[RotationBase]):
        tempList = []

        #return hard zero when chain lengt does not match
        if rotations1.__len__() != rotations2.__len__():
            return [0.0]

        for i in range(0, rotations1.__len__()):
            if rotations1[i] is not None and rotations2[i] is not None:
                distance = rotations1[i].DistanceNorm(rotations2[i])
                distance = abs(distance)/np.pi

                similarity = 1 - distance

                tempList.append(similarity)

        if tempList.__len__() == 0:
            tempList.append(0.0)

        return np.mean(tempList)

    @staticmethod
    def CompareDistances(distances1: List[float], distances2: List[float], boundingBox1: BoundingBoxBase,
                         boundingBox2: BoundingBoxBase):
        tempList = []

        #return hard zero when chain lengt does not match
        if distances1.__len__() != distances2.__len__():
            return [0.0]

        #get bigger bounding box:

        area1 = boundingBox1.Area()
        area2 = boundingBox2.Area()
        areaNorm = math.sqrt(np.mean([area1, area2]))

        invalidCounter = 0
        for i in range(0, distances1.__len__()):
            if distances1[i] is not None and distances2[i] is not None:
                distance = distances1[i] - distances2[i]

                if areaNorm != 0.0:
                    similarity = 1 - math.sqrt((math.pow(distance, 2) / areaNorm))
                else:
                    similarity = 0.0

                tempList.append(similarity)
            else:
                invalidCounter += 1

        if tempList.__len__() == 0:
            tempList.append(0.0)

        return np.mean(tempList)

    def ComparePose(self, other: 'DescriptorSet'):
        #compare relativeAngles
        relativeAngleDiff = self.CompareRotations(self.relativeAngles, other.relativeAngles)
        relativeDistanceDiff = self.CompareDistances(self.relativeDistances, other.relativeDistances,
                                                     self.boundingBox, other.boundingBox)

        return np.mean([np.mean(relativeAngleDiff), np.mean(relativeDistanceDiff)])

    def ToVectorRepresentation(self) -> np.array:
        ret = []
        '''
        bbVectors = self.boundingBox.toList()
        for vec in bbVectors:
            ret.extend(vec)
            pass
        '''
        for ang in self.relativeAngles:
            try:
                ret.extend(ang.ToComplex().toList())
            except:
                ret.append([None, None])

        ret.extend(self.relativeDistances)

        return ret

    def CompareBB(self, other: 'DescriptorSet'):
        return self.boundingBox.CalculateIoU(other.boundingBox)

    def __str__(self):
        return str(self.relativeAngles) + " | " + str(self.relativeDistances) + " | " +  \
               str(self.baseAngles) + " | " + str(self.baseDistances) + " | " + str(self.boundingBoxes)