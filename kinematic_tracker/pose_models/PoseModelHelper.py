from typing import List, Union
from guthoms_helpers.base_types.PoseBase import PoseBase

import numpy as np

class PoseModelHelper(object):

    @staticmethod
    def PoseListToNumpy(poses: List[PoseBase]) -> np.array:

        rawList = []

        for pose in poses:
            rawList.append(pose.toNp())

        return np.array(rawList)
