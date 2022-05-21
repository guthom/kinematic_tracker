from guthoms_helpers.base_types.Pose3D import Pose3D
from guthoms_helpers.base_types.Pose2D import Pose2D

class Joint(object):

    def __init__(self, name: str, index: int):
        self.name: str = name
        self.index: int = index

        self.poses2D: Pose2D = None
        self.poses3D: Pose3D = None

    def Update2D(self, pose: Pose2D):
        self.poses2D = pose

    def Update3D(self, pose: Pose3D):
        self.poses3D = pose
