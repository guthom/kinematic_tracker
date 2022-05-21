from typing import Union, List
from kinematic_tracker.pose_models.PoseModelBase import PoseModelBase
from kinematic_tracker.pose_models.KinematicChain2D import KinematicChain2D
from guthoms_helpers.base_types.PoseBase import PoseBase
from kinematic_tracker.pose_models.ModelMappings import ModelMappings
from guthoms_helpers.common_helpers.ColorHelper import ColorHelper

class PoseModelHuman17_2D(PoseModelBase):
    '''
    Complete Keypoint structure
    linkNames = ["nose",            #keypoint 0
                 "left_eye",        #keypoint 1
                 "right_eye",       #keypoint 2
                 "left_ear",        #keypoint 3
                 "right_ear",       #keypoint 4
                 "left_shoulder",   #keypoint 5
                 "right_shoulder",  #keypoint 6
                 "left_elbow",      #keypoint 7
                 "right_elbow",     #keypoint 8
                 "left_wrist",      #keypoint 9
                 "right_wrist",     #keypoint 10
                 "left_hip",        #keypoint 11
                 "right_hip",       #keypoint 12
                 "left_knee",       #keypoint 13
                 "right_knee",      #keypoint 14
                 "left_ankle",      #keypoint 15
                 "right_ankle",     #keypoint 16
                 ]
    '''

    def __init__(self, mapping: ModelMappings=ModelMappings.COCO):

        linkColors = ColorHelper.GetUniqueColors(17)

        torsoNames = [
            "left_shoulder",
            "left_hip",
            "right_hip",
            "right_shoulder"
            ]
        completeChain = KinematicChain2D("human", torsoNames, closed=True, linkColors=linkColors[0:4])

        leftArmNames = [
            "left_shoulder",  # keypoint 5
            "left_elbow",  # keypoint 7
            "left_wrist",  # keypoint 9
        ]

        leftArm = KinematicChain2D("leftArm", leftArmNames, linkColors=[linkColors[0],
                                                                        linkColors[4],
                                                                        linkColors[5]])

        rightArmNames = [
            "right_shoulder",  #keypoint 6
            "right_elbow",     #keypoint 8
            "right_wrist",     #keypoint 10
            ]
        rightArm = KinematicChain2D("rightArm", rightArmNames, linkColors=[linkColors[4],
                                                                           linkColors[6],
                                                                           linkColors[7]])

        leftLegNames = [
            "left_hip",        #keypoint 11
            "left_knee",       #keypoint 13
            "left_ankle",      #keypoint 15
            ]
        leftLeg = KinematicChain2D("leftLeg", leftLegNames, linkColors=[linkColors[1],
                                                                        linkColors[8],
                                                                        linkColors[9]])

        rightLegNames = [
            "right_hip",       #keypoint 12
            "right_knee",      #keypoint 14
            "right_ankle",     #keypoint 16
            ]
        rightLeg = KinematicChain2D("rightLeg", rightLegNames, linkColors=[linkColors[2],
                                                                        linkColors[10],
                                                                        linkColors[11]])

        # define headsegment
        headNames = ["nose"]  # keypoint 0
        head = KinematicChain2D("head_base", headNames, linkColors=[linkColors[12]])

        headLeftNames = [
            "nose",  # keypoint 0
            "left_eye",  # keypoint 1
            "left_ear"  # keypoint 3
        ]
        headLeft = KinematicChain2D("headLeft", headLeftNames, linkColors=[linkColors[12],
                                                                           linkColors[13],
                                                                           linkColors[14]])

        headRightNames = [
            "nose",  # keypoint 0
            "right_eye",  # keypoint 2
            "right_ear"  # keypoint 4
        ]
        headRight = KinematicChain2D("headRight", headRightNames, linkColors=[linkColors[12],
                                                                              linkColors[15],
                                                                              linkColors[16]])

        head.AddChain(headLeft, "nose")
        head.AddChain(headRight, "nose")

        #build kinematicChain
        completeChain.AddChain(leftArm, "left_shoulder")
        completeChain.AddChain(rightLeg, "right_hip")
        completeChain.AddChain(leftLeg, "left_hip")
        completeChain.AddChain(rightArm, "right_shoulder")
        completeChain.AddChain(head)

        super(PoseModelHuman17_2D, self).__init__(completeChain, mapping=mapping)

    def PosesToChain(self, poses: PoseBase, mapping: ModelMappings) -> List[PoseBase]:
        if mapping == ModelMappings.CHAIN:
            # do nothing
            pass
        elif mapping == ModelMappings.COCO:
            poses = self.CocoToChainModel(poses)
        else:
            raise Exception("Pose Mapping is not jet supported!")
        return poses

    def ChainToPoses(self, mapping: ModelMappings) -> List[PoseBase]:
        poses = self.kinematicChain.GetPoseList()
        if mapping == ModelMappings.CHAIN:
            return poses
        elif mapping == ModelMappings.COCO:
            return self.ChainModelToCoco(poses)
        else:
            raise Exception("Pose Mapping is not jet supported!")

    @staticmethod
    def CocoToChainModel(poses: List[PoseBase]) -> List[PoseBase]:
        '''
         input                                    ->    output
         "nose"-------------#keypoint 0  ---------->    "left_shoulder"----#keypoint 5
         "left_eye"---------#keypoint 1  ---------->    "left_hip"---------#keypoint 11
         "right_eye"--------#keypoint 2  ---------->    "right_hip"--------#keypoint 12
         "left_ear"---------#keypoint 3  ---------->    "right_shoulder"---#keypoint 6
         "right_ear"--------#keypoint 4  ---------->    "left_elbow"-------#keypoint 7
         "left_shoulder"----#keypoint 5  ---------->    "left_wrist"-------#keypoint 9
         "right_shoulder"---#keypoint 6  ---------->    "left_knee"--------#keypoint 13
         "left_elbow"-------#keypoint 7  ---------->    "left_ankle"-------#keypoint 15
         "right_elbow"------#keypoint 8  ---------->    "right_knee"-------#keypoint 14
         "left_wrist"-------#keypoint 9  ---------->    "right_ankle"------#keypoint 16
         "right_wrist"------#keypoint 10 ---------->    "right_elbow"------#keypoint 8
         "left_hip"---------#keypoint 11 ---------->    "right_wrist"------#keypoint 10
         "right_hip"--------#keypoint 12 ---------->    "nose"-------------#keypoint 0
         "left_knee"--------#keypoint 13 ---------->    "left_eye"---------#keypoint 1
         "right_knee"-------#keypoint 14 ---------->    "left_ear"---------#keypoint 3
         "left_ankle"-------#keypoint 15 ---------->    "right_eye"--------#keypoint 2
         "right_ankle"------#keypoint 16 ---------->    "right_ear"--------#keypoint 4
        '''

        ret = list()
        # torsoPoses
        ret.append(poses[5])  # left_shoulder
        ret.append(poses[11])  # left_hip
        ret.append(poses[12])  # right_hip
        ret.append(poses[6])  # right_shoulder

        # leftArmSegmet
        ret.append(poses[7])  # left_elbow
        ret.append(poses[9])  # left_wrist

        # rightArmSegmet
        ret.append(poses[13])  # left_knee
        ret.append(poses[15])  # left_wrist

        # leftLegSegment
        ret.append(poses[14])  # right_knee
        ret.append(poses[16])  # right_ankle

        # rightLegSegment
        ret.append(poses[8])   # right_elbow
        ret.append(poses[10])  # right_wrist

        # headSegment add last because unconnected chains will be handled last
        ret.append(poses[0])  # nose

        ret.append(poses[1])  # left_eye
        ret.append(poses[3])  # left_ear

        ret.append(poses[2])  # right_eye
        ret.append(poses[4])  # right_ear

        return ret


    @staticmethod
    def ChainModelToCoco(poses: List[PoseBase]) -> List[PoseBase]:
        '''
         input                                    ->    output
         "left_shoulder"---#keypoint 0  ---------->    "nose"-------------#keypoint 12
         "left_hip"--------#keypoint 1  ---------->    "left_eye"---------#keypoint 13
         "right_hip"-------#keypoint 2  ---------->    "right_eye"--------#keypoint 15
         "right_shoulder"--#keypoint 3  ---------->    "left_ear"---------#keypoint 14
         "left_elbow"------#keypoint 4  ---------->    "right_ear"--------#keypoint 16
         "left_wrist"------#keypoint 5  ---------->    "left_shoulder"----#keypoint 0
         "left_knee"-------#keypoint 6  ---------->    "right_shoulder"---#keypoint 3
         "left_ankle"------#keypoint 7  ---------->    "left_elbow"-------#keypoint 4
         "right_knee"------#keypoint 8 ---------->     "right_elbow"------#keypoint 10
         "right_ankle"-----#keypoint 9 ---------->     "left_wrist"-------#keypoint 5
         "right_elbow"-----#keypoint 10  ---------->   "right_wrist"------#keypoint 11
         "right_wrist"-----#keypoint 11  ---------->   "left_hip"---------#keypoint 1
         "nose"------------#keypoint 12 ---------->    "right_hip"--------#keypoint 2
         "left_eye"--------#keypoint 13 ---------->    "left_knee"--------#keypoint 6
         "left_ear"--------#keypoint 14 ---------->    "right_knee"-------#keypoint 8
         "right_eye"-------#keypoint 15 ---------->    "left_ankle"-------#keypoint 7
         "right_ear"-------#keypoint 16 ---------->    "right_ankle"------#keypoint 9
        '''


        ret = list()
        ret.append(poses[12]) #keypoint 0
        ret.append(poses[13]) #keypoint 1
        ret.append(poses[15]) #keypoint 2
        ret.append(poses[14]) #keypoint 3
        ret.append(poses[16]) #keypoint 4
        ret.append(poses[0]) #keypoint 5
        ret.append(poses[3]) #keypoint 6
        ret.append(poses[4]) #keypoint 7
        ret.append(poses[10]) #keypoint 8
        ret.append(poses[5]) #keypoint 9
        ret.append(poses[11]) #keypoint 10
        ret.append(poses[1]) #keypoint 11
        ret.append(poses[2]) #keypoint 12
        ret.append(poses[6]) #keypoint 13
        ret.append(poses[8]) #keypoint 14
        ret.append(poses[7]) #keypoint 15
        ret.append(poses[9]) #keypoint 16

        return ret
