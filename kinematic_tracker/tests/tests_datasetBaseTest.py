import unittest
import os
from unittest import TestCase
from kinematic_tracker.pose_models.PoseModelRopose_2D import KinematicChain2D
from kinematic_tracker.pose_models.PoseModelHuman17_2D import KinematicChain2D
from guthoms_helpers.base_types.Pose2D import Pose2D
from guthoms_helpers.common_helpers.RandomHelper import RandomHelper

from guthoms_helpers.filesystem.FileHelper import FileHelper
from ropose_dataset_tools.DataSetLoader import LoadDataSet
import random
import numpy as np
import cv2
import copy

class DatasetBaseTest(TestCase):

    def LoadTestData(self):
        filePath = FileHelper.GetFilePath(os.path.abspath(__file__))
        filePath = os.path.join(filePath, "test_data/ropose_test_dataset/")
        self.roposeDatasets = LoadDataSet(filePath)

    def CreateHumanTestData(self):
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
        self.humanTestData = []

        testData = []
        testData.append(Pose2D.fromData(500.0, 300.0, 0.0)) #"nose"
        testData.append(Pose2D.fromData(475.0, 270.0, 0.0)) #"left_eye"
        testData.append(Pose2D.fromData(525.0, 270.0, 0.0)) #"right_eye"
        testData.append(Pose2D.fromData(430.0, 280.0, 0.0)) #"left_ear"
        testData.append(Pose2D.fromData(570.0, 280.0, 0.0)) #"right_ear"

        testData.append(Pose2D.fromData(400.0, 450.0, 0.0)) #"left_shoulder"
        testData.append(Pose2D.fromData(600.0, 450.0, 0.0)) #"right_shoulder"

        testData.append(Pose2D.fromData(350.0, 550.0, 0.0)) #"left_elbow"
        testData.append(Pose2D.fromData(650.0, 550.0, 0.0)) #"right_elbow"

        testData.append(Pose2D.fromData(300.0, 500.0, 0.0)) #"left_wrist"
        testData.append(Pose2D.fromData(700.0, 500.0, 0.0)) #"right_wrist"

        testData.append(Pose2D.fromData(400.0, 700.0, 0.0)) #"left_hip"
        testData.append(Pose2D.fromData(600.0, 700.0, 0.0)) #"right_hip"

        testData.append(Pose2D.fromData(350.0, 800.0, 0.0)) #"left_knee"
        testData.append(Pose2D.fromData(650.0, 800.0, 0.0)) #"right_knee"

        testData.append(Pose2D.fromData(400.0, 900.0, 0.0)) #"left_ankle"
        testData.append(Pose2D.fromData(600.0, 900.0, 0.0)) #"right_ankle"

        self.humanTestData.append(copy.deepcopy(testData))
        randomRange = 15
        for i in range(0, 10):
            for i in range(0, testData.__len__()):
                testData[i].trans.x += RandomHelper.RandomInt(randomRange, ignoreVal=0)
                testData[i].trans.y += RandomHelper.RandomInt(randomRange, ignoreVal=0)

            self.humanTestData.append(copy.deepcopy(testData))

    def setUp(self):
        self.LoadTestData()
        self.CreateHumanTestData()
