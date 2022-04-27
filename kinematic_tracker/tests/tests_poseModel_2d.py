import unittest
import os
from unittest import TestCase
from kinematic_tracker.pose_models.PoseModelRopose_2D import PoseModelRopose_2D
from kinematic_tracker.pose_models.PoseModelHuman17_2D import PoseModelHuman17_2D
from kinematic_tracker.tests.tests_datasetBaseTest import DatasetBaseTest
from kinematic_tracker.pose_models.PoseModelHelper import PoseModelHelper
from kinematic_tracker.pose_models.ModelMappings import ModelMappings
from guthoms_helpers.common_stuff.FPSTracker import FPSTracker
import numpy as np
import cv2
import copy

class PoseModelTests2D(DatasetBaseTest):

    def test_RoposeModel(self):
        roposeModel = PoseModelRopose_2D()

        fpsTracker = FPSTracker(meanCount=1)
        for dataset in self.roposeDatasets:
            roposeModel.UpdatePoses(dataset.rgbFrame.projectedJoints)
            fpsTracker.FinishRun()

        print("Mean FPS: " + str(fpsTracker.MeanFPS()))

        pass

    def test_Human17Model(self):
        humanModel = PoseModelHuman17_2D()

        fpsTracker = FPSTracker(meanCount=1)

        for dataset in self.humanTestData:
            humanModel.UpdatePoses(dataset)
            fpsTracker.FinishRun()

        print("Mean FPS: " + str(fpsTracker.MeanFPS()))


    def test_HumanPoseMapping(self):
        humanModel = PoseModelHuman17_2D()

        fpsTracker = FPSTracker(meanCount=1)

        for dataset in self.humanTestData:
            humanModel.UpdatePoses(dataset)

            poseList = humanModel.GetPoses(mapping=ModelMappings.COCO)
            self.assertEqual(dataset, poseList)

            fpsTracker.FinishRun()

        print("Mean FPS: " + str(fpsTracker.MeanFPS()))

    def test_NumpyConversionMapping(self):
        humanModel = PoseModelHuman17_2D()

        fpsTracker = FPSTracker(meanCount=1)

        for dataset in self.humanTestData:

            humanModel.UpdatePoses(dataset)
            poseList = humanModel.GetPoses(mapping=ModelMappings.COCO)

            self.assertEqual(dataset, poseList)

            poseNp = PoseModelHelper.PoseListToNumpy(poseList)
            print(str(poseNp))

            print("-----------------------------------------------")
            mappedNp = PoseModelHelper.PoseListToNumpy(dataset)
            print(str(mappedNp))

            print("-----------------------------------------------")

            test = poseNp - mappedNp
            print(str(test))

            #self.assertTrue(equal)

            fpsTracker.FinishRun()

        print("Mean FPS: " + str(fpsTracker.MeanFPS()))