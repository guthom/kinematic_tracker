import unittest
import os
from kinematic_tracker.tests.tests_datasetBaseTest import DatasetBaseTest
from kinematic_tracker.pose_models.PoseModelRopose_2D import PoseModelRopose_2D
from kinematic_tracker.pose_models.PoseModelHuman17_2D import PoseModelHuman17_2D
from kinematic_tracker.pose_models.PoseModelHelper import PoseModelHelper
import numpy as np
import cv2
import copy
import copy

class DrawingTests2D(DatasetBaseTest):

    def setUp(self):
        self.LoadTestData()
        self.CreateHumanTestData()
        self.image = np.zeros((1000, 1000, 3), np.uint8)


    def test_DrawRopose(self):
        display = cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

        roposeModel = PoseModelRopose_2D()

        for dataset in self.roposeDatasets:
            roposeModel.UpdatePoses(dataset.rgbFrame.projectedJoints)

            drawFrame = copy.copy(self.image)

            drawFrame = roposeModel.Draw(drawFrame)

            boudningBox = roposeModel.kinematicChain.GetBoundingBox()
            boudningBox.Draw(drawFrame, color=[255, 0.0, 0.0])

            cv2.imshow('image', drawFrame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    def test_HumanDrawing(self):
        display = cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

        humanModel = PoseModelHuman17_2D()

        for dataset in self.humanTestData:
            humanModel.UpdatePoses(dataset)

            drawFrame = copy.copy(self.image)

            drawFrame = humanModel.Draw(drawFrame)

            boudningBox = humanModel.kinematicChain.GetBoundingBox()
            boudningBox.Draw(drawFrame, color=[255, 0.0, 0.0])


            cv2.imshow('image', drawFrame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    def tearDown(self):
        pass
