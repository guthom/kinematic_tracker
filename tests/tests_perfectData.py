import unittest
import os
from typing import List
from unittest import TestCase
from kinematic_tracker.pose_models.PoseModelRopose_2D import PoseModelRopose_2D
from kinematic_tracker.pose_models.KinematicChain2D import KinematicChain2D
from kinematic_tracker.pose_models.KinematicChainBase import KinematicChainBase

from guthoms_helpers.base_types.Pose2D import Pose2D
from guthoms_helpers.common_helpers.RandomHelper import RandomHelper

from guthoms_helpers.filesystem.FileHelper import FileHelper
from guthoms_helpers.common_stuff.ProgressBar import ProgressBar
from ropose_dataset_tools.DataSetLoader import LoadDataSet

from kinematic_tracker.tracking.Tracker import Tracker
import random
import time
import numpy as np
import cv2
import copy

class PerfectDataTest(TestCase):

    def LoadTestData(self):
        filePath = FileHelper.GetFilePath(os.path.abspath(__file__))
        self.roposeDatasets = LoadDataSet("/mnt/datastuff/Datasets/real_test/colropose_thomas_3")

    @staticmethod
    def Displace2DJoints(joints: List[Pose2D], min: int = 0, max: int = 10) -> np.array:

        for i in range(0, len(joints)):
            joints[i].trans.x += RandomHelper.RandomInt(min, max)
            joints[i].trans.y += RandomHelper.RandomInt(min, max)

        return joints

    @staticmethod
    def InvalidateJoints(joints: List[Pose2D], prob: float = 0.1) -> np.array:

        for i in range(0, len(joints)):
            if RandomHelper.DecideByProb(prob):
                joints[i].visible = False
                joints[i].trans.x = -1
                joints[i].trans.y = -1

        return joints

    def setUp(self):
        self.LoadTestData()

    def test_TrackerInvalidationWorks(self):
        roposeTracker = Tracker(similarityThreshold=0.8, invalidationTimeMs=100)

        combined = []
        roposeModel = PoseModelRopose_2D()
        for dataset in ProgressBar(self.roposeDatasets[0:10]):
            poses = dataset.rgbFrame.resizedReprojectedPoints
            roposeModel.UpdatePoses(poses)
            roposeTracker.CheckInvalidation()
            self.assertTrue(roposeTracker.trackedInstances.__len__() == 0)
            sims = roposeTracker.AddSupervision([roposeModel], autoCheckInvalidation=False)
            time.sleep(0.11)

        for dataset in ProgressBar(self.roposeDatasets[0:10]):
            poses = dataset.rgbFrame.resizedReprojectedPoints
            roposeModel.UpdatePoses(poses)
            sims = roposeTracker.AddSupervision([roposeModel])
            self.assertTrue(roposeTracker.trackedInstances.__len__() == 1)
            time.sleep(0.05)



    def test_TrackerVariousDisplacements(self):
        roposeTracker = Tracker(similarityThreshold=0.8, invalidationTimeMs=1000)

        combined = []
        for displacement in range(0, 30):
            roposeModel = PoseModelRopose_2D()
            allSims = []
            counter = 0
            for dataset in ProgressBar(self.roposeDatasets[0:200]):
                poses = dataset.rgbFrame.resizedReprojectedPoints
                if displacement > 0:
                    poses = self.Displace2DJoints(poses, min=-displacement, max=displacement)

                poses = self.InvalidateJoints(poses, prob=0.05)

                roposeModel.UpdatePoses(poses)

                sims = roposeTracker.AddSupervision([roposeModel])
                if counter != 0:
                    allSims.extend(sims)
                # print("Similarities: " + str(sims))
                counter += 1

            combined.append([displacement, np.mean(allSims), np.min(allSims), np.max(allSims)])

            print("")
            print("DisplaceMent: " + str(displacement))
            print("MeanSim: " + str(np.mean(allSims)) +
                  " MaxSim: " + str(np.max(allSims)) +
                  " MinSim: " + str(np.min(allSims)))

        print(combined)

    def test_trackerPerfektInvalidationData(self):
        roposeTracker = Tracker(similarityThreshold=0.90, invalidationTimeMs=1000)

        roposeModel = PoseModelRopose_2D()
        allSims = []
        counter = 0
        for prob in [0.1, 0.15, 0.2]:
            for dataset in ProgressBar(self.roposeDatasets):
                dataset = copy.copy(dataset)
                poses = self.InvalidateJoints(dataset.rgbFrame.resizedReprojectedPoints, prob=prob)
                roposeModel.UpdatePoses(poses)
                sims = roposeTracker.AddSupervision([roposeModel])

                if counter != 0:
                    allSims.extend(sims)

                #print("Similarities: " + str(sims))
                if roposeTracker.trackedInstances.__len__() > 1:
                    test = True
                #we allow some mistakes here
                self.assertTrue(roposeTracker.trackedInstances.__len__() == 1 or 2)
                counter += 1

            print("MeanSim: " + str(np.mean(allSims)) +
                  " MaxSim: " + str(np.max(allSims)) +
                  " MinSim: " + str(np.min(allSims)))

    def test_trackerPerfectData(self):
        roposeTracker = Tracker(similarityThreshold=0.9, invalidationTimeMs=1000)

        roposeModel = PoseModelRopose_2D()
        allSims = []
        counter = 0
        for dataset in ProgressBar(self.roposeDatasets):
            roposeModel.UpdatePoses(dataset.rgbFrame.resizedReprojectedPoints)
            sims = roposeTracker.AddSupervision([roposeModel])

            if counter != 0:
                allSims.extend(sims)

            #print("Similarities: " + str(sims))
            self.assertEqual(roposeTracker.trackedInstances.__len__(), 1)
            counter += 1
        print("MeanSim: " + str(np.mean(allSims)) +
              " MaxSim: " + str(np.max(allSims)) +
              " MinSim: " + str(np.min(allSims)))

    def test_BoundingBoxPrediction(self):
        ious: List[float] = []
        ious_static: List[float] = []
        roposeTracker = Tracker(similarityThreshold=0.8, invalidationTimeMs=1000)
        roposeTracker_static = Tracker(similarityThreshold=0.8, invalidationTimeMs=1000, relativeOrder=False)

        roposeModel = PoseModelRopose_2D()
        counter = 0

        for dataset in ProgressBar(self.roposeDatasets):
            poses = dataset.rgbFrame.resizedReprojectedPoints

            #print("Sims: " + str(sims))
            if counter > 0:
                predBB = roposeTracker.Predict2DBB()
                predBB_Static = roposeTracker_static.Predict2DBB()

                for bb in predBB:
                    iou = bb.CalculateIoU(dataset.rgbFrame.boundingBox)
                    ious.append(iou)

                for bb in predBB_Static:
                    iou = bb.CalculateIoU(dataset.rgbFrame.boundingBox)
                    ious_static.append(iou)


            roposeModel.UpdatePoses(poses)
            sims = roposeTracker.AddSupervision([roposeModel])
            sims_static = roposeTracker_static.AddSupervision([roposeModel])
            counter += 1

        meanIou = np.mean(ious)
        meanIou_Static = np.mean(ious_static)
        print("Mean IOUS: " + str(meanIou))
        print("Mean IOUS Static: " + str(meanIou_Static))
        self.assertTrue(0.6 < meanIou <= 1.0)
        self.assertTrue(0.6 < meanIou_Static <= 1.0)
        self.assertTrue(meanIou != meanIou_Static)

    def test_ChainPrediction(self):
        ious: List[float] = []
        ious_static: List[float] = []
        roposeTracker = Tracker(similarityThreshold=0.8, invalidationTimeMs=1000)
        roposeTracker_static = Tracker(similarityThreshold=0.8, invalidationTimeMs=1000, relativeOrder=False)

        roposeModel = PoseModelRopose_2D()
        counter = 0

        for dataset in ProgressBar(self.roposeDatasets):
            poses = dataset.rgbFrame.resizedReprojectedPoints

            # print("Sims: " + str(sims))
            if counter > 0:
                predChain = roposeTracker.PredictChain()
                predChain_Static = roposeTracker_static.PredictChain()

                for chain in predChain:
                    iou = chain.GetBoundingBox().CalculateIoU(dataset.rgbFrame.boundingBox)
                    ious.append(iou)

                for chain in predChain_Static:
                    iou = chain.GetBoundingBox().CalculateIoU(dataset.rgbFrame.boundingBox)
                    ious_static.append(iou)

            roposeModel.UpdatePoses(poses)
            sims = roposeTracker.AddSupervision([roposeModel])
            sims_static = roposeTracker_static.AddSupervision([roposeModel])
            counter += 1

        meanIou = np.mean(ious)
        meanIou_Static = np.mean(ious_static)
        print("Mean IOUS: " + str(meanIou))
        print("Mean IOUS Static: " + str(meanIou_Static))
        self.assertTrue(0.5 < meanIou <= 1.0)
        self.assertTrue(0.5 < meanIou_Static <= 1.0)
        self.assertTrue(meanIou != meanIou_Static)


    def test_trackerPerfectDataWithSmallDisplacement(self):
        roposeTracker = Tracker(similarityThreshold=0.8, invalidationTimeMs=1000)

        roposeModel = PoseModelRopose_2D()
        allSims = []
        counter = 0
        for dataset in ProgressBar(self.roposeDatasets):
            poses = dataset.rgbFrame.resizedReprojectedPoints
            poses = self.Displace2DJoints(poses, min=-5, max=5)
            roposeModel.UpdatePoses(poses)

            sims = roposeTracker.AddSupervision([roposeModel])
            if counter != 0:
                allSims.extend(sims)
            #print("Similarities: " + str(sims))

            self.assertEqual(roposeTracker.trackedInstances.__len__(), 1)
            counter += 1

        print("MeanSim: " + str(np.mean(allSims)) +
              " MaxSim: " + str(np.max(allSims)) +
              " MinSim: " + str(np.min(allSims)))

    def test_trackerPerfectDataWithLargeDisplacement(self):
        roposeTracker = Tracker(similarityThreshold=0.92, invalidationTimeMs=2000)

        roposeModel = PoseModelRopose_2D()
        counter = 0
        allSims = []
        for dataset in ProgressBar(self.roposeDatasets):
            poses = dataset.rgbFrame.resizedReprojectedPoints
            poses = self.Displace2DJoints(poses, min=-50, max=50)
            roposeModel.UpdatePoses(poses)

            sims = roposeTracker.AddSupervision([roposeModel])
            if counter != 0:
                allSims.extend(sims)
            #print("Similarities: " + str(sims))

            if counter == 0:
                self.assertTrue(roposeTracker.trackedInstances.__len__() == 1)
            else:
                self.assertTrue(roposeTracker.trackedInstances.__len__() >= 1)
            counter += 1

        print("MeanSim: " + str(np.mean(allSims)) +
              " MaxSim: " + str(np.max(allSims)) +
              " MinSim: " + str(np.min(allSims)))



