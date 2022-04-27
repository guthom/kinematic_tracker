import unittest
import os, sys
from kinematic_tracker.tests.tests_datasetBaseTest import DatasetBaseTest
from kinematic_tracker.pose_models.KinematicChainBase import KinematicChainBase
from kinematic_tracker.pose_models.PoseModelRopose_2D import PoseModelRopose_2D
from kinematic_tracker.pose_models.PoseModelHuman17_2D import PoseModelHuman17_2D
from kinematic_tracker.tracking.TrackingDescriptor import TrackingDescriptor
from kinematic_tracker.tracking.Tracker import Tracker
from guthoms_helpers.base_types.PoseBase import PoseBase
from guthoms_helpers.common_helpers.RandomHelper import RandomHelper

from guthoms_helpers.base_types.Vector2D import Vector2D
import random
import numpy as np
import cv2
import copy
from typing import List

class TrackingTests2D(DatasetBaseTest):

    def test_DescriptorCalculationRopose(self):
        for dataset in self.roposeDatasets:
            roposeModel = PoseModelRopose_2D()
            roposeModel.UpdatePoses(dataset.rgbFrame.projectedJoints)
            desctiptor = TrackingDescriptor(roposeModel.kinematicChain)
            self.assertIsNotNone(desctiptor)

    def test_DescriptorComparisonRopose(self):
        firstPoseModel = PoseModelRopose_2D()
        firstPoseModel.UpdatePoses(self.roposeDatasets[0].rgbFrame.projectedJoints)
        firstDescriptor = TrackingDescriptor(firstPoseModel.kinematicChain)

        for i in range(1, self.roposeDatasets.__len__()):
            roposeModel1 = PoseModelRopose_2D()
            roposeModel1.UpdatePoses(self.roposeDatasets[i-1].rgbFrame.projectedJoints)

            roposeModel2 = PoseModelRopose_2D()
            roposeModel2.UpdatePoses(self.roposeDatasets[i].rgbFrame.projectedJoints)

            descriptor1 = TrackingDescriptor(roposeModel1.kinematicChain)
            descriptor2 = TrackingDescriptor(roposeModel2.kinematicChain)

            #compare pose descriptor with itself should always be a perfect match

            similarity = descriptor1.ComparePose(descriptor1)
            self.assertEqual(similarity, 1.0)
            print("Self: " + str(similarity))

            similarity = descriptor1.ComparePose(descriptor2)
            self.assertLess(similarity, 1.0)
            print("Last: " + str(similarity))

            similarity = firstDescriptor.ComparePose(descriptor1)
            #first descriptor is the same as i = 1
            if i > 1:
                self.assertLess(similarity, 1.0)
            print("First: " + str(similarity))

            #test if posematching is just relative to base
            tempPoses = copy.deepcopy(self.roposeDatasets[i-1].rgbFrame.projectedJoints)

            #statically manipulate poses to move the whole kinematic chain
            offset = Vector2D(int(random.randrange(100)), int(random.randrange(100)))
            for j in range(0, tempPoses.__len__()):
                tempPoses[j].trans += offset

            roposeModel2.UpdatePoses(tempPoses)
            descriptor2 = TrackingDescriptor(roposeModel2.kinematicChain)
            similarity = descriptor1.ComparePose(descriptor2)
            self.assertEqual(similarity, 1.0)
            print("Relative: " + str(similarity) + " - Offset: " + str(offset))
        print("--------------- Ropose2D Comparison test finish! ---------------")

    def test_DescriptorComparisonHuman(self):

        firstPoseModel = PoseModelHuman17_2D()
        firstPoseModel.UpdatePoses(self.humanTestData[0])
        firstDescriptor = TrackingDescriptor(firstPoseModel.kinematicChain)

        for i in range(1, self.humanTestData.__len__()):
            testModel1 = PoseModelHuman17_2D()
            testModel1.UpdatePoses(self.humanTestData[i-1])

            testModel2 = PoseModelHuman17_2D()
            testModel2.UpdatePoses(self.humanTestData[i])

            descriptor1 = TrackingDescriptor(testModel1.kinematicChain)
            descriptor2 = TrackingDescriptor(testModel2.kinematicChain)

            #compare pose descriptor with itself should always be a perfect match

            similarity = descriptor1.ComparePose(descriptor1)
            self.assertEqual(similarity, 1.0)
            print("Self: " + str(similarity))

            similarity = descriptor1.ComparePose(descriptor2)
            self.assertLess(similarity, 1.0)
            print("Last: " + str(similarity))

            similarity = firstDescriptor.ComparePose(descriptor1)
            #first descriptor is the same as i = 1
            if i > 1:
                self.assertLess(similarity, 1.0)
            print("First: " + str(similarity))

            #test if posematching is just relative to base
            tempPoses = copy.deepcopy(self.humanTestData[i-1])

            #statically manipulate poses to move the whole kinematic chain
            offset = Vector2D(int(random.randrange(100)), int(random.randrange(100)))
            for j in range(0, tempPoses.__len__()):
                tempPoses[j].trans += offset

            testModel2.UpdatePoses(tempPoses)
            descriptor2 = TrackingDescriptor(testModel2.kinematicChain)
            similarity = descriptor1.ComparePose(descriptor2)
            self.assertEqual(similarity, 1.0)
            print("Relative: " + str(similarity) + " - Offset: " + str(offset))

        print("--------------- Human2D Comparison test finish! ---------------")

    def test_trackerMasking(self):
        return
        tracker = Tracker()
        roposeModel = PoseModelRopose_2D()
        for i in range(0, self.roposeDatasets.__len__()):
            poses = self.roposeDatasets[i].rgbFrame.projectedJoints
            roposeModel.UpdatePoses(poses)

            sim = tracker.AddSupervision(roposeModel)
            print("Calculated Similarity: " + str(sim))
            # tracked instances have to stay 1
            self.assertIs(tracker.trackedInstances.__len__(), 1)

        # return to first known pose and track back
        for j in range(self.roposeDatasets.__len__() - 1, 0, -1):
            roposeModel.UpdatePoses(self.roposeDatasets[j].rgbFrame.projectedJoints)
            sim = tracker.AddSupervision(roposeModel)
            print("Calculated Similarity: " + str(sim))
            # tracked instances have to stay 1
            self.assertIs(tracker.trackedInstances.__len__(), 1)

            # check for new Instance creation
            roposeModel2 = PoseModelRopose_2D()

    def randomlyInvalidatePose(self, chain: KinematicChainBase) -> KinematicChainBase:
        for pose in chain.poses:
            if RandomHelper.DecideByProb(0.2):
                pose.visible = False
                return chain

        return chain

    def test_trackerRopose(self):

        for times in range(0, 4):
            tracker = Tracker(similarityThreshold=0.80)

            roposeModel = PoseModelRopose_2D()

            for i in range(0, self.roposeDatasets.__len__()):
                poses = self.roposeDatasets[i].rgbFrame.projectedJoints
                roposeModel.UpdatePoses(poses)
                roposeModel.kinematicChain = self.randomlyInvalidatePose(roposeModel.kinematicChain)

                sim = tracker.AddSupervision([roposeModel])
                print("Calculated Similarity: " + str(sim))
                #tracked instances have to stay 1
                self.assertIn(tracker.trackedInstances.__len__(), [1, 2, 3])

            #return to first known pose and track back
            for i in range(self.roposeDatasets.__len__()-1, 0, -1):
                roposeModel.UpdatePoses(self.roposeDatasets[i].rgbFrame.projectedJoints)
                roposeModel.kinematicChain = self.randomlyInvalidatePose(roposeModel.kinematicChain)

                sim = tracker.AddSupervision([roposeModel])
                print("Calculated Similarity: " + str(sim))
                # tracked instances have to stay 1
                self.assertIn(tracker.trackedInstances.__len__(), [1, 2, 3])
                tracker.CheckInvalidation()

            # check for new Instance creation
            roposeModel2 = PoseModelRopose_2D()
            for i in range(self.roposeDatasets.__len__()-1, 0, -1):
                roposeModel2.UpdatePoses(self.roposeDatasets[i].rgbFrame.projectedJoints)
                roposeModel.kinematicChain = self.randomlyInvalidatePose(roposeModel.kinematicChain)
                roposeModel.UpdatePoses(self.roposeDatasets[i - self.roposeDatasets.__len__()].rgbFrame.projectedJoints)

                sim = tracker.AddSupervision([roposeModel2, roposeModel])

                print("Calculated Similarities - First: " + str(sim[0]) + " Second: " + str(sim[1]))
                # tracked instances have to stay 1
                self.assertIn(tracker.trackedInstances.__len__(), [1, 2])

    def test_BBPrediction(self):
        tracker = Tracker(similarityThreshold=0.8)

        roposeModel = PoseModelRopose_2D()
        for i in range(0, self.roposeDatasets.__len__()):
            poses = self.roposeDatasets[i].rgbFrame.projectedJoints
            roposeModel.UpdatePoses(poses)
            tracker.AddSupervision([roposeModel])

            #predictBBs = tracker.PredictBB(2)