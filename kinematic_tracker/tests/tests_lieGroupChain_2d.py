import unittest
import os
from unittest import TestCase
from kinematic_tracker.pose_models.PoseModelRopose_2D import PoseModelRopose_2D
from kinematic_tracker.pose_models.PoseModelHuman17_2D import PoseModelHuman17_2D
from kinematic_tracker.tests.tests_datasetBaseTest import DatasetBaseTest
from kinematic_tracker.pose_models.PoseModelHelper import PoseModelHelper
from kinematic_tracker.pose_models.ModelMappings import ModelMappings
from guthoms_helpers.common_stuff.FPSTracker import FPSTracker

from kinematic_tracker.pose_models.KinematicChain2D_Lie import KinematicChain2D_Lie
import numpy as np
import cv2
import copy

class LieGroupChainTests2D(DatasetBaseTest):

    def test_Conversion(self):
        roposeModel = PoseModelRopose_2D()

        fpsTracker = FPSTracker(meanCount=1)
        for dataset in self.roposeDatasets:
            roposeModel.UpdatePoses(dataset.rgbFrame.projectedJoints)

            lieChain = KinematicChain2D_Lie(roposeModel.kinematicChain)

