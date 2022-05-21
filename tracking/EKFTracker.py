from typing import Optional, Union
from kinematic_tracker.pose_models.PoseModelBase import PoseModelBase
from kinematic_tracker.tracking.TrackedEKFInstance import TrackedEKFInstance as TrackedInstance
from kinematic_tracker.tracking.TrackingInformation import TrackingInformation
from guthoms_helpers.base_types.PoseBase import PoseBase
from guthoms_helpers.base_types.BoundingBoxBase import BoundingBoxBase
from guthoms_helpers.base_types.OpticalFlow import OpticalFlow
from kinematic_tracker.pose_models.KinematicChainBase import KinematicChainBase

from kinematic_tracker.tracking.TrackingDescriptor import TrackingDescriptor
import numpy as np
import copy
import math
import cv2

from typing import List, Dict

class EKFTracker(object):

    def __init__(self, similarityThreshold: float=0.9, iouThreshold: float=0.25, invalidationTimeMs: int = 5000,
                 minVisiblePoses: int = 3, relativeOrder:bool = True):
        self.trackedInstances: List[TrackedInstance] = []
        self.simThreashold = similarityThreshold
        self.iouThreshold = iouThreshold
        self.invalidationTime = invalidationTimeMs
        self.minVisiblePoses = minVisiblePoses
        self.relativeOrder = relativeOrder


    def CheckModel(self, poseModel: PoseModelBase):

        visibleCounter = 0
        poses = poseModel.GetPoses()

        for pose in poses:
            if pose.visible:
                visibleCounter += 1

        #if enough joints are visible and the resulting BB area/diag is not 0.0
        if visibleCounter >= self.minVisiblePoses and poseModel.kinematicChain.GetBoundingBox().DiagLength() > 0.0:
            return True

        return False

    def GetBoundingBoxes(self) -> List[BoundingBoxBase]:
        ret: List[BoundingBoxBase] = []
        for instance in self.trackedInstances:
            ret.append(instance.trackingInfos.kinematicChain.GetBoundingBox())

        return ret

    def GetNotKnownInstances(self, boundingBoxes: List[BoundingBoxBase], iouThreshold: Optional[float] = None,
                             predict: bool = True, usePose: bool = False, useHistoryCount: int=5,
                             histWeightFactor:float = 0.5, weighted: bool = True) -> List[BoundingBoxBase]:
        ret = []
        if iouThreshold is None:
            iouThreshold = self.iouThreshold

        ious = []

        if self.trackedInstances.__len__() <= 0:
            return []

        if predict:
            if not usePose:
                knwonBB = self.Predict2DBB(useHistoryCount,
                                           weighted=weighted,
                                           weigthFactor=histWeightFactor)
            else:
                knwonBB = self.PredictChain(useHistoryCount,
                                            weighted=weighted,
                                            weigthFactor=histWeightFactor,
                                            asBoundigBox=True)
        else:
            knwonBB = self.GetBoundingBoxes()

        #if we have no detection at all just append all known instances
        if boundingBoxes.__len__() == 0:
            ret.extend(knwonBB)
            return ret

        for bb in knwonBB:
            iou = []

            for boundingBox in boundingBoxes:
                sim = boundingBox.CalculateIoU(bb)
                iou.append(sim)

            ious.append(iou)

        if ious.__len__() == 0:
            return []

        #filter based on max ious and threshold
        npArray = np.array(ious)
        maxValuesIndex = np.argmax(npArray, axis=1)

        for i in range(0, npArray.shape[0]):
            maxValue = npArray[i, maxValuesIndex[i]]
            #delete detection if the max IOU is greater than the set threshold, this means every box with a smaller
            # th is counted as UNKNOWN
            if maxValue < iouThreshold:
                ret.append(knwonBB[i])
        return ret


    def AddSupervision(self, poseModel: List[PoseModelBase], autoCheckInvalidation: bool = True) -> List[float]:

        retSims = []

        if autoCheckInvalidation:
            self.CheckInvalidation()

        matchedIndexes = []

        for model in poseModel:

            #if the model have not enough visible joints discard it
            if not self.CheckModel(model):
                retSims.append(0.0)
                continue

            trackInfos = TrackingInformation(model.kinematicChain)

            if self.trackedInstances.__len__() == 0:
                newInstance = TrackedInstance(trackInfos)
                self.trackedInstances.append(newInstance)
                retSims.append(0.0)
                continue

            similarities = self.CalcSimilarities(self.trackedInstances, trackInfos.descriptor)

            trackedInstanceIndex = self.FindMatch(similarities, self.simThreashold)

            #new instance found, descriptor is not matching with knwon instances
            if trackedInstanceIndex == -1:
                newInstance = TrackedInstance(trackInfos)
                self.trackedInstances.append(newInstance)
                retSims.append(0.0)
            #update known instance
            else:
                if trackedInstanceIndex not in matchedIndexes:
                    self.trackedInstances[trackedInstanceIndex].Update(trackInfos)
                    matchedIndexes.append(trackedInstanceIndex)

                retSims.append(similarities[trackedInstanceIndex])

        return retSims

    def Predict2DBB(self, maxHistoryCount: int = 5, weighted: bool = True,
                    weigthFactor: float = 0.9, weightMethod: str = "expo") -> List[BoundingBoxBase]:
        #calculate mean change of the bounding box over the known history
        resBB = []
        for instance in self.trackedInstances:
            history = instance.history.GetLastItems(maxHistoryCount, delete=False)
            if history is not None and history.__len__() > 1:
                compLength = history.__len__() - 1
                # create to track the 4 Edge points of an bounding box
                flows: List[List[OpticalFlow]] = [list() for i in range(0, 4)]
                # get weigths for histroy
                flowWeigths = self.GetPredictionWeigths(maxHistoryCount, history.__len__(), weigthFactor,
                                                        weightMethod=weightMethod)

                maxSpeed = 0.0
                latestBB: BoundingBoxBase = history[-1].descriptor.kinematicChain.GetBoundingBox()
                if self.relativeOrder:
                    for i in range(0, history.__len__() - 1):
                        bb1: BoundingBoxBase = history[i].descriptor.kinematicChain.GetBoundingBox()
                        edgesBB1 = bb1.GetEdgePoints()
                        bb2: BoundingBoxBase = history[i + 1].descriptor.kinematicChain.GetBoundingBox()
                        edgesBB2 = bb2.GetEdgePoints()

                        for j in range(0, edgesBB1.__len__()):
                            flows[j].append(edgesBB1[j].CalculateFlow(edgesBB2[j]))
                            maxSpeed = max(maxSpeed, flows[j][-1].speed)
                else:
                    for i in range(0, history.__len__() - 1):
                        bb1: BoundingBoxBase = history[i].descriptor.kinematicChain.GetBoundingBox()
                        edgesBB1 = bb1.GetEdgePoints()
                        bb2: BoundingBoxBase = latestBB
                        edgesBB2 = bb2.GetEdgePoints()

                        for j in range(0, edgesBB1.__len__()):
                            flows[j].append(edgesBB1[j].CalculateFlow(edgesBB2[j]))
                            maxSpeed = max(maxSpeed, flows[j][-1].speed)

                #meanFlow:
                meanFlows = []
                for j in range(0, flows.__len__()):
                    if weighted:
                        for k in range(0, flows[j].__len__()):
                            #weight = flowWeigths[k] * flows[j][k].speed/maxSpeed
                            flows[j][k] *= flowWeigths[k]

                        meanFlows.append(OpticalFlow.Mean(flows[j]))

                # add mean optical flow to edgpoints of bb
                movedBBEdges = latestBB.GetEdgePoints()
                for j in range(0, movedBBEdges.__len__()):
                    # add mean flow displacement
                    disp = meanFlows[j].Displacement()
                    movedBBEdges[j] += disp
                resBB.append(latestBB.CreateBoundingBox(keyPoints=movedBBEdges))
        return resBB

    def GetPredictionWeigths(self, maxHistoryCount: int, histLength: int, weigthFactor: float,
                             weightMethod: str ="expo") -> List[float]:
        maxHistoryCount = min(maxHistoryCount, histLength)
        flowWeigths = []
        if weightMethod == "expo":
            for x in range(0, maxHistoryCount-1, 1):
                weight = math.exp(-weigthFactor/2 * x)
                flowWeigths.append(weight)

        elif weightMethod == "lin":
            for x in range(0, maxHistoryCount-1, 1):
                weight = 1 - x/maxHistoryCount * weigthFactor
                weight = max(weight, 0.0)
                flowWeigths.append(weight)

            flowWeigths = np.array(flowWeigths) / np.max(flowWeigths)
        else:
            raise Exception("Unknown Weigth Method!")

        #normalize weights
        flowWeigths = np.array(flowWeigths) / np.sum(flowWeigths)
        if not np.isclose(np.sum(flowWeigths), 1.0):
            print("Sum: " + str(np.sum(flowWeigths)))
        #print("wFactor: " + str(weigthFactor))
        #print(weightMethod + ": " + str(flowWeigths))
        return list(reversed(flowWeigths))

    def PredictChain(self, maxHistoryCount: int = 5, weighted: bool = True,
                     weigthFactor: float = 0.5, asBoundigBox: bool=False,  weightMethod: str = "expo") -> \
            List[Union[KinematicChainBase, BoundingBoxBase]]:
        # calculate optical flow of each pose of the kinematic chain
        resChains:  List[Union[KinematicChainBase, BoundingBoxBase]] = []

        for instance in self.trackedInstances:
            # Get History, index -1 ist the last known tracking information
            history: List[TrackingInformation] = instance.history.GetLastItems(maxHistoryCount, delete=False)
            compLength = history.__len__() - 1
            meanFlows = [list() for i in range(0, history[0].kinematicChain.linkCount)]
            if history is not None and history.__len__() > 1:
                flowWeigths = self.GetPredictionWeigths(maxHistoryCount, history.__len__(), weigthFactor,
                                                        weightMethod=weightMethod)
                #print("flowWeights: " + str(flowWeigths))

                if self.relativeOrder:
                    for i in range(0, compLength):
                        chain1: KinematicChainBase = history[i].descriptor.kinematicChain
                        chain2: KinematicChainBase = history[i + 1].descriptor.kinematicChain
                        flows = chain1.CalculateOpticalFlow(chain2)

                        for j in range(0, flows.__len__()):
                            meanFlows[j].append(flows[j])

                else:
                    latestChain = history[-1].descriptor.kinematicChain
                    for i in range(0, compLength):
                        chain1: KinematicChainBase = history[i].descriptor.kinematicChain
                        flows = chain1.CalculateOpticalFlow(latestChain)

                        for j in range(0, flows.__len__()):
                            meanFlows[j].append(flows[j])

                # meanFlow
                meanFlow: List[OpticalFlow] = list()
                if weighted:
                    for j in range(0, meanFlows.__len__()):
                        meanFlow.append(OpticalFlow.Mean(meanFlows[j], flowWeigths))
                else:
                    for j in range(0,  meanFlows.__len__()):
                        meanFlow.append(OpticalFlow.Mean(meanFlows[j]))

                movedPoses: List[PoseBase] = []
                for i in range(0, meanFlow.__len__()):
                    mean = meanFlow[i]
                    moved = history[-1].kinematicChain.poses[i]
                    displacement = mean.Displacement()
                    moved.trans += displacement
                    movedPose = type(instance.trackingInfos.kinematicChain.poses[0])(moved)
                    movedPoses.append(movedPose)

                newChain: KinematicChainBase = copy.deepcopy(instance.trackingInfos.kinematicChain)
                newChain.UpdatePoses(movedPoses)

                if not asBoundigBox:
                    resChains.append(newChain)
                else:
                    resChains.append(newChain.GetBoundingBox())

        return resChains

    def DrawInstances(self, image: np.array) -> np.array:
        for instance in self.trackedInstances:
            image = instance.trackingInfos.kinematicChain.Draw(image)
        return image

    def DrawBoundingBoxes(self, image: np.array, description: str=None, color: List[int] = [255, 0.0, 0.0]) -> np.array:
        for instance in self.trackedInstances:
            image = instance.trackingInfos.kinematicChain.GetBoundingBox().Draw(image, description, color)
        return image

    def DrawInformation(self, image: np.array, description: Optional[str]=None) -> np.array:

        if description is None:
            description = "Instances: "
        else:
            description = description + "-Instances: "

        cv2.putText(image, description + str(self.trackedInstances.__len__()), (20, 80),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(255, 0, 0), lineType=1, thickness=3)

        return image

    def CheckInvalidation(self):
        for trackedInstance in self.trackedInstances:
            valid = trackedInstance.Valid(self.invalidationTime)
            if not valid:
                self.trackedInstances.remove(trackedInstance)

    def FindMatch(self, similarities: List[float], threshold: float= 0.8) -> int:
        if similarities.__len__() == 0:
            return -1

        maxIndex = similarities.index(max(similarities))

        if similarities[maxIndex] > threshold:
            return maxIndex
        else:
            return -1

    def CalcSimilarities(self, trackedInstances: List[TrackedInstance], descriptor: TrackingDescriptor,
                         useHistory: int = 0) -> List[float]:

        similarities = []

        for instance in trackedInstances:

            instanceSim = []
            descriptors = [descriptor]

            if useHistory > 0:
                raise Exception("Not Implemented!")

            for descriptor in descriptors:
                instanceSim.append(instance.trackingInfos.descriptor.CompareAll(descriptor))

            similarities.append(float(np.mean(instanceSim)))

        return similarities







