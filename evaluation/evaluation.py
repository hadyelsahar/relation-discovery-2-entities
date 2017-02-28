import math


class ClusterEvaluation:

    def __init__(self, groundtruthlabels, predicted_clusters):
        self.relations = {}
        self.groundtruthsets, self.assessableElemSet = self.createGroundTruthSets(groundtruthlabels)
        self.predictedsets = self.createPredictedSets(predicted_clusters)

    def createGroundTruthSets(self, labels):

        groundtruthsets= {}
        assessableElems = set()

        for i, c in enumerate(labels):
            assessableElems.add(i)
            groundtruthsets.setdefault(c, set()).add(i)

        return groundtruthsets, assessableElems

    def createPredictedSets(self, cs):

        predictedsets = {}
        for i, c in enumerate(cs):
            predictedsets.setdefault(c, set()).add(i)

        return predictedsets

    def b3precision(self, response_a, reference_a):
        # print response_a.intersection(self.assessableElemSet), 'in precision'
        return len(response_a.intersection(reference_a)) / float(len(response_a.intersection(self.assessableElemSet)))

    def b3recall(self, response_a, reference_a):
        return len(response_a.intersection(reference_a)) / float(len(reference_a))

    def b3TotalElementPrecision(self):
        totalPrecision = 0.0
        for c in self.predictedsets:
            for r in self.predictedsets[c]:
                totalPrecision += self.b3precision(self.predictedsets[c],
                                                   self.findCluster(r, self.groundtruthsets))

        return totalPrecision / float(len(self.assessableElemSet))

    def b3TotalElementRecall(self):
        totalRecall = 0.0
        for c in self.predictedsets:
            for r in self.predictedsets[c]:
                totalRecall += self.b3recall(self.predictedsets[c], self.findCluster(r, self.groundtruthsets))

        return totalRecall / float(len(self.assessableElemSet))

    def findCluster(self, a, setsDictionary):
        for c in setsDictionary:
            if a in setsDictionary[c]:
                return setsDictionary[c]

    def printEvaluation(self):

        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()
        betasquare = math.pow(0.5, 2)
        if recB3 == 0.0 and precB3 == 0.0:
            F1B3 = 0.0
            F05B3 = 0.0
        else:
            betasquare = math.pow(0.5, 2)
            F1B3 = (2 * recB3 * precB3) / (recB3 + precB3)
            F05B3 = ((1+betasquare) * recB3 * precB3)/((betasquare*precB3)+recB3)

        m = {'Elementwise B3 F1': F1B3, 'F0.5': F05B3, 'B3 recall': recB3, 'B3 precision': precB3}
        return m

    def getF05(self):
        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()
        betasquare = math.pow(0.5, 2)
        if recB3 == 0.0 and precB3 == 0.0:
            F05B3 = 0.0
        else:
            F05B3 = ((1+betasquare) * recB3 * precB3)/((betasquare*precB3)+recB3)
        return F05B3

    def getF1(self):
        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()

        if recB3 == 0.0 and precB3 == 0.0:
            F1B3 = 0.0
        else:
            F1B3 = (2 * recB3 * precB3) / (recB3 + precB3)
        return F1B3

