import numpy as np

class SimulationData:
    def loadDataSet(self):
        postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                       ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                       ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                       ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                       ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                       ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        classVec = [0, 1, 0, 1, 0, 1]
        return postingList, classVec

    def createVocablist(self, dataSet):
        worldList = set([])
        for doc in dataSet:
            worldList = worldList | set(doc)
        return list(worldList)

    def createVocabList(self, dataSet):
        vocabSet = set([])  # create empty set
        for document in dataSet:
            vocabSet = vocabSet | set(document)  # union of the two sets
        return list(vocabSet)

    def setOfWorld(self, voclist, inputData):
        retVec = [0] * len(voclist)
        for voc in voclist:
            if voc in inputData:
                retVec[voclist.index(voc)] = 1
        return retVec


class utitliy(object):

    def trainNB(self, dataSet, dataLabel):
        numDataSet = len(dataSet)
        worldNum = len(dataSet[0])
        pc = sum(dataLabel) / len(dataLabel)
        p0 = np.ones(worldNum); p1 = np.ones(worldNum)
        p0Dem = 2; p1Dem = 2
        for i in range(len(dataLabel)):
            if dataLabel[i] == 1:
                p1 += dataSet[i]
                p1Dem += sum(dataSet[i])
            else:
                p0 += dataSet[i]
                p0Dem += sum(dataSet[i])
        return np.log(p0/p0Dem), np.log(p1/p1Dem), pc

    def classNB(self, vec, pv0, pv1, pc):
        '''
        p(c0|vec) = p(vec|c0)*p(c0)/p(vec) = p(vec0|c0) * p(vec1|c0)*...*p(vecn|c0)
        p(c1|vec) = ......
        :param vec:  输入的特征
        :param pv0:  p(w0|c0) p(w1|c0) .... p(wn|c0)
        :param pv1:
        :param pc:
        :return:
        pv0 计算了样本属于c0 时 特的所有属性的概率，pv1同
        假如输入的特征 是 [0, 1, 0, 1, 0] 1表示当前单词在样本词集中出现了，0没有出现，
        及输入的特征是vec = ['my', 'cal']
        p(c0|vec) = p('my'|c0) * p（'cal'|c0) * pc0 / p('my')*p('cal')
        tp(c1|vec) 同
        没有用到样本所有特征
        分类时只关心所给特征，与样本其他特征无关
        所以计算p(c0|vec) 只关心出现过的单词，vec*pv1就过滤掉了不关心的特征
        '''
        p1 = vec * pv1
        p0 = vec * pv0
        p1 = sum(p1) + np.log(pc)
        p0 = sum(p0) + np.log(pc)
        if p1 > p0:
            return 1
        else:
            return 0

