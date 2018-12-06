import bayes
import numpy as np

if __name__ == '__main__':
    print(str('start excute {}').format(__name__))

    loadData = bayes.SimulationData()
    worldSet, worldLabel = loadData.loadDataSet()
    worldList = loadData.createVocablist(worldSet)

    trainMat = []
    for world in worldSet:
        trainMat.append(loadData.setOfWorld(worldList, world))

    tool = bayes.utitliy()
    p0, p1, pc = tool.trainNB(np.array(trainMat), np.array(worldLabel))

    te = ['stupid', 'garbage']
    vec = loadData.setOfWorld(worldList, te)
    c = tool.classNB(np.array(vec), p0, p1, pc)
    print(str('{} \n {}').format(vec, c))
