import os

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from DataSplit import balanceNormalSamples


def getNoisyMinMax(dataX, epsilon):
    noisyDataX = np.random.laplace(loc=0.0, scale=1 / epsilon, size=(len(dataX), len(dataX[0])))
    noisyDataX = dataX + noisyDataX
    minX =noisyDataX.min(axis=0)# np.array([np.min(noisyDataX[:, x]) for x in range(len(noisyDataX[0]))])
    maxX =noisyDataX.max(axis=0) #np.array([np.max(noisyDataX[:, x]) for x in range(len(noisyDataX[0]))])
    return minX, maxX


def getMinMax(dataX):
    minX = dataX.min(axis=0)#np.array([np.min(dataX[:, x]) for x in range(len(dataX[0]))])
    maxX = dataX.min(axis=0)#np.array([np.max(dataX[:, x]) for x in range(len(dataX[0]))])
    return minX, maxX


def preprocessData(dataX):
    dataX *= 100
    dataX = dataX.astype(np.int32)
    return dataX


def removeNan(data):
    assert (len(data)) > 0
    col_mean = np.nanmean(data, axis=0)
    inds = np.where(np.isnan(data))
    data[inds] = 0# np.take(col_mean, inds[1])
    return data


def getDataFromPath(NormalPath, TumorPath, delimiter, balance=False, isShuffle=True, remove_test=False):
    neg_data = pd.read_csv(NormalPath, sep=delimiter, header=0)
    xNeg = neg_data.iloc[:, 1:].to_numpy(dtype=float)
    xNeg = xNeg
    xNeg = xNeg.transpose()
    if remove_test:
        xNeg = xNeg[:-8]
    xNeg = removeNan(xNeg)

    pos_data = pd.read_csv(TumorPath, sep=delimiter, header=0)
    xPos = pos_data.iloc[:, 1:].to_numpy(dtype=float)
    xPos = xPos  # [:,1:len(x1[0])]
    xPos = xPos.transpose()
    if remove_test:
        xPos = xPos[:-8]
    xPos = removeNan(xPos)

    # assert (len(xPos[0]) == len(xNeg[0]))
    assert not np.isnan(xPos).any() and not np.isnan(xNeg).any()
    if balance:
        xNeg, xPos = balanceNormalSamples(xNeg, xPos)
        assert len(xPos) == len(xNeg)

    # make class labes
    y0 = np.zeros((xNeg.shape[0]))
    y1 = np.ones((xPos.shape[0]))

    dataX = np.append(xNeg, xPos, axis=0)
    dataY = np.append(y0, y1, axis=0)
    dataY = dataY.astype(np.int)
    if isShuffle:
        dataX, dataY = shuffle(dataX, dataY)

    return dataX, dataY


def getData(dataset, type='train',seed=1234):
    neg_data = pd.read_csv(os.path.join(f'smc_track_data',f'{dataset}-Normal-{type}.txt'), sep="\t", header=0)
    x0 = neg_data.iloc[:, 1:].to_numpy(dtype=float)
    x0 = x0
    x0 = x0.transpose()
    x0 = removeNan(x0)
    y0 = np.zeros((x0.shape[0]))

    # retrieve the positive samples
    pos_data = pd.read_csv(os.path.join('smc_track_data',f'{dataset}-Tumor-{type}.txt'), sep="\t", header=0)
    x1 = pos_data.iloc[:, 1:].to_numpy(dtype=float)
    x1 = x1  # [:,1:len(x1[0])]
    x1 = x1.transpose()
    x1 = removeNan(x1)
    assert (len(x1[0]) == len(x0[0]))
    y1 = np.ones((x1.shape[0]))

    dataX = np.append(x0, x1, axis=0)
    dataY = np.append(y0, y1, axis=0)
    dataX = np.where(np.isnan(dataX), 0, dataX)
    dataX, dataY = shuffle(dataX, dataY, random_state=seed)
    dataY = dataY.astype(np.int)
    return dataX, dataY


def bucketizeData(dataX, numbins, epsilon, sensitivity=1):
    for cIndex in range(dataX.shape[1]):
        hist, bin_edges = np.histogram(dataX[:, cIndex], bins=numbins)
        digitized = np.digitize(dataX[:, cIndex], bin_edges)
        # total = 0
        if epsilon == 0:#no privacy case
            dataX[:, cIndex] = digitized
        else:
            probs = np.zeros(shape=(numbins, digitized.shape[0]))
            for bin in range(1, numbins + 1):
                probs[bin - 1] = np.exp(epsilon * abs((numbins - abs(digitized - bin))) / (sensitivity))  #

            total = probs.sum(axis=0)
            probs = probs / np.tile(total.transpose(), (probs.shape[0], 1))
            r = np.random.rand((probs.shape[1]))
            for bin in range(numbins, -1, -1):
                dataX[np.where(probs[0:bin, :].sum(0) >= r), cIndex] = bin
            assert dataX[:, cIndex].min() >= 1
        # assert dataX[:, cIndex].max() >= numbins
        # dataX[:, cIndex] = probs.argmax(axis=0)+1
    return dataX


def getSplitData(dataset):
    data = np.load(os.path.join('public_data', dataset + '_' + 'train.npz'))
    dataTrainX, dataTrainY = data['arr_0'], data['arr_1']
    data.close()
    data = np.load(os.path.join('public_data', dataset + '_' + 'test.npz'))
    dataTestX, dataTestY = data['arr_0'], data['arr_1']
    data.close()

    return dataTrainX, dataTrainY, dataTestX, dataTestY


def getDPData(dataset, EPSILON):
    dataX = np.load(os.path.join('private_data', dataset + '_' + str(EPSILON) + '_expohistogramX.np.npy'))
    dataY = np.load(os.path.join('private_data', dataset + '_' + 'laplaceY.np.npy'))
    return dataX, dataY
