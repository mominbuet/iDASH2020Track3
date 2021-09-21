import os

import numpy as np

from Utils import getData

EPSILONs, SENSITIVITY = [5,10,15], 1
DATASETNAME = 'BC-TCGA'
NUMBINS = 10
dataX, dataY = getData(DATASETNAME)
for EPSILON in EPSILONs:
    digitized, noisyDigitized, exponentialNoisyDigitized = np.zeros(dataX.shape), np.zeros(dataX.shape), np.zeros(
        dataX.shape)
    dataX *= 100
    dataX = dataX.astype(np.int32)

    noisyDataX = np.random.laplace(loc=0.0, scale=1 / EPSILON, size=(len(dataX), len(dataX[0])))
    noisyDataX = dataX + noisyDataX
    np.save(os.path.join('private_data', DATASETNAME + '_'+str(EPSILON)+'_laplaceX.np'), noisyDataX)
    # np.save(os.path.join('private_data', DATASETNAME + '_laplaceY.np'), dataY)

    for cIndex in range(dataX.shape[1]):

        hist, bin_edges = np.histogram(dataX[:, cIndex], bins=NUMBINS)
        noisyHist, noisybin_edges = np.histogram(noisyDataX, bins=NUMBINS)
        # max = np.max(dataX[:, index])
        # noisyMax = np.max(noisyDataX)
        digitized = np.digitize(dataX[:, cIndex], bin_edges)
        # noisyDigitized = np.digitize(noisyDataX[:, cIndex], bin_edges)

        total = 0
        probs = np.zeros(shape=(NUMBINS, digitized.shape[0]))
        for bin in range(0, NUMBINS + 1):
            probs[bin - 1] = np.exp(EPSILON * abs(NUMBINS - abs(digitized - bin)) / (2 * SENSITIVITY))

        exponentialNoisyDigitized[:, cIndex] = probs.argmax(axis=0) + 1
        # np.save(os.path.join('private_data', DATASETNAME + '_' + str(EPSILON) + '_histogramX.np'), noisyDigitized)
    np.save(os.path.join('private_data', DATASETNAME + '_' + str(EPSILON) + '_expohistogramX.np'),
                exponentialNoisyDigitized)
    print("Done " + str(EPSILON))
