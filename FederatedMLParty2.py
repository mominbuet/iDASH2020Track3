import configparser
import json
import logging
import operator
import socket
from math import sqrt
import time
from sklearn.model_selection import train_test_split

from DescisionTreeHelper import subsample, build_tree
from FederatedMLParty1 import getHistogram, getExponentialHist, aggregateDict, getSums, calcMean, calcVariance, \
    addNoise
from SocketUtils import *
from Utils import *

config = configparser.ConfigParser()
config.read('idashTrack3.config')


# OTHERPORT = MYPORT
# OTHERIP = "127.0.0.1"


# DATASETNAME = 'BC-TCGA'
# EPSILON = 3
# NUMBINS = 10


def setArrays(sock, inputArray, relate):
    for index, x in enumerate(inputArray):
        data = recvData(sock)
        # print(data.decode())
        current = float(data.decode())
        if relate(current, inputArray[index]):
            inputArray[index] = current
        sendData(sock, str(x).encode())
    return inputArray


def main():
    start_time = time.time()
    # set values from config
    Party1Port = config.getint("NetworkIDs", "Party1Port")
    Party1IP = config.get("NetworkIDs", "Party1IP")
    Party2NormalData = config.get("DataInfo", "Party2Normal")
    Party2TumorData = config.get("DataInfo", "Party2Tumor")
    CSVDelimiter = config.get("DataInfo", "CSVDelimiter")

    epsilon = config.getfloat("PrivacyParams", "Epsilon")
    multiplier = config.getfloat("TrainingInfo", "multiplier")
    randomstate = config.getint("TrainingInfo", 'randomstate')

    inputPrivacy = config.getboolean("PrivacyParams", 'InputPrivacy')
    customHistogram = config.getboolean("PrivacyParams", 'CustomHistogram')
    exponentialHistogram = config.getboolean("PrivacyParams", 'ExponentialHistogram')
    noisyMechanism = config.getboolean("PrivacyParams", 'NoisyMechanism')

    numBins = config.getint('TrainingInfo', 'NumBins')
    MLAlgorithm = config.get('TrainingInfo', 'Algorithm')
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging._nameToLevel[config.get("TrainingInfo", "DebugLevel")])
    # dataX, dataY = getData(DATASETNAME)
    # dataTrainX, _, dataTrainY, _ = train_test_split(dataX, dataY, random_state=1235,
    #                                                                 train_size=int(len(dataX) * 0.8))
    #
    # dataX, dataY = dataTrainX[len(dataTrainX) // 2 + 1:], dataTrainY[len(dataTrainY) // 2 + 1:]  # 2nd party
    # dataX = preprocessData(dataX)
    #for competetion
    dataX, dataY = getDataFromPath(Party2NormalData, Party2TumorData, CSVDelimiter, balance=True,
                                   isShuffle=True, remove_test=False)

    dataX, _, dataY, _ = train_test_split(dataX, dataY, random_state=randomstate,
                                          train_size=int(len(dataX) * 0.8))
    indices = np.random.randint(low=len(dataX) // 2-1, high=len(dataX), size=len(dataX))
    dataX = dataX[indices]
    dataY = dataY[indices]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('connecting to {}:{}'.format(Party1IP,Party1Port))
    try:
        sock = connectSocket(sock, Party1IP, Party1Port)
        print("Successfully Connected Party 1, started {}".format(MLAlgorithm))

        inputData = []
        reducedDimension = config.getint("TrainingInfo", "ReducedDimension")
        if reducedDimension > 0:
            dataXScaled = dataX / np.linalg.norm(dataX)
            varianceColumn = dataXScaled.var(axis=0)
            sortedIndices = np.argsort(varianceColumn)
            sendLargemsg(sock, json.dumps(sortedIndices.tolist()))
            sortedIndices = json.loads(recvLargeMsg(sock))
            logging.debug("columns taken {}".format(sortedIndices))
            dataX = dataX[:,sortedIndices[-reducedDimension:]]

        if inputPrivacy:
            if epsilon < 0.1:
                logging.error("Epsilon must be greater than 0.1")

            epsilon /= dataX.shape[1]
            if customHistogram:
                dataX = dataX * multiplier
                minX, maxX = getNoisyMinMax(dataX, epsilon/2)
                minX = setArrays(sock, minX, operator.lt)
                maxX = setArrays(sock, maxX, operator.gt)

                inputData = getHistogram(minX, maxX, dataX, numBins)
                # print(digitized.shape)
                # if CustomExponentialHistogram:
                inputData = getExponentialHist(inputData, numBins, epsilon/2)
            elif exponentialHistogram:
                dataX = dataX * multiplier
                inputData = bucketizeData(dataX, numbins=numBins, epsilon=epsilon)
            else:
                #this should not be used!!
                # norm = np.linalg.norm(dataX)
                # inputData = 100 * (dataX / norm)
                inputData = dataX
                noisyDataX = np.random.laplace(loc=0.0, scale=1 / epsilon, size=(len(dataX), len(dataX[0])))
                inputData = inputData + noisyDataX
        else:
            inputData = dataX

        logging.debug(inputData.shape)
        dataset = np.append(inputData, dataY[:, None], axis=1)
        if MLAlgorithm == 'NAIVEBAYES':
            sums, class_lengths, separated = getSums(dataset)
            if noisyMechanism:
                sums = addNoise(sums, epsilon)
            # sums[class_value].append(len(rows))
            logging.debug(sums)
            sendLargemsg(sock, json.dumps(sums))
            data = recvLargeMsg(sock)
            sums = aggregateDict(sums, data, separated.keys())
            # print(sums)

            sendLargemsg(sock, json.dumps(class_lengths))
            data = recvLargeMsg(sock)
            class_lengths = aggregateDict(class_lengths, data, separated.keys())
            # print(class_lengths)

            stat1 = calcMean(sums, class_lengths)
            logging.debug(stat1)

            partialvariance = calcVariance(separated, stat1, class_lengths)
            sendLargemsg(sock, json.dumps(partialvariance))
            data = recvLargeMsg(sock)
            partialvariance = aggregateDict(partialvariance, data, separated.keys())
            stat2 = dict()
            for key in partialvariance.keys():
                stat2[key] = np.sqrt(partialvariance[key])
            logging.debug(stat1)

        elif  MLAlgorithm == 'RFOREST':
            trees = list()
            n_trees = config.getint("TrainingInfo","n_trees")
            max_depth = config.getint("TrainingInfo","max_depth")
            min_size= config.getint("TrainingInfo","min_size")
            sample_size = 1.0
            n_features = round(sqrt(len(dataX[0]) - 1))
            for i in range(n_trees):
                if sample_size == 1:
                    sample = shuffle(dataset)
                else:
                    sample = subsample(dataset, sample_size)
                tree = build_tree(sample, max_depth, min_size, n_features)
                trees.append(tree)
            sendLargemsg(sock,json.dumps(trees))

        print('finished P2')
        print('finished P1, Execution Time {} seconds'.format(int(time.time() - start_time)))

        data = recvData(sock)
        # print(data.decode())

        sendData(sock, b'done')
        closeSocket(sock)
    except Exception as e:
        print(e)
        closeSocket(sock)
    finally:
        closeSocket(sock)


if __name__ == "__main__":
    main()
