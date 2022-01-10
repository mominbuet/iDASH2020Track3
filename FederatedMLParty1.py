import configparser
import json
import logging
import operator
import pickle
import shutil
import socket
import time
from math import sqrt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRFClassifier

from DescisionTreeHelper import subsample, build_tree, bagging_predict
from SocketUtils import *
from TestModel import getPrediction
from TestModel import main as TestModelRun
from Utils import *

# DATASETNAME = 'BC-TCGA2020'
# DATASETNAME = 'GSE25066'
# default values
# MYPORT = 12125
# OTHERIP = "127.0.0.1"
# EPSILON, SENSITIVITY = 1, 1
# INPUTPRIVACY = False
# CUSTOMHISTOGRAM = False
# EXPONENTIALHISTOGRAM = False
# NOISYMECHANISM = True
# NUMBINS = 10
# ALGO = 'NAIVEBAYES'
# CustomExponentialHistogram = False

config = configparser.ConfigParser()
config.read('idashTrack3.config')


def setArrays(sock, inputArray, relate):
    for index, x in enumerate(inputArray):
        sendData(sock, str(x).encode())
        data = recvData(sock)
        current = float(data.decode())
        if relate(current, x):
            inputArray[index] = current
    return inputArray


def getHistogram(minX, maxX, dataX, numbins):
    histDataX = np.zeros(dataX.shape)

    for col in range(len(dataX[0])):
        bin_edges = [minX[col] + i * (maxX[col] - minX[col]) / numbins for i in range(numbins + 1)]
        histDataX[:, col] = np.digitize(dataX[:, col], bin_edges)
    return histDataX


def getExponentialHist(digitized, numbins, EPSILON, sensitivity=1):
    exponentialNoisyDigitized = digitized
    perColumnEpsilon = EPSILON / digitized.shape[1]
    for cIndex in range(digitized.shape[1]):

        probs = np.zeros(numbins)
        for index, val in enumerate(digitized[:, cIndex]):
            for bin in range(0, numbins + 1):
                probs[bin - 1] = np.exp(perColumnEpsilon * abs(numbins - abs(val - bin)) / (2 * sensitivity))
            total = probs.sum(axis=0)
            probs = probs / np.tile(total.transpose(), (probs.shape[0], 1))
            r = np.random.rand((probs.shape[1]))
            for bin in range(numbins, -1, -1):
                exponentialNoisyDigitized[np.where(probs[0:bin, :].sum(0) >= r), cIndex] = bin
            assert exponentialNoisyDigitized[:, cIndex].min() >= 0
            assert exponentialNoisyDigitized[:, cIndex].max() <= numbins + 1
    return exponentialNoisyDigitized


def aggregateDict(sums, data, class_values):
    data = json.loads(data.decode())
    data = {float(k): [float(i) for i in v] for k, v in data.items()}

    for class_value in class_values:
        sums[class_value] = np.asarray(data[class_value]) + np.asarray(sums[class_value])
        assert (sums[class_value].shape[0] == len(data[class_value]) == len(sums[class_value]))
    return sums


def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector[:-1])

    return separated


def addNoise(sums, EPSILON):
    # split on the number of classes as they are disjoint
    EPSILON /= len(sums.items())
    for class_value, _ in sums.items():
        sums[class_value] = (sums[class_value] + np.random.laplace(loc=0.0, scale=1 / EPSILON,
                                                                   size=(len(sums[class_value])))).tolist()
    return sums


def getSums(dataset):
    separated = separate_by_class(dataset)
    sums = dict()
    class_lengths = dict()
    for class_value, rows in separated.items():
        sums[class_value] = np.round(np.asarray(rows).sum(axis=0), decimals=2).tolist()
        class_lengths[class_value] = [len(rows)]
    return sums, class_lengths, separated


def calcMean(sums, class_lengths):
    mean = dict()
    # total = 0
    # for class_name in class_lengths.keys():
    #     total += class_lengths[class_name][0]
    for class_name in sums.keys():
        mean[class_name] = sums[class_name] / class_lengths[class_name]
    return mean


def calcVariance(inputDataX, mean, class_length):
    partialVariance = dict()
    # total = 0
    # for class_name in class_length.keys():
    #     total += class_length[class_name][0]
    for class_name in mean.keys():
        inputData = np.asarray(inputDataX[class_name])
        variance = ((inputData - np.tile(mean[class_name].transpose(), (inputData.shape[0], 1))) ** 2).sum(axis=0) / (
                class_length[class_name][0] - 1)
        assert inputData.shape[1] == variance.shape[0]
        partialVariance[class_name] = np.round(variance, decimals=2).tolist()
    return partialVariance


def getXGBOOSTModels():
    models = dict()
    # define the number of trees to consider
    n_trees = [10, 50, 100, 500, 1000, 5000]
    for v in n_trees:
        models[str(v)] = XGBRFClassifier(n_estimators=v, subsample=0.9, colsample_bynode=0.2)
    return models


def main():
    start_time = time.time()
    # set values from config
    Party1Port = config.getint("NetworkIDs", "Party1Port")
    Party1IP = "0.0.0.0"  # config.get("NetworkIDs", "Party1IP")

    Party1NormalData = config.get("DataInfo", "Party1Normal")
    Party1TumorData = config.get("DataInfo", "Party1Tumor")

    CSVDelimiter = config.get("DataInfo", "CSVDelimiter")

    # epsilon = config.getfloat("PrivacyParams", "Epsilon")
    multiplier = config.getfloat("TrainingInfo", "multiplier")

    randomstate = config.getint("TrainingInfo", 'randomstate')  # using this to set the make train-test set
    inputPrivacy = config.getboolean("PrivacyParams", 'InputPrivacy')
    checkTrainingAccuracy = config.getboolean("PrivacyParams", 'CheckTrainingAccuracy')

    customHistogram = config.getboolean("PrivacyParams", 'CustomHistogram')
    exponentialHistogram = config.getboolean("PrivacyParams", 'ExponentialHistogram')
    # assert customHistogram != exponentialHistogram, 'Please select either CustomHistogram or ExponentialHistogram'

    noisyMechanism = config.getboolean("PrivacyParams", 'NoisyMechanism')  # experimental
    if not inputPrivacy and not noisyMechanism:
        logging.error("No privacy setting selected! Please put yes to 'NoisyMechanism' or 'InputPrivacy'")
        # return

    NumBins = config.getint('TrainingInfo', 'NumBins')
    MLAlgo = config.get('TrainingInfo', 'Algorithm')

    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging._nameToLevel[config.get("TrainingInfo", "DebugLevel")])

    aucs = []

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # socket.gethostname()
    server_socket.bind((Party1IP, Party1Port))
    server_socket.listen(1)
    print('Waiting for Party 2 on {}:{}'.format(socket.gethostname(), Party1Port))

    allDataX, allDataY = getDataFromPath(Party1NormalData, Party1TumorData, CSVDelimiter, balance=False,
                                   isShuffle=True,
                                   remove_test=False, random_state=randomstate)
    while True:
        # now our endpoint knows about the OTHER endpoint.
        clientsocket, address = server_socket.accept()
        print("Party 1: Accepted connection %s:%s" % (address[0], address[1]))

        for ind in range(randomstate, randomstate + 10):
            epsilon = config.getfloat("PrivacyParams", "Epsilon")
            start_time = time.time()
            # data Section
            # for competetion
            # for random data
            # dataX, dataY = getData(DATASETNAME)
            # only take 80% for training
            currDataX, _, curDataY, _ = train_test_split(allDataX, allDataY, random_state=ind,
                                                         train_size=int(len(allDataX) * 0.8))

            # dataX, dataY = dataTrainX[:len(dataTrainX) // 2], dataTrainY[:len(dataTrainY) // 2]

            # saving a copy before noise to test on this data later
            # dataOriginalX = dataX
            indices = np.random.randint(low=0, high=len(curDataY), size=len(curDataY) // 2)
            currDataX = currDataX[indices]
            curDataY = curDataY[indices]
            dataOriginalX = currDataX
            dataOriginalY = curDataY

            # inputData = []
            sortedIndices = []
            reducedDimension = config.getint("TrainingInfo", "ReducedDimension")
            if reducedDimension > 0:
                dataXScaled = currDataX / np.linalg.norm(currDataX)
                varianceColumn = dataXScaled.var(axis=0)
                sortedIndices = np.argsort(varianceColumn)
                data = recvLargeMsg(clientsocket)
                data = json.loads(data.decode())
                multiplier = 2
                tmp = sortedIndices[-reducedDimension * multiplier:]
                sortedIndices = tmp[
                    np.in1d(sortedIndices[-reducedDimension * multiplier:], data[-reducedDimension * 4:])]
                sendLargemsg(clientsocket, json.dumps(sortedIndices[-reducedDimension:].tolist()))
                currDataX = currDataX[:, sortedIndices[-reducedDimension:]]
                dataOriginalX = dataOriginalX[:, sortedIndices[-reducedDimension:]]

            if inputPrivacy:

                if epsilon == 0:
                    logging.error("Epsilon must be greater than 0 for private input settings")
                else:
                    epsilon /= currDataX.shape[1]
                if customHistogram:
                    # increase the
                    currDataX = currDataX * multiplier
                    minX, maxX = getNoisyMinMax(currDataX, epsilon / 2)  # spending half of the budget for noisy min-max
                    minX = setArrays(clientsocket, minX, operator.lt)
                    maxX = setArrays(clientsocket, maxX, operator.gt)

                    inputData = getHistogram(minX, maxX, currDataX, NumBins)
                    # if CustomExponentialHistogram:
                    inputData = getExponentialHist(inputData, NumBins, epsilon / 2)

                elif exponentialHistogram:
                    currDataX = currDataX * multiplier
                    inputData = bucketizeData(currDataX, numbins=NumBins, epsilon=epsilon)
                else:
                    # this should not be used!! Sensitivity not okay
                    # norm = np.linalg.norm(dataX)
                    inputData = currDataX
                    # inputData = 100 * (dataX / norm)
                    noisyDataX = np.random.laplace(loc=0.0, scale=1 / epsilon, size=(len(currDataX), len(currDataX[0])))
                    inputData = inputData + noisyDataX + 0.00001
            else:
                # no input privacy
                currDataX = currDataX * multiplier
                inputData = currDataX

            logging.info("Algorithm {} input shape {} Epsilon {}".format(MLAlgo, inputData.shape,
                                                                         config.getfloat("PrivacyParams", "Epsilon")))
            dataset = np.append(inputData, curDataY[:, None], axis=1)
            accuracy = 1.0
            if MLAlgo == 'NAIVEBAYES':
                sums, class_lengths, separated = getSums(dataset)
                if noisyMechanism: #add noice to the sum values rather than data
                    sums = addNoise(sums, epsilon)
                # sums[class_value].append(len(rows))
                # logging.debug(sums)
                data = recvLargeMsg(clientsocket)
                sendLargemsg(clientsocket, json.dumps(sums))
                sums = aggregateDict(sums, data, separated.keys())
                # logging.debug(sums)

                data = recvLargeMsg(clientsocket)
                sendLargemsg(clientsocket, json.dumps(class_lengths))
                class_lengths = aggregateDict(class_lengths, data, separated.keys())
                # print(class_lengths)

                stat1 = calcMean(sums, class_lengths)
                logging.debug(stat1)

                partialvariance = calcVariance(separated, stat1, class_lengths)

                data = recvLargeMsg(clientsocket)
                sendLargemsg(clientsocket, json.dumps(partialvariance))
                partialvariance = aggregateDict(partialvariance, data, separated.keys())

                stat2 = dict()
                for key in partialvariance.keys():
                    stat2[key] = np.sqrt(partialvariance[key])
                logging.debug(stat2)
                # train accuracy

                if checkTrainingAccuracy:
                    accuracy = accuracy_score(getPrediction(dataOriginalX, class_lengths, stat1, stat2),
                                              dataOriginalY)
                    logging.info('checking training accuracy: {}'.format(accuracy))

                savePath = os.path.join('saved_model', MLAlgo)
                if os.path.exists(savePath):
                    shutil.rmtree(savePath)
                os.makedirs(savePath)
                with open(savePath + os.path.sep + 'model.pickle', 'wb') as f:
                    pickle.dump(class_lengths, f, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle.dump(stat1, f, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle.dump(stat2, f, protocol=pickle.HIGHEST_PROTOCOL)
                    # if accuracy 0 for traininig then switch the predictions
                    pickle.dump(accuracy < .5, f, protocol=pickle.HIGHEST_PROTOCOL)
                    if len(sortedIndices) > 0:
                        pickle.dump(sortedIndices[-reducedDimension:], f,
                                    protocol=pickle.HIGHEST_PROTOCOL)  # accuracy < 0.55
                    logging.info("saved model into {}".format(savePath))

            elif MLAlgo == 'RFOREST':
                trees = list()
                n_trees = config.getint("TrainingInfo", "n_trees")
                max_depth = config.getint("TrainingInfo", "max_depth")
                min_size = config.getint("TrainingInfo", "min_size")
                sample_size = 1.0
                n_features = round(sqrt(len(currDataX[0]) - 1))
                for i in range(n_trees):
                    if sample_size == 1:
                        sample = shuffle(dataset)
                    else:
                        sample = subsample(dataset, sample_size)
                    tree = build_tree(sample, max_depth, min_size, n_features)
                    trees.append(tree)
                if checkTrainingAccuracy:
                    predicted = [bagging_predict(trees, row) for row in dataOriginalX]
                    accuracy = accuracy_score(predicted, dataOriginalY)
                    logging.info('checking training accuracy: {}'.format(accuracy))

                p2trees = recvLargeMsg(clientsocket)
                p2trees = json.loads(p2trees.decode())
                for p2tree in p2trees:
                    trees.append(p2tree)
                savePath = os.path.join('saved_model', MLAlgo)
                if os.path.exists(savePath):
                    shutil.rmtree(savePath)
                os.makedirs(savePath)

                with open(savePath + os.path.sep + 'model.pickle', 'wb') as f:
                    pickle.dump(trees, f, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle.dump(accuracy < .35, f, protocol=pickle.HIGHEST_PROTOCOL)
                    if len(sortedIndices) > 0:
                        pickle.dump(sortedIndices[-reducedDimension:], f, protocol=pickle.HIGHEST_PROTOCOL)
                    logging.info("saved model into {}".format(savePath))

            elif MLAlgo == 'XGBOOST':
                p2DataY = recvLargeMsg(clientsocket)
                p2DataX = recvLargeMsg(clientsocket)

                p2DataX = json.loads(p2DataX.decode())
                p2DataY = json.loads(p2DataY.decode())
                joinedX = [*inputData, *p2DataX]
                joinedY = [*curDataY, *p2DataY]
                model = XGBRFClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.2)
                model.fit(joinedX, joinedY)
                savePath = os.path.join('saved_model', MLAlgo)
                if os.path.exists(savePath):
                    shutil.rmtree(savePath)
                os.makedirs(savePath)
                accuracy = 1.0  # dont switch
                with open(savePath + os.path.sep + 'model.pickle', 'wb') as f:
                    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle.dump(accuracy < .35, f, protocol=pickle.HIGHEST_PROTOCOL)
                    if len(sortedIndices) > 0:
                        pickle.dump(sortedIndices[-reducedDimension:], f, protocol=pickle.HIGHEST_PROTOCOL)
                    logging.info("saved model into {}".format(savePath))

            print('finished P1, Train Execution Time {} seconds'.format(int(time.time() - start_time)))


            # TestTumorFile = config.get("DataInfo", "Party2Tumor")
            # TestNormalFile = config.get("DataInfo", "Party2Normal")
            ###
            # currDataX, allDataY = getDataFromPath(TestNormalFile, TestTumorFile, CSVDelimiter, balance=False, isShuffle=True)
            # TestTumorFile = config.get("DataInfo", "Party1Tumor")
            # TestNormalFile = config.get("DataInfo", "Party1Normal")
            # tmpX, tmpY = getDataFromPath(TestNormalFile, TestTumorFile, CSVDelimiter, balance=False, isShuffle=True)
            # currDataX =np.concatenate((currDataX,tmpX))
            # allDataY = np.concatenate((allDataY,tmpY))
            # TestTumorFile = config.get("DataInfo", "TestTumorFile")
            # TestNormalFile = config.get("DataInfo", "TestNormalFile")
            # tmpX, tmpY = getDataFromPath(TestNormalFile, TestTumorFile, CSVDelimiter, balance=False, isShuffle=True)
            # currDataX = np.concatenate((currDataX, tmpX))
            # allDataY = np.concatenate((allDataY, tmpY))

            # currDataX, allDataY = getDataFromPath(TestNormalFile, TestTumorFile, CSVDelimiter, balance=False, isShuffle=True)
            _, dataTestX, _, dataTestY = train_test_split(allDataX, allDataY, random_state=ind,
                                                          train_size=int(len(allDataX) * 0.8))
            start_time = time.time()
            _, _, _, auc = TestModelRun(dataTestX, dataTestY)
            print('finished P1, Test Execution Time {} seconds'.format((time.time() - start_time)))

            # this auc is averaged over X times with X different runs
            aucs.append(auc)

        sendData(clientsocket, b'done')
        data = recvData(clientsocket)
        if data == b'done':
            closeSocket(clientsocket)
            break

    closeSocket(server_socket)

# for i in range(10):
#     _, testDataX, _, testDataY = train_test_split(allDataX, allDataY, train_size=int(len(allDataY) * 0.8), shuffle=True)
#     accuracy, precision, recall, auc = TestModelRun(testDataX, testDataY)
#     # print("accuracy {:.3f} precision {:.3f} recall {:.3f} auc {:.3f}".
#     #       format(accuracy, precision * 100, recall * 100, auc))
#     aucs.append(auc)
    print(f'test auc {aucs[0]} average auc {np.average(aucs)}')
# TestModelRun()


if __name__ == "__main__":
    main()
