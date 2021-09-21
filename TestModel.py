import configparser
import logging
import pickle
from math import exp, pi, sqrt, isnan, log

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

from DescisionTreeHelper import bagging_predict
from Utils import *

config = configparser.ConfigParser()
config.read('idashTrack3.config')


# Calculate the Gaussian probability distribution function for x
def calculate_probability(testDataRow, mean, stdev):
    assert testDataRow.shape[0] == mean.shape[0] == stdev.shape[0]
    exponent = 0
    for index, val in enumerate(testDataRow):
        stdevVal = 0.0001 if stdev[index] == 0 or isnan(stdev[index]) else stdev[index]
        # stdevVal = stdev[index]
        try:
            exponent += log(exp(-((val - mean[index]) ** 2 / (2 * stdevVal ** 2))) * (1 / (sqrt(2 * pi) * stdevVal)))
        except:
            exponent += 0.0

    return exponent


def getPrediction(dataTestX, class_lengths, stat1, stat2):
    total_rows = sum([class_lengths[label][0] for label in class_lengths])
    probabilities = dict()
    # mismatch = 0
    predicted = []
    for row in dataTestX:
        for class_value in class_lengths.keys():
            # stdDev= np.nan_to_num(stdDev,nan=.001)

            probabilities[class_value] = -1 * (log(
                class_lengths[class_value] / float(total_rows)) + calculate_probability(
                row, stat1[class_value], stat2[class_value]))
            # if isnan(probabilities[class_value]):
            #     probabilities[class_value] = 0.0
        # print(probabilities.values(), end=': ')
        predicted.append(np.asarray(list(probabilities.values())).argmax())
        # mismatch += abs(np.asarray(list(probabilities.values())).argmax() - testDataY)
        # logging.debug(
        #     "probs({:.2f} vs {:.2f}) predicted {}: original {}".format(probabilities[0], probabilities[1],
        #                                                                (np.asarray(list(
        #                                                                    probabilities.values())).argmax()),
        #                                                                testDataY))

    return predicted


def switchOutputs(predicted):
    logging.info("Reversing the output classes")
    predicted = [1 - p for p in predicted]
    return predicted


def main(dataTestX=None, dataTestY=None):
    CustomHistogram = config.getboolean("PrivacyParams", 'CustomHistogram')
    ExponentialHistogram = config.getboolean("PrivacyParams", 'ExponentialHistogram')

    TestTumorFile = config.get("DataInfo", "TestTumorFile")
    TestNormalFile = config.get("DataInfo", "TestNormalFile")
    CSVDelimiter = config.get("DataInfo", "CSVDelimiter")
    multiplier = config.getfloat("TrainingInfo", "multiplier")
    NumBins = config.getint('TrainingInfo', 'NumBins')
    MLAlgo = config.get('TrainingInfo', 'Algorithm')
    ReducedDimension = config.getint("TrainingInfo", "ReducedDimension")
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging._nameToLevel[config.get("TrainingInfo", "DebugLevel")])
    if dataTestX is None:
        dataTestX, dataTestY = getDataFromPath(TestNormalFile, TestTumorFile, CSVDelimiter, balance=False,
                                               isShuffle=True)
        logging.info("logging with file data {}".format(TestTumorFile))
    else:
        logging.info("logging with incoming data")
    logging.info("TestSet size {} Algorithm {}".format(dataTestY.shape, MLAlgo))
    if ExponentialHistogram or CustomHistogram:
        dataTestX = multiplier * dataTestX
        dataTestX = bucketizeData(dataTestX, numbins=NumBins, epsilon=0)

    # load model
    savePath = os.path.join('saved_model', MLAlgo)
    if os.path.exists(savePath):
        with open(savePath + os.path.sep + 'model.pickle', 'rb') as f:
            if MLAlgo == "NAIVEBAYES":
                class_lengths = pickle.load(f)
                stat1 = pickle.load(f)
                stat2 = pickle.load(f)
                switchPredY = pickle.load(f)  # unused, always false
                if ReducedDimension > 0:
                    sortedIndices = pickle.load(f)
                    dataTestX = dataTestX[:, sortedIndices]
                    logging.info("Dimension Reduced to {}".format(dataTestX.shape))

                predicted = getPrediction(dataTestX, class_lengths, stat1, stat2)
                if switchPredY:
                    predicted = switchOutputs(predicted)

            elif MLAlgo == "RFOREST":
                trees = pickle.load(f)
                switchPredY = pickle.load(f)
                if ReducedDimension > 0:
                    sortedIndices = pickle.load(f)
                    dataTestX = dataTestX[:, sortedIndices]
                    logging.info("Dimension Reduced to {}".format(dataTestX.shape))
                logging.info('loaded {} trees from 2 parties'.format(len(trees)))
                predicted = [bagging_predict(trees, row) for row in dataTestX]
                if switchPredY:
                    predicted = switchOutputs(predicted)

            # logging.info("prediction {}".format(predicted))
            # logging.info("true {}".format(dataTestY))
            precision, recall, _, _ = precision_recall_fscore_support(dataTestY, predicted, average='micro')
            auc = roc_auc_score(dataTestY, predicted, average='micro')
            accuracy = accuracy_score(predicted, dataTestY)
            logging.info("accuracy {:.3f} precision {:.3f} recall {:.3f} auc {:.3f}".
                  format(accuracy, precision * 100, recall * 100, auc))
            return (accuracy,precision,recall, auc)
    else:
        logging.error("Save Folder ({}) not found!".format(savePath))
        return (-1,-1,-1,-1)

if __name__ == "__main__":
    main()
