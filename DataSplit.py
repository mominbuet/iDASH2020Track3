from math import ceil
import pandas as pd
from sklearn.model_selection import train_test_split

from Utils import *


def getXY(dataXTrainPos, dataXTrainNeg):
    dataXTrain = np.concatenate((dataXTrainPos, dataXTrainNeg))
    dataYTrain = np.concatenate((np.ones(dataXTrainPos.shape[0]), np.zeros(dataXTrainNeg.shape[0])))
    dataXTrain, dataYTrain = shuffle(dataXTrain, dataYTrain)
    return dataXTrain, dataYTrain


def balanceNormalSamples(negativeX, positiveX):
    if positiveX.shape[0] > negativeX.shape[0]:
        negativeX = negativeX.tolist()
        negativeX = negativeX * ceil(len(positiveX) / len(negativeX))
        negativeX = np.asarray(negativeX[:len(positiveX)])
    else:
        positiveX = positiveX.tolist()
        positiveX = negativeX * ceil(len(negativeX) / len(positiveX))
        positiveX = np.asarray(positiveX[:len(negativeX)])

    return negativeX, positiveX


def makeSplits():
    datasets = ['BC-TCGA2020', 'GSE2034', 'GSE25066']
    for dataset in datasets:
        print(dataset)
        dataX, dataY = getData(dataset)
        dataXPos = dataX[np.where(dataY == 1)]
        dataXNeg = dataX[np.where(dataY == 0)]
        print('Positives:Negatives {}:{}'.format(dataXPos.shape[0], dataXNeg.shape[0]))

        # 80:20 split of both tumor and normal samples
        dataXTrainPos, dataXTestPos = train_test_split(dataXPos, test_size=int(len(dataXPos) * .2))
        dataXTrainNeg, dataXTestNeg = train_test_split(dataXNeg, test_size=int(len(dataXNeg) * .2))

        # balance normal samples to match tumor samples
        dataXTrainNeg, dataXTrainPos = balanceNormalSamples(dataXTrainNeg, dataXTrainPos)
        dataXTestNeg, dataXTestPos = balanceNormalSamples(dataXTestNeg, dataXTestPos)
        # print('Balanced Positives:Negatives {}:{}'.format(dataXTrainNeg.shape[0], dataXTrainPos.shape[0]))

        # Shuffle and add class labels
        dataXTrain, dataYTrain = getXY(dataXTrainPos, dataXTrainNeg)
        dataXTest, dataYTest = getXY(dataXTestPos, dataXTestNeg)

        # np.savez(os.path.join('public_data', dataset + "_train"), dataXTrain, dataYTrain)
        # np.savez(os.path.join('public_data', dataset + "_test"), dataXTest, dataYTest)


# def main(normalFilePath,tumorFilePath):
#     # os.path.join('smc_track_data', dataset + '-Normal-train.txt')
#     neg_data = pd.read_csv(normalFilePath, sep="\t", header=0)
#     x0 = neg_data.iloc[:, 1:].to_numpy(dtype=float)
#     x0 = x0
#     x0 = x0.transpose()
#     x0P1 = x0[:len(x0)//2]
#     x0P2 = x0[len(x0) // 2:]
#
#     # retrieve the positive samples
#     pos_data = pd.read_csv(tumorFilePath, sep="\t", header=0)
#     x1 = pos_data.iloc[:, 1:].to_numpy(dtype=float)
#     x1 = x1  # [:,1:len(x1[0])]
#     x1 = x1.transpose()
#     assert (len(x1[0]) == len(x0[0]))
#
#     x1P1 = x0[:len(x0)//2]
#     x1P2 = x0[len(x0) // 2:]
#
#
# if __name__ == "__main__":
#     main()
