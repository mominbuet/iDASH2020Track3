import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from Utils import getData

# transfer the appropriate data into x and y matrices
# (x for features and y for labels)
# Note: To achieve the best training results, positive and
#      negative samples should be equally represented in
#      the training dataset.

# retrieve the negative samples

dataset = 'BC-TCGA'
dataX, dataY = getData(dataset)
# dataX = 100*dataX

dataX = np.where(dataX < 0, -1, 1)
errors = []
for i in range(20):
    dataX, dataY = shuffle(dataX, dataY)

    clf = tree.DecisionTreeClassifier()
    # clf = AdaBoostClassifier()
    dataTrainX, dataTestX, dataTrainY, dataTestY = train_test_split(dataX, dataY, train_size=int(len(dataX) * 0.8))
    clf = clf.fit(dataTrainX, dataTrainY)

    y_pred = clf.predict(dataTestX)
    errors.append(accuracy_score(dataTestY, y_pred))
print(np.average(errors) * 100)
print(np.min(errors) * 100)
