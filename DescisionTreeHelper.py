# Random Forest Algorithm on Sonar Dataset
from csv import reader
from math import sqrt
from random import randrange
from random import seed

# from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from Utils import *




# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, testDataset, algorithm, n_folds, *args):
    # folds = cross_validation_split(dataset, n_folds)
    scores = list()
    # kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    # for train_index, test_index in kf.split(X=dataset[:,0:len(dataset[0])-2], y=dataset[:,-1]):
    # train_set = dataset  # [train_index]
    # test_set = testDataset
    predicted = algorithm(dataset, testDataset, *args)
    actual = [row[-1] for row in testDataset]
    accuracy = accuracy_metric(actual, predicted)
    precision, recall, _, _ = precision_recall_fscore_support(actual, predicted, average='micro')
    auc = roc_auc_score(actual, predicted, average='micro')
    print("accuracy {:.3f} precision {:.3f} recall {:.3f} auc {:.3f}".
          format(accuracy, precision * 100, recall * 100, auc))
    scores.append(accuracy)

    return scores


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Select the best split point for a dataset
def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0]) - 1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        if sample_size == 1:
            sample = shuffle(train)
        else:
            sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return (predictions)


import configparser

config = configparser.ConfigParser()
config.read('idashTrack3.config')


def main():
    seed(1)
    Party1NormalData = config.get("DataInfo", "Party1Normal")
    Party1TumorData = config.get("DataInfo", "Party1Tumor")
    CSVDelimiter = config.get("DataInfo", "CSVDelimiter")
    NCOMPONENTS = 50
    datasets = ['GSE2034', 'GSE25066', 'BC-TCGA2020']  #
    for dataset in datasets:
        print('processing datasets:{}'.format(dataset))
        dataX, dataY = getData(dataset)

        dataTrainX, dataTestX, dataTrainY, dataTestY = train_test_split(dataX, dataY, test_size=int(0.2 * len(dataY)))

        # PCA part
        # pca = PCA(n_components=NCOMPONENTS)
        # # # normalize
        # # dataX1 = dataX1 / np.linalg.norm(dataX1)
        # # dataX2 = dataX2 / np.linalg.norm(dataX2)
        # dataTrainX = dataTrainX / np.linalg.norm(dataTrainX)
        # dataTestX = dataTestX / np.linalg.norm(dataTestX)
        # # # dataTrainX, dataTestX = dataX[:len(dataTrainY)], dataX[len(dataTrainY):]
        # #
        # dataTrainX = pca.fit_transform(dataTrainX)
        # dataTestX = pca.transform(dataTestX)
        # pca = PCA(n_components=NCOMPONENTS)
        # dataTrainX = np.append(newDataTrainX, pca.fit_transform(dataX2), axis=0)

        # dataTrainX, dataTestX = dataX[:len(dataTrainY)], dataX[len(dataTrainY):]
        # bucketize data
        dataTrainX = 100 * dataTrainX
        dataTestX = 100 * dataTestX
        dataTrainX = bucketizeData(dataTrainX, numbins=10, epsilon=0)
        dataTestX = bucketizeData(dataTestX, numbins=10, epsilon=0)

        # dataX = dataX * 100
        trainDataset = np.append(dataTrainX, dataTrainY[:, None], axis=1)
        testDataset = np.append(dataTestX, dataTestY[:, None], axis=1)

        # evaluate algorithm
        n_folds = 2
        max_depth = 5
        min_size = 1
        sample_size = 0.8
        n_features = int(sqrt(len(trainDataset[0]) - 1))
        for n_trees in [10, 15, 20, ]:
            scores = evaluate_algorithm(trainDataset, testDataset, random_forest, n_folds, max_depth, min_size,
                                        sample_size, n_trees, n_features)
            print('Trees: {} {} {:.3f}'.format(n_trees, scores, (sum(scores) / float(len(scores)))))
        print('\n\n')


if __name__ == "__main__":
    # Test Naive Bayes on Iris Dataset
    main()
