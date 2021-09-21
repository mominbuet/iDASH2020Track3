from csv import reader
from math import sqrt
from random import randrange
from random import seed

import numpy as np
# Load a CSV file
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from NaiveBayesClassifier import bucketizeData
from Utils import getSplitData





# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(train_set, test_set, algorithm, *args):
    scores = list()

    predicted = algorithm(train_set, test_set, *args)
    actual = [row[-1] for row in test_set]
    accuracy = accuracy_metric(actual, predicted)
    precision, recall, _, _ = precision_recall_fscore_support(actual, predicted, average='micro')
    auc = roc_auc_score(actual, predicted, average='micro')
    print("accuracy {:.3f} precision {:.3f} recall {:.3f} auc {:.3f}".
          format(accuracy, precision * 100, recall * 100, auc))
    scores.append(accuracy)
    return scores


# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return (predictions)


def main():
    # Test the kNN on the Iris Flowers dataset
    seed(1)
    datasets = ['GSE2034', 'BC-TCGA2020', 'GSE25066']
    numbins = 10
    for dataset in datasets:
        print('processing datasets:{}'.format(dataset))
        dataTrainX, dataTrainY, dataTestX, dataTestY = getSplitData(dataset)


        dataTrainX = 100* dataTrainX/ np.linalg.norm(dataTrainX)
        dataTestX = 100 * dataTestX / np.linalg.norm(dataTestX)

        # dataTrainX = bucketizeData(dataTrainX, numbins=numbins, EPSILON=0)
        # dataTestX = bucketizeData(dataTestX, numbins=numbins, EPSILON=0)

        # dataX = dataX * 100
        trainDataset = np.append(dataTrainX, dataTrainY[:, None], axis=1)
        testDataset = np.append(dataTestX, dataTestY[:, None], axis=1)

        num_neighbors = 5
        scores = evaluate_algorithm(trainDataset,testDataset, k_nearest_neighbors, num_neighbors)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))


if __name__ == '__main__':
    main()
