from math import exp, log
from math import pi
from math import sqrt
from random import randrange
from random import seed
from sklearn.manifold import TSNE
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from Utils import getSplitData


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
def evaluate_algorithm(dataset, testDataset, algorithm, n_folds, *args):
    # kf = KFold(n_splits=n_folds, shuffle=True)
    scores = list()
    #     train_set = list(folds)
    #     train_set.remove(fold)
    #     train_set = sum(train_set, [])
    #     test_set = list()
    #     for row in fold:
    #         row_copy = list(row)
    #         test_set.append(row_copy)
    #         row_copy[-1] = None
    #     predicted = algorithm(train_set, test_set, *args)
    #     actual = [row[-1] for row in fold]
    #     accuracy = accuracy_metric(actual, predicted)
    #     scores.append(accuracy)
    # for train_index, test_index in kf.split(dataset):
    train_set = dataset  # [train_index]
    test_set = testDataset  # dataset[test_index]
    predicted = algorithm(train_set, test_set, *args)
    actual = [row[-1] for row in test_set]
    accuracy = accuracy_metric(actual, predicted)
    precision, recall, _, _ = precision_recall_fscore_support(actual, predicted, average='micro')
    auc = roc_auc_score(actual, predicted, average='micro')
    print("accuracy {:.3f} precision {:.3f} recall {:.3f} auc {:.3f}".
          format(accuracy, precision * 100, recall * 100, auc))
    scores.append(accuracy)

    return scores


# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)

    return separated


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del (summaries[-1])
    return summaries


# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    if mean == 0 or stdev == 0:
        return 0
    try:
        exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return log((1 / (sqrt(2 * pi) * stdev)) * exponent)
    except:
        return 0


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] += calculate_probability(row[i], mean, stdev)

    return probabilities


# Predict the class for a given row
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    # print(probabilities)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value

    return best_label


# Naive Bayes Algorithm
def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    return (predictions)




def main():
    seed(1)
    datasets = ['GSE25066', 'GSE2034', 'BC-TCGA2020']
    numbins = 10
    for dataset in datasets:
        print('processing datasets:{}'.format(dataset))
        dataTrainX, dataTrainY, dataTestX, dataTestY = getSplitData(dataset)

        # #bucketize data
        # dataTrainX = bucketizeData(dataTrainX, numbins=numbins, EPSILON=0)
        # dataTestX = bucketizeData(dataTestX, numbins=numbins, EPSILON=0)

        # dataTrainX = np.concatenate((dataTrainX, dataTrainX),axis=0)
        # dataTrainY = np.concatenate((dataTrainY, dataTrainY), axis=0)
        # PCA part

        pca = PCA()
        # normalize
        dataTrainX = dataTrainX / np.linalg.norm(dataTrainX)
        dataTestX = dataTestX / np.linalg.norm(dataTestX)
        # dataTrainX, dataTestX = dataX[:len(dataTrainY)], dataX[len(dataTrainY):]

        dataTrainX = pca.fit_transform(dataTrainX)
        dataTestX = pca.transform(dataTestX)
        # dataTrainX, dataTestX = dataX[:len(dataTrainY)], dataX[len(dataTrainY):]



        # dataX = dataX * 100
        trainDataset = np.append(dataTrainX, dataTrainY[:, None], axis=1)
        testDataset = np.append(dataTestX, dataTestY[:, None], axis=1)

        # evaluate algorithm
        n_folds = 10
        scores = evaluate_algorithm(trainDataset, testDataset, naive_bayes, n_folds)
        # print('Scores: %s' % scores)
        # print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))


if __name__ == "__main__":
    # Test Naive Bayes on Iris Dataset
    main()
