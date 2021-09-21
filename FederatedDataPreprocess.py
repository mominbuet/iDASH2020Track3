import requests
import syft as sy
import torch
from numpy import expand_dims
from sklearn.decomposition import PCA
# websocket client. It sends commands to the node servers
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient
from torch.utils.data import TensorDataset

from NaiveBayesClassifier import bucketizeData
from Utils import *

alice_address = "http://alice:5000"
bob_address = "http://bob:5001"

DATASET = 'GSE25066'
EPSILON = 0
NCOMPONENTS = 32


def main():
    hook = sy.TorchHook(torch)

    # Connect direcly to grid nodes
    compute_nodes = {}

    compute_nodes["Alice"] = DataCentricFLClient(hook, alice_address)
    compute_nodes["Bob"] = DataCentricFLClient(hook, bob_address)

    # Check if they are connected
    for key, value in compute_nodes.items():
        print("Is " + key + " connected?: " + str(value.ws.connected))

    dataTrainX, dataTrainY, dataTestX, dataTestY = getSplitData(DATASET)
    dataTestY = dataTestY.astype(np.int)
    dataTrainY = dataTrainY.astype(np.int)

    ##reduce dimension
    pca = PCA(n_components=NCOMPONENTS)
    # normalize
    dataTrainX = dataTrainX / np.linalg.norm(dataTrainX)
    dataTestX = dataTestX / np.linalg.norm(dataTestX)
    # dataTrainX, dataTestX = dataX[:len(dataTrainY)], dataX[len(dataTrainY):]

    dataTrainX = pca.fit_transform(dataTrainX)
    dataTestX = pca.transform(dataTestX)
    # if CustomExponentialHistogram or ExponentialHistogram or CustomHistogram:
    dataTrainX = bucketizeData(dataTrainX, numbins=10, epsilon=EPSILON)

    dataTrainX = expand_dims(dataTrainX, axis=2)

    # N_SAMPLES = dataTrainX.shape[0]  # Number of samples
    # DIMENSION = len(dataX[0])
    # dataTrainY = dataTrainY.astype(np.int32)
    # dataTestY = dataTestY.astype(np.int32)

    trainset = torch.utils.data.TensorDataset(torch.Tensor(dataTrainX),
                                              torch.Tensor(dataTrainY).type(torch.long))  # create your datset
    # print(trainset)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=dataTrainX.shape[0], shuffle=True)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,)),  # mean and std
    # ])
    # MNIST_PATH = './dataset'  # Path to save MNIST dataset
    #
    # # Download and load MNIST dataset
    # trainset = datasets.MNIST(MNIST_PATH, download=True, train=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=10000, shuffle=True)

    dataiter = iter(trainloader)
    trainX, trainY = dataiter.next()  # Train images and their labels
    dataset_trainX = torch.split(trainX, int(dataTrainX.shape[0] / len(compute_nodes)),
                                 dim=0)
    dataset_trainY = torch.split(trainY, int(dataTrainY.shape[0] / len(compute_nodes)),
                                 dim=0)

    for index, _ in enumerate(compute_nodes):
        dataset_trainX[index] \
            .tag("#X", "#" + DATASET + str(EPSILON)) \
            .describe("The input datapoints to the " + DATASET + " dataset.")

        dataset_trainY[index] \
            .tag("#Y", "#" + DATASET + str(EPSILON)) \
            .describe("The input labels to the " + DATASET + " dataset.")

    for index, key in enumerate(compute_nodes):
        print("Sending data to", key)

        dataset_trainX[index].send(compute_nodes[key], garbage_collect_data=False)
        dataset_trainY[index].send(compute_nodes[key], garbage_collect_data=False)

    print("Alice's tags: ", requests.get(alice_address + "/data-centric/dataset-tags").json())
    print("Bob's tags: ", requests.get(bob_address + "/data-centric/dataset-tags").json())


if __name__ == "__main__":
    main()
