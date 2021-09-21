import glob
import logging
import os

import numpy as np
import syft as sy
import torch as th
import torch.optim as optim
from numpy import expand_dims
from sklearn.decomposition import PCA
from syft.grid.public_grid import PublicGridNetwork
from torch import Tensor, long
from torch import save as saveModel
from torch import load as loadModel
from torch.utils.data import TensorDataset, DataLoader

from FederatedDataPreprocess import DATASET, EPSILON, NCOMPONENTS
from NaiveBayesClassifier import bucketizeData
from NetFiles import *
from Utils import getSplitData

# from torch import device,cuda

grid_address = "http://network:7000"  # address
N_EPOCHS = 200  # number of epochs to train
N_TEST = 10  # number of test
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class Arguments():
    def __init__(self):
        self.test_batch_size = N_TEST
        self.epochs = N_EPOCHS
        self.lr = 0.01
        self.log_interval = 5
        self.device = th.device("cpu")
        # self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


args = Arguments()


def epoch_total_size(data):
    total = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            total += data[i][j].shape[0]

    return total


def main():
    print("Dataset {}".format(DATASET))
    hook = sy.TorchHook(th)

    # Connect direcly to grid nodes
    my_grid = PublicGridNetwork(hook, grid_address)

    model = NetSmall()
    print(model)
    model.to(args.device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    data = my_grid.search("#" + DATASET + str(EPSILON), "#X")  # images
    target = my_grid.search("#" + DATASET + str(EPSILON), "#Y")  # labels
    assert len(data) > 1
    data = list(data.values())  # returns a pointer
    target = list(target.values())  # returns a pointer

    # print(data)
    # print(target)

    dataTrainX, dataTrainY, dataTestX, dataTestY = getSplitData(DATASET)
    dataTestY = dataTestY.astype(np.int)

    ##reduce dimension
    pca = PCA(n_components=NCOMPONENTS)
    # normalize
    dataTrainX = dataTrainX / np.linalg.norm(dataTrainX)
    dataTestX = dataTestX / np.linalg.norm(dataTestX)
    # dataTrainX, dataTestX = dataX[:len(dataTrainY)], dataX[len(dataTrainY):]

    pca.fit(dataTrainX)
    dataTestX = pca.transform(dataTestX)
    # if CustomExponentialHistogram or ExponentialHistogram or CustomHistogram:
    dataTestX = bucketizeData(dataTestX, numbins=10, EPSILON=0)

    dataTestX = expand_dims(dataTestX, axis=2)

    testset = TensorDataset(Tensor(dataTestX), Tensor(dataTestY).type(long))
    testloader = DataLoader(testset, batch_size=20, shuffle=True)
    logging.info("test dataset size {}".format(len(dataTestY)))

    currentMaxAccuracy = 0

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,)),  # mean and std
    # ])
    # testset = datasets.MNIST('./dataset', download=False, train=False, transform=transform)
    # testloader = th.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)

    def train(args):
        model.train()
        epoch_total = epoch_total_size(data)

        current_epoch_size = 0
        for i in range(len(data)):
            j = 0
            # for j in range(len(data[i])):
            current_epoch_size += len(data[i][j])
            worker = data[i][j].location  # worker hosts data
            # print("current worker {}".format(worker.id))
            model.send(worker)  # send model to PyGridNode worker
            optimizer.zero_grad()

            pred = model(data[i][j])
            loss = F.nll_loss(pred, target[i][j])
            loss.backward()

            optimizer.step()
            model.get()  # get back the model

            loss = loss.get()

            if epoch % args.log_interval == 0:
                print('Train Epoch: {} | With {} data |: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, worker.id, current_epoch_size, epoch_total,
                    100. * current_epoch_size / epoch_total, loss.item()))

    def test(args, currentMaxAccuracy):
        if epoch % args.log_interval == 0:

            model.eval()
            test_loss = 0
            correct = 0
            with th.no_grad():
                for data, target in testloader:
                    data, target = data.to(args.device), target.to(args.device)
                    output = model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(testloader.dataset)
            accuracy = 100. * correct / len(testloader.dataset)

            print('Epoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(epoch,
                                                                                              test_loss, correct,
                                                                                              len(testloader.dataset),
                                                                                              accuracy))
            if accuracy > currentMaxAccuracy:
                if not os.path.exists(os.path.join('saved_models', DATASET)):
                    os.mkdir(os.path.join('saved_models', DATASET))
                else:
                    files = glob.glob('saved_models/' + DATASET + '/*')
                    if len(files) > 5:
                        files = sorted(files, key=lambda x: int((x.split('_')[-1].replace('.pt', ''))))
                        os.remove(files[0])

                saveModel(model.state_dict(), '{}{}model_{}_{:.0f}.pt'
                          .format(os.path.join('saved_models', DATASET), os.path.sep, epoch, accuracy))
                return accuracy
        return currentMaxAccuracy

    if os.path.exists(os.path.join('saved_models', DATASET)):
        files = glob.glob('saved_models/' + DATASET + '/*')
        files = sorted(files, key=lambda x: int((x.split('_')[-1].replace('.pt', ''))))
        currentMaxAccuracy = int(files[-1].split('_')[-1].replace('.pt', ''))
        model.load_state_dict(loadModel(files[-1]))#-1 for highest accuracy


    for epoch in range(N_EPOCHS):
        train(args)
        # if (epoch % 3 == 0):
        currentMaxAccuracy = test(args, currentMaxAccuracy)


if __name__ == "__main__":
    main()
