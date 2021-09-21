import torch.nn as nn
import torch.nn.functional as F


class NetLarge(nn.Module):
    def __init__(self):
        super(NetLarge, self).__init__()
        self.conv1 = nn.Conv1d(32, 64, 8, padding=8, stride=1)
        # self.conv1_bn = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(64, 128, 16, padding=16, stride=2)
        self.conv3 = nn.Conv1d(128, 128, 16, padding=16, stride=2)
        # self.conv2_bn = nn.BatchNorm1d(128)
        # self.conv3 = nn.Conv1d(128, 256, 16, padding=16, stride=2)
        # self.conv1_bn = nn.BatchNorm1d(26)
        # self.max2 = nn.MaxPool1d(3, stride=2)
        # self.conv2 = nn.Conv1d(128, 256, 32, padding=32, stride=2)
        # self.conv2 = nn.Conv1d(17814//2, 17814//4, 16, padding=16, stride=2)
        # self.conv3 = nn.Conv1d(17814 // 4, 17814 // 8, 32, padding=32, stride=2)
        # self.conv4 = nn.Conv1d(17814 // 8, 17814 // 16, 64, padding=64, stride=2)
        self.drop25 = nn.Dropout(p=0.25)
        self.drop5 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 5, 256)
        self.fc2 = nn.Linear(256, 2)
        # self.fc3 = nn.Linear(512, 2)
        # self.drop2 = nn.Dropout(p=0.5)

    def forward(self, x):
        #
        # print(x.shape)
        x = self.pool3(F.relu(self.conv1(x)))
        # x = self.conv1_bn(x)
        # print(x.shape)
        x = self.pool3(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        # x = self.drop25(x)
        # print(x.shape)
        # x = self.conv2_bn(x)

        # print(x.shape)
        # logging.info("after 2nd conv {}".format(x.shape))
        # x = F.max_pool1d(F.relu(self.conv3(x)), 4, stride=2)
        # logging.info("after 3rd conv {}".format(x.shape))
        # x = F.max_pool1d(x, 2)
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.max_pool1d(x, 2)
        # print(x.shape)
        x = x.view(-1, x.shape[-2] * x.shape[-1])
        x = self.drop5(F.relu(self.fc1(x)))
        # x = self.drop5(F.relu(self.fc2(x)))
        # logging.info("after 1st fc {}".format(x.shape))
        x = self.fc2(x)
        # print(x.shape)
        return F.log_softmax(x, dim=1)


class NetSmall(nn.Module):
    def __init__(self):
        super(NetSmall, self).__init__()
        self.conv1 = nn.Conv1d(32, 64, 8, padding=8, stride=1)
        # self.conv1_bn = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(64, 128, 16, padding=16, stride=2)
        # self.conv2_bn = nn.BatchNorm1d(128)
        # self.conv3 = nn.Conv1d(128, 256, 16, padding=16, stride=2)
        # self.conv1_bn = nn.BatchNorm1d(26)
        # self.max2 = nn.MaxPool1d(3, stride=2)
        # self.conv2 = nn.Conv1d(128, 256, 32, padding=32, stride=2)
        # self.conv2 = nn.Conv1d(17814//2, 17814//4, 16, padding=16, stride=2)
        # self.conv3 = nn.Conv1d(17814 // 4, 17814 // 8, 32, padding=32, stride=2)
        # self.conv4 = nn.Conv1d(17814 // 8, 17814 // 16, 64, padding=64, stride=2)
        self.drop25 = nn.Dropout(p=0.25)
        self.drop5 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 5, 256)
        self.fc2 = nn.Linear(256, 2)
        # self.fc3 = nn.Linear(512, 2)
        # self.drop2 = nn.Dropout(p=0.5)

    def forward(self, x):
        #
        # print(x.shape)
        x = self.pool3(F.relu(self.conv1(x)))
        # x = self.conv1_bn(x)
        # print(x.shape)
        x = self.pool3(F.relu(self.conv2(x)))
        x = x.view(-1, x.shape[-2] * x.shape[-1])
        x = self.drop5(F.relu(self.fc1(x)))
        # x = self.drop5(F.relu(self.fc2(x)))
        # logging.info("after 1st fc {}".format(x.shape))
        x = self.fc2(x)
        # print(x.shape)
        return F.log_softmax(x, dim=1)
