import torch
from torch import nn
import torch.nn.functional as func


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(768, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, batch):
        x = self.fc1(batch)
        x = func.relu(x)
        x = self.fc2(x)
        return func.relu(x)


class Net2(nn.Module):  # 87% with k = [2, 4, 8, 16, 32, 64], k_amount = 64
    def __init__(self):
        super(Net2, self).__init__()
        # dropout=.3, channels=16, kernels=[2, 4, 8, 16, 32, 64], classes=2
        dropout = .3
        channels = 16
        kernels = [2, 4, 8, 16, 32, 64]
        classes = 2

        self.conv1 = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # self.norm = nn.BatchNorm1d()

        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=750),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(dropout),
        # )
        self.lin = nn.Sequential(
            nn.Linear(channels * len(kernels), 32),
            nn.Linear(32, classes),
        )

    def forward(self, batch):
        x = torch.reshape(batch, (-1, 1, 768))
        x = [self.relu(conv(x)) for conv in self.conv1]
        kernel_sizes = [767, 765, 761, 753, 737, 705]
        x = [func.max_pool1d(i, k) for i, k in zip(x, kernel_sizes)]
        # x = [func.max_pool2d(l, (1, l.size(2).item())) for l in x]
        x = torch.cat(x, dim=1).squeeze(2)
        x = self.dropout(x)
        # out = self.drop(out)
        # logits = self.fc(out.squeeze(2))
        logits = self.lin(x)
        return logits


class Net3(nn.Module):  # 87% with k = [2, 4, 8, 16, 32, 64], k_amount = 64
    def __init__(self):
        super(Net3, self).__init__()
        # dropout=.3, channels=16, kernels=[2, 4, 8, 16, 32, 64], classes=2
        dropout = .3
        channels = 16
        kernels = [2, 4, 8, 16, 32]
        classes = 2

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=8)
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16)
        self.conv5 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=32)
        self.dropout = nn.Dropout(dropout)
        # self.norm = nn.BatchNorm1d()

        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=750),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(dropout),
        # )
        self.lin = nn.Sequential(
            nn.Linear(400, 32),
            nn.Linear(32, classes),
        )

    def forward(self, batch):
        x = batch.view(len(batch), 1, 768)
        x1 = func.relu(self.conv1(x)).view(len(batch), 16, 1, 767)
        x2 = func.relu(self.conv2(x)).view(len(batch), 16, 1, 765)
        x3 = func.relu(self.conv3(x)).view(len(batch), 16, 1, 761)
        x4 = func.relu(self.conv4(x)).view(len(batch), 16, 1, 753)
        x5 = func.relu(self.conv5(x)).view(len(batch), 16, 1, 737)
        x1 = func.max_pool2d(x1, kernel_size=(150, 150), padding=0)
        x2 = func.max_pool2d(x2, kernel_size=(150, 150), padding=0)
        x3 = func.max_pool2d(x3, kernel_size=(150, 150), padding=0)
        x4 = func.max_pool2d(x4, kernel_size=(150, 150), padding=0)
        x5 = func.max_pool2d(x5, kernel_size=(140, 140), padding=0)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = x.view((len(batch), 80 * 5))
        x = self.dropout(x)
        # out = self.drop(out)
        # logits = self.fc(out.squeeze(2))
        logits = self.lin(x)
        return logits


class Net5(nn.Module):  # 87% with k = [2, 4, 8, 16, 32, 64], k_amount = 64
    def __init__(self):
        super(Net5, self).__init__()
        # dropout=.3, channels=16, kernels=[2, 4, 8, 16, 32, 64], classes=2
        self.conv = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.mp = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(16 * 12 * 12, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, batch):
        x = self.conv(batch)
        x = func.relu(x)
        x = self.mp(x)
        x = x.view(-1, 16 * 12 * 12)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        dropout = .25
        channels1 = 2
        channels2 = 4
        channels3 = 6
        kernels1 = [4, 8, 16]
        kernels2 = [2, 3, 4]
        kernels3 = [3, 4, 5]
        classes = 2

        self.conv0 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=channels1, kernel_size=kernels1[0]), nn.ReLU(),
                                   nn.MaxPool1d(3),
                                   nn.Conv1d(in_channels=channels1, out_channels=channels2, kernel_size=kernels2[0]),
                                   nn.ReLU(), nn.MaxPool1d(4),
                                   nn.Conv1d(in_channels=channels2, out_channels=channels3, kernel_size=kernels3[0]),
                                   nn.ReLU(), nn.MaxPool1d(5))
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=channels1, kernel_size=kernels1[1]), nn.ReLU(),
                                   nn.MaxPool1d(3),
                                   nn.Conv1d(in_channels=channels1, out_channels=channels2, kernel_size=kernels2[1]),
                                   nn.ReLU(), nn.MaxPool1d(4),
                                   nn.Conv1d(in_channels=channels2, out_channels=channels3, kernel_size=kernels3[1]),
                                   nn.ReLU(), nn.MaxPool1d(5))
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=channels1, kernel_size=kernels1[2]), nn.ReLU(),
                                   nn.MaxPool1d(3),
                                   nn.Conv1d(in_channels=channels1, out_channels=channels2, kernel_size=kernels2[2]),
                                   nn.ReLU(), nn.MaxPool1d(4),
                                   nn.Conv1d(in_channels=channels2, out_channels=channels3, kernel_size=kernels3[2]),
                                   nn.ReLU(), nn.MaxPool1d(5))

        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Sequential(
            nn.Linear(204, 32),
            nn.Linear(32, classes),
        )

    def forward(self, batch):
        x = torch.reshape(batch, (-1, 1, 768))
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat([x0, x1, x2], dim=-1)
        x = x.view(len(batch), -1)
        x = self.dropout(x)
        logits = torch.softmax(self.lin(x), dim=-1)
        return logits