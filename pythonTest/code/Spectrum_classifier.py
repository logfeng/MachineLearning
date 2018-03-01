import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Hyper Parameters
EPOCH = 1 #20  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001  # learning rate
BAIS_INIT = 0

# n_target = 5
n_fold = 20
X = np.loadtxt("train_data.csv", delimiter = ',', skiprows = 1)  # data
Y = np.loadtxt("train_label.csv", delimiter = ',', skiprows = 1) # label
# X = pd.read_csv("train_data.csv").astype(np.float32)  # data
# Y = pd.read_csv("train_label.csv").astype(np.float32) # label

# wavenumber = np.load("x_name.npy")


# define your network
class CNN_classifier(nn.Module):
    def __init__(self, bn=False):
        super(CNN_classifier, self).__init__()
        self.do_bn = bn
        # self.bn_input = nn.BatchNorm1d(momentum=0.5, num_features=1716)
        self.conv1 = nn.Sequential(                                                         # input shape (1, 1, 1716) floor((Lin - kernel_size)/stride + 1)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=21, stride=2, padding=0), # (16, 1, 848)
            nn.BatchNorm1d(momentum=0.4, num_features=16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),                                          # output shape (16, 1, 424)
        )
        self.conv2 = nn.Sequential(                                                         # intput shape (16, 1, 424)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=11, stride=2, padding=0),# (32, 1, 207)
            nn.BatchNorm1d(momentum=0.4, num_features=32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),                                          # output shape (32, 1, 103)
        )
        self.conv3 = nn.Sequential(                                                         # input shape (32, 1, 103)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0), # (64, 1, 50)
            nn.BatchNorm1d(momentum=0.4, num_features=64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),                                          # output shape (64, 1, 25)
        )
        self.dense = nn.Sequential(
            nn.Linear(64 * 25, 2048),  # Dense(2048)
            nn.BatchNorm1d(momentum=0.4, num_features=2048),
            nn.LeakyReLU(negative_slope=0.5),
            nn.Dropout(0.5),
        )
        self.out = nn.Sequential(
            nn.Linear(2048, 1541),     # fully connected layer, output probability of miner classes [.1, .2, .7] unnormalized
            nn.BatchNorm1d(momentum=0.4, num_features=1541),
            nn.Softmax(1),
        )

    def _set_init(self, layer):
        inti.normal(layer.weight, mean=0., std=.1)
        init.constant(layer.bias, BIAS_INIT)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        output = self.out(x)
        return output, x


train_loss = np.zeros(n_fold)
valid_loss = np.zeros(n_fold)

# arrange data
for f in range(n_fold):
    print("Fold %d --------------------------------------------" % f)
    train_data, test_data, train_label, test_label = train_test_split(X, Y, test_size=0.05)
    train_data = Data.TensorDataset(torch.unsqueeze(torch.FloatTensor(train_data), dim=1),
                                    torch.LongTensor(train_label))
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_y = torch.LongTensor(test_label)
    test_x = Variable(torch.unsqueeze(torch.FloatTensor(test_data), dim=1))
    # net work instance
    cnn = CNN_classifier()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss(weight= None)          # the target label is not one-hotted, weight is proportional to the number of samples in class C

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            b_x = Variable(x)                          # batch x
            b_y = Variable(y).type(torch.LongTensor)   # batch y

            output = cnn(b_x)[0]           # cnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()          # clear gradients for this training step
            loss.backward()                # backpropagation, compute gradients
            optimizer.step()               # apply gradients

            if step % 15 == 0:
                test_output, last_layer = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                test_y_tmp = Variable(test_y)
                cel = loss_func(test_output, test_y_tmp)
                # print (type(pred_y))
                # print (type(test_y))
                accuracy = sum(pred_y == test_y) / float(test_y.size(0))
                print('Epoch: ', epoch,
                      '| Train loss: %.4f' % loss.data[0],
                      '| Test loss: %.4f' % cel.data.numpy()[0],
                      '| Accuracy loss: %.2f' % accuracy)

    print("SAVING LOSS.....................................")
    train_loss[f] = loss.data.numpy()[0]
    valid_loss[f] = cel.data.numpy()[0]
    np.save("train_loss.npy", train_loss)
    np.save("valid_loss.npy", valid_loss)

    print("SAVING MODELS...................................")
    torch.save(cnn, 'fold_%d_train_%.4f_valid_%.4f.pkl' % (f, train_loss[f], valid_loss[f]))
