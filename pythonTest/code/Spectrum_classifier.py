import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# Hyper Parameters
EPOCH = 20  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001  # learning rate
BAIS_INIT = 0

# n_target = 5
n_fold = 20
X = np.load("train_data.npy").astype(np.float32)  # data
Y = np.load("train_label.npy").astype(np.float32) # label

wavenumber = np.load("x_name.npy")


# define your network
class CNN_classifier(nn.Module):
    def __init__(self, bn=False):
        super(CNN_classifier, self).__init__()
        self.do_bn = bn
        #         self.bn_input = nn.BatchNorm1d(momentum=0.5, num_features=3578)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=21, stride=2, padding=0),
            nn.BatchNorm1d(momentum=0.4, num_features=16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=11, stride=2, padding=0),
            nn.BatchNorm1d(momentum=0.4, num_features=32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm1d(momentum=0.4, num_features=64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.dense = nn.Sequential(
            nn.Linear(64 * 54, 2048),  # Dense(2048)
            nn.BatchNorm1d(momentum=0.4, num_features=2048),
            nn.LeakyReLU(negative_slope=0.5),
            nn.Dropout(0.5),
        )
        self.out = nn.Sequential(
            nn.Linear(2048, 1671),        # fully connected layer, output probability of miner classes [.1, .2, .7] unnormalized
            nn.BatchNorm1d(momentum=0.4, num_features=1671),
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
    print("Target %d Fold %d --------------------------------------------" % f)
    train_data, test_data, train_label, test_label = train_test_split(X, Y, test_size=0.05)
    train_data = Data.TensorDataset(torch.unsqueeze(torch.FloatTensor(train_data), dim=1),
                                    torch.FloatTensor(train_label))
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_y = Variable(torch.FloatTensor(test_label))
    test_x = Variable(torch.unsqueeze(torch.FloatTensor(test_data), dim=1))
    # net work instance
    cnn = CNN_classifier()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            b_x = Variable(x)  # batch x
            b_y = Variable(y).type(torch.FloatTensor)  # batch y

            output = cnn(b_x)[0]  # cnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step == 15:
                test_output, _ = cnn(test_x)
                prediction = torch.max(test_output, 1)[1]
                pred_y = prediction.data.numpy().squeeze()
                target_y = y.data.numpy()
                cel = loss_func(test_output, test_y)
                accuracy = sum(pred_y == target_y)/200.  # denominator has to change
                print('Epoch: ', epoch,
                      '| Train loss: %.4f' % loss.data[0],
                      '| Test loss: %.4f' % cel.data.numpy()[0],
                      '| Accuracy loss: %.4f' % accuracy)

    print("SAVING LOSS.....................................")
    train_loss[t * n_fold + f] = loss.data.numpy()[0]
    valid_loss[t * n_fold + f] = mse.data.numpy()[0]
    np.save("train_loss.npy", train_loss)
    np.save("valid_loss.npy", valid_loss)

    print("SAVING MODELS...................................")
    torch.save(cnn, 'target_%d_fold_%d_train_%.4f_valid_%.4f.pkl' % (t, f, train_loss[t * n_fold + f], valid_loss[t * n_fold + f]))
