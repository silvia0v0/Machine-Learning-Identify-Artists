import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



'''CNN'''


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(nn.Linear(64 * 5 * 5, 256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 4))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


'''Load Data'''
def unpickle(filename):
    with open(filename, 'rb') as f:
        dic = pickle.load(f)
        return dic

train_dict = unpickle('train_batch_05072005.pkl')
test_dict = unpickle('test_batch_05072005.pkl')
x_train_all = train_dict['data'].reshape(8242, 3, 64, 64)
y_train_all = train_dict['labels']
num_paints = train_dict['paint_number']
print(x_train_all.shape)
print(y_train_all.shape)
vali_rate = 0.2
train_count = 6000

x_train = x_train_all[:train_count]
y_train = y_train_all[:train_count]
x_val = x_train_all[train_count:train_count+2000]
y_val = y_train_all[train_count:train_count+2000]

x_test = np.array(test_dict['data'])
y_test = np.array(test_dict['labels'])
print(x_test.shape)
print(y_test.shape)


'''normalization'''
x_train_normalized = x_train/255
x_val_normalized = x_val/255
x_test_normalized = x_test/255


'''mini-batch preparation'''
# init network
conv_net = ConvNet()
print('model structure: ', conv_net)
# init optimizer
optimizer = optim.Adam(conv_net.parameters(),lr=1e-3)
# set loss function
criterion = nn.CrossEntropyLoss()
# prepare for mini-batch stochastic gradient descent
n_iteration = 40
batch_size = 256
n_data = x_train_normalized.shape[0]
n_batch = int(np.ceil(n_data/batch_size))

# convert X_train and X_val to tensor and flatten them
# X_train_tensor = Tensor(X_train_normalized).reshape(n_train_data,-1)
# X_val_tensor = Tensor(X_val_normalized).reshape(1000,-1)
X_train_tensor = torch.Tensor(x_train_normalized)
X_val_tensor = torch.Tensor(x_val_normalized)

# convert training label to tensor and to type long
y_train_tensor = torch.Tensor(y_train).long()
y_val_tensor = torch.Tensor(y_val).long()

print('X train tensor shape:', X_train_tensor.shape)


'''training'''

def get_correct_and_accuracy(y_pred, y):
    # y_pred is the nxC prediction scores
    # give the number of correct and the accuracy
    n = y.shape[0]
    # find the prediction class label
    _ ,pred_class = y_pred.max(dim=1)
    correct = (pred_class == y).sum().item()
    return correct ,correct/n

## start
train_loss_list = np.zeros(n_iteration)
train_accu_list = np.zeros(n_iteration)
val_loss_list = np.zeros(n_iteration)
val_accu_list = np.zeros(n_iteration)

for i in range(n_iteration):
    # first get a minibatch of data

    total_train_loss = 0
    total_train_accuracy = 0

    for j in range(n_batch):
        batch_start_index = j * batch_size
        # get data batch from the normalized data
        X_batch = X_train_tensor[batch_start_index:batch_start_index + batch_size]
        # get ground truth label y
        y_batch = y_train_tensor[batch_start_index:batch_start_index + batch_size]

        y_pred = conv_net.forward(X_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()  # 即将梯度初始化为零
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        accu = get_correct_and_accuracy(y_pred, y_batch)[1]
        total_train_accuracy += accu

    ave_train_loss = total_train_loss / n_batch
    train_accu = total_train_accuracy / n_batch

    y_val_pred = conv_net.forward(X_val_tensor)
    val_loss = criterion(y_val_pred, y_val_tensor)  # why y_val_tensor not y_val?
    val_accu = get_correct_and_accuracy(y_val_pred, y_val_tensor)[1]
    print("Iter %d ,Train loss: %.3f, Train acc: %.3f, Val loss: %.3f, Val acc: %.3f"
          % (i, ave_train_loss, train_accu, val_loss, val_accu))
    ## add to the logs so that we can use them later for plotting
    train_loss_list[i] = ave_train_loss
    train_accu_list[i] = train_accu
    val_loss_list[i] = val_loss
    val_accu_list[i] = val_accu

print(train_loss_list)
print(val_loss_list)

x_axis = np.arange(n_iteration)
plt.plot(x_axis, train_loss_list, label='train loss')
plt.plot(x_axis, val_loss_list, label='val loss')
plt.legend()
plt.show()

## plot training accuracy versus validation accuracy
plt.plot(x_axis, train_accu_list, label='train acc')
plt.plot(x_axis, val_accu_list, label='val acc')
plt.legend()
plt.show()




'''testing'''

# Test Method 1
pred_ = torch.zeros(473).long()

for i in range(473): # for each artwork
    
    l = len(x_test_normalized[i])
    X_test_tensor = torch.Tensor(x_test_normalized[i].reshape(l, 3, 64, 64))
    y_test_pred = conv_net.forward(X_test_tensor)
    
    _ ,pred_class = y_test_pred.max(dim=1)
    
    
    y_test_artwork_pred = max([(pred_class == i).sum().item() for i in range(4)])
    pred_[i] = y_test_artwork_pred


# convert testing label to tensor and to type long
y_test_tensor = torch.Tensor(y_test).long()

correct = (pred_ == y_test_tensor).sum().item()
n = y_test_tensor.shape[0]





test_accu = correct/n
 
print("Test Accuracy:", test_accu)



