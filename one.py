from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import dataloader
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from datetime import datetime

#Set Contexts
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx

#load the data
num_inputs = 784
batch_size = 64
num_instances = 60000
data = dataloader.DataLoader()
train_data,train_labels = data.load_data()
test_data,test_labels = data.load_data(mode = 'test')
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.30, random_state=42)

train_data = []
for index,data in enumerate(X_train):
    temp = y_train[index]
    train_data.append((data,temp))
 
num_instances = len(train_data)

val_data = []
for index,data in enumerate(X_val):
    temp = y_val[index]
    val_data.append((data,temp))
    
test_data = []
for index,data in enumerate(test_data):
    temp = test_labels[index]
    test_data.append((data,temp))

train_data = gluon.data.DataLoader(train_data,batch_size, shuffle=True)
test_data = gluon.data.DataLoader(test_data,batch_size, shuffle=False)
val_data = gluon.data.DataLoader(val_data,batch_size,shuffle=False)

#normalizing function
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

#Defining the model using nn.Sequential
num_outputs = 10
network = gluon.nn.Sequential()
with network.name_scope():
    network.add(gluon.nn.Dense(1024, activation="relu"))
    network.add(gluon.nn.Dense(512, activation="relu"))
    network.add(gluon.nn.Dense(256, activation="relu"))
    network.add(gluon.nn.Dense(num_outputs))

#Parameter Initialization
network.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)

#loss definition
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

#optimizer
lr = 0.001
param_update = gluon.Trainer(network.collect_params(), 'SGD', {'learning_rate': lr})
# param_update = gluon.Trainer(network.collect_params(), 'adam', {'learning_rate': lr})
# param_update = gluon.Trainer(network.collect_params(), 'adagrad', {'learning_rate': lr})
# param_update = gluon.Trainer(network.collect_params(), 'rmsprop',{'learning_rate': lr, 'gamma1': 0.9})
# param_update = gluon.Trainer(network.collect_params(), 'adadelta', {'rho': 0.99})

#Training Accuracy Evaluation
def train_accuracy(data_iter, network):
    acc = mx.metric.Accuracy()
    for i,(data, label) in enumerate(data_iter):
        data,label = transform(data,label)
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        output = network(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

#Validation and test accuracy evaluation
def accuracy(data_iter,network):
    acc = mx.metric.Accuracy()
    cum_loss = 0.0
    for i,(data,label) in enumerate(data_iter):
        data,label = transform(data,label)
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        output = network(data)
        loss = softmax_cross_entropy(output,label)
        cum_loss += nd.sum(loss).asscalar()
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1],cum_loss

network_1_train_loss = []
network_1_valid_loss = []

#Training the Model
epochs = 5

now = datetime.now()
for i in range(epochs):
    cumulative_loss = 0
    for e,(data,label) in enumerate(train_data):
        data,label = transform(data,label)
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = network(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        param_update.step(data.shape[0])
        cumulative_loss += nd.sum(loss).asscalar()
    val_accuracy,val_loss = accuracy(val_data,network)
    network_1_train_loss.append(cumulative_loss/num_instances)
    network_1_valid_loss.append(val_loss/(60000-num_instances))

later = datetime.now()
avg_epochs_dur = (later-now).total_seconds()
avg_epochs_dur /= epochs
val_accuracy,val_loss = accuracy(val_data,network)
train_accuracy = train_accuracy(train_data,network)

print("Epochs: %s. Loss: %s, Train_acc: %s,Validation_acc: %s,Average Epoch duration(in secs): %s" %
      (i+1, cumulative_loss/num_instances, train_accuracy,val_accuracy,avg_epochs_dur))

plt.figure("Image2")
plt.title("Training loss vs Epoch")
network_1_valid_loss =  [float(i)/sum(network_1_valid_loss) for i in network_1_valid_loss]
network_1_train_loss =  [float(i)/sum(network_1_train_loss) for i in network_1_train_loss]

plt.plot(network_1_train_loss, c="blue", label = "SGD")
plt.plot(network_1_train_loss, c="blue", label = "ADAM")
plt.plot(network_1_train_loss, c="blue", label = "rmsprop")
plt.plot(network_1_train_loss, c="blue", label = "ADAGRAD")
plt.plot(network_1_train_loss, c="blue", label = "ADADELTA")
plt.legend()
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.savefig('exp_'+str("Image2")+'.png')

