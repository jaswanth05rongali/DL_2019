from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import dataloader
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from datetime import datetime

#Set Contexts
ctx = mx.cpu() if mx.test_utils.list_gpus() else mx.cpu()
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

#Defining the network using nn.Sequential
num_outputs = 10
network = gluon.nn.Sequential()
with network.name_scope():
	network.add(gluon.nn.Dense(1024, activation="relu"))
	network.add(gluon.nn.Dense(512, activation="relu"))
	network.add(gluon.nn.Dense(256, activation="relu"))
	network.add(gluon.nn.Dense(num_outputs))

#Parameter Initialization for the network
network.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)

#loss for both the network and logistic regression model
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

#optimizer for the network
lr = 0.001
param_update = gluon.Trainer(network.collect_params(), 'adam', {'learning_rate': lr})

#Training the network
def train_network():
	epochs = 10
	for i in range(epochs):
		for e,(data,label) in enumerate(train_data):
			data,label = transform(data,label)
			data = data.as_in_context(model_ctx)
			label = label.as_in_context(model_ctx)
			with autograd.record():
				output = network(data)
				loss = softmax_cross_entropy(output,label)
			loss.backward()
			param_update.step(data.shape[0])	

#Defining the logistic regression model
log_model = gluon.nn.Dense(num_outputs)

#Initializing the parameters of the logistic regression model
log_model.collect_params().initialize(mx.init.Normal(sigma=0.5), ctx=model_ctx)

#Optimizer for the logistic model
log_lr = 0.01
log_trainer = gluon.Trainer(log_model.collect_params(), 'adam' ,{'learning_rate':log_lr})

#Training the fully connected network
train_network()

no_layers = 2 #Number of hidden layers to be considered whose activations are taken as input in logistic reg model

#getting the hidden layer ouputs 
def get_the_hiddenlayer_outputs(no_layers,t_data):
	if no_layers>=3:
		print("Only three hidden layers in the network")
		exit()

	net = gluon.nn.Sequential()
	for x in range(no_layers):
		list = network[:x]

	for item in list:
		net.add(item)

	output = net(t_data)
	return output

#Accuracy Evaluation of the log_reg model
def evaluate_accuracy(data_iter, network):
    acc = mx.metric.Accuracy()
    for i,(data, label) in enumerate(data_iter):
    	data,label = transform(data,label)
    	data = data.as_in_context(model_ctx)
    	label = label.as_in_context(model_ctx)
    	act_output = get_the_hiddenlayer_outputs(no_layers,data)
    	output = network(act_output)
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
		act_output = get_the_hiddenlayer_outputs(no_layers,data)
		output = network(act_output)
		loss = softmax_cross_entropy(output,label)
		cum_loss += nd.sum(loss).asscalar()
		predictions = nd.argmax(output, axis=1)
		acc.update(preds=predictions, labels=label)
	return acc.get()[1],cum_loss

#Training the logistic regression model
epochs = 15
cumulative_loss = 0
for i in range(epochs):
	for e,(data,label) in enumerate(train_data):
		data,label = transform(data,label)
		data = data.as_in_context(model_ctx)
		label = label.as_in_context(model_ctx)
		act_out = get_the_hiddenlayer_outputs(no_layers,data)
		with autograd.record():
			output = log_model(act_out)
			loss = softmax_cross_entropy(output,label)
		loss.backward()
		log_trainer.step(data.shape[0])
		cumulative_loss += nd.sum(loss).asscalar()


val_accuracy,val_loss = accuracy(val_data,log_model)
train_accuracy = evaluate_accuracy(train_data,log_model)
test_accuracy,test_loss = accuracy(test_data,log_model)
print("Epochs: %s. Loss: %s, Train_acc: %s, Validation_acc: %s, test_accuracy: %s" %
      (i+1, cumulative_loss/num_instances, train_accuracy, val_accuracy,test_accuracy))

log_model.save_parameters("NN3.params")