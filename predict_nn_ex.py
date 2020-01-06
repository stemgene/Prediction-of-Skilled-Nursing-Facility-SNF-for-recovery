
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import os
import numpy as np
import pandas as pd
from datetime import datetime
import sys

# define network class
class Network(nn.Module):
	def __init__(self):
		super().__init__()
		# Inputs to hidden layer linear transformation
		self.hidden1 = nn.Linear(96, 67)
		self.hidden2 = nn.Linear(67, 38)
		self.hidden3 = nn.Linear(38, 10)

		# Output layer, 2 nodes = one for each class
		# if we were using a sigmoid activation for our output layer, we would have only one node. softmax requires one per class which is why here we use two.
		self.output = nn.Linear(10, 2)

		# Define sigmoid activation and softmax output 
		self.sigmoid1 = nn.Sigmoid()
		self.sigmoid2 = nn.Sigmoid()
		self.sigmoid3 = nn.Sigmoid()
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, x):
		x = x.view(x.shape[0], -1)

		# Pass the input tensor through each of our operations
		x = self.hidden1(x)
		x = self.sigmoid1(x)

		x = self.hidden2(x)
		x = self.sigmoid2(x)

		x = self.hidden3(x)
		x = self.sigmoid3(x)

		x = self.output(x)
		x = self.softmax(x)
		return x

# define dataset class for data loader
class mydata(data.Dataset):
	# define how to create an instance, 
	# requires the path to a csv data file (data_file_path)
	# requires the name of the label/output column, as given in the csv
	def __init__(self, data_file_path, label_name):
		df = pd.read_csv(data_file_path)
		labels = df[label_name].values
		data = df.drop(label_name, axis=1)
		self.datalist = data
		self.labels = labels

	def __getitem__(self, index):
		return torch.Tensor(np.asarray(self.datalist.iloc[[index]].astype(float))), self.labels[index]

	def __len__(self):
		return self.datalist.shape[0]

# identify where the weights you want to load are 
weight_fil = "example hp run/best_weights.pth"

# identify where the data you want to test on is using a command line argument
# can also hard code this
data_fil = sys.argv[1]

# set necessary hyperparameters
batch_size = 64
loss_func = nn.NLLLoss()
confidence_threshold = 0.5
loss_adj_conf_thresh = np.log(confidence_threshold)

# # initialize model
# model = Network()

# # load weights
# model.load_state_dict(torch.load(weight_fil))

model = torch.load(weight_fil)

# put model in evaluation mode (sets dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results)
model.eval()

# create loaders to feed in data to the network in batches
eval_set = mydata(data_fil, 'SNF')
eval_loader = torch.utils.data.DataLoader( dataset = eval_set , batch_size= batch_size , shuffle = True)

# track metrics over dataset
eval_loss = 0.0
eval_accuracy = 0.0
eval_counter = 0.0

# loop through eval data
for i, (images, labels) in enumerate(eval_loader):
	
	# run the model on the eval batch
	outputs = model(images)
	
	# compute eval loss
	loss = loss_func(outputs, labels)
	eval_loss += loss.item()

	# calculate eval accuracy
	for i in range(0,len(labels)):
		temp_label = labels[i]
		temp_pred = outputs[i,1]
		if temp_pred > loss_adj_conf_thresh:
			temp_pred = 1.0
		else:
			temp_pred = 0.0
		if float(temp_pred) - float(temp_label) == 0:
			eval_accuracy += 1.0
		eval_counter += 1.0

eval_accuracy = eval_accuracy / eval_counter


print("Accuracy = " + str(eval_accuracy))
print("Loss = " + str(eval_loss))
