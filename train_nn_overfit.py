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

# This implementation does not use cuda or gpu acceleration

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

# define network class
class Network(nn.Module):
	def __init__(self, dropout_rate = 0.0):
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

		# define dropout layers
		self.dropout1 = nn.Dropout(p=dropout_rate)
		self.dropout2 = nn.Dropout(p=dropout_rate)
		self.dropout3 = nn.Dropout(p=dropout_rate)

		# define batch norm layers
		self.batchnorm1 = nn.BatchNorm1d(67)
		self.batchnorm2 = nn.BatchNorm1d(38)
		self.batchnorm3 = nn.BatchNorm1d(10)
		

	def forward(self, x):
		x = x.view(x.shape[0], -1)
		# Pass the input tensor through each of our operations
		
		x = self.hidden1(x)
		x = self.sigmoid1(x)
		x = self.dropout1(x)
#		x = self.batchnorm1(x)

		x = self.hidden2(x)
		x = self.sigmoid2(x)
		x = self.dropout2(x)
#		x = self.batchnorm2(x)

		x = self.hidden3(x)
		x = self.sigmoid3(x)
		x = self.dropout3(x)
#		x = self.batchnorm3(x)

		x = self.output(x)
		x = self.softmax(x)
		
		return x


# set hyperparameters
# you should record all your hyperparameters in readable (and usable) format in one block to make it easy to reference
# this includes arcitechture features like dropout rate, if batchnorm is present, etc.
	# set our loss funciton to negative log likelihood
	# there are many availalbe loss functions in pytorch
	# you can also define your own loss function, although autograd may not work with a custom function
loss_func = nn.NLLLoss()
loss_func_name = 'negative log likelihood'
learn_rate = 3.5e-4
num_epochs = 400
batch_size = 64
confidence_threshold = 0.5
loss_adj_conf_thresh = np.log(confidence_threshold)
optimizer_name = 'Adam'
start_time = datetime.now()
momentum = 0.0

# overfitting correction hyperparameters
early_stopping_thresh = -0.4   
early_stopping_num_epochs = 50
dropout_rate = 0.15
l2_lambda = 1e-5
# if you want to add learning rate scheduler, use *torch.optim.lr_scheduler* https://pytorch.org/docs/stable/optim.html


# you should  create a unique ID for this hyperparameter run, this should be a folder you save all relevent files to (hyperparams file, training logs, model weights, etc.)
run_id = "example hp run"
os.mkdir(run_id)

# record all hyperparameters that might be useful to reference later
# once you get to a couple dozen hyperparameter runs you will be very thankful for this
with open(run_id + '/hyperparams.csv', 'w') as wfil:
	wfil.write("loss function," + loss_func_name + '\n')
	wfil.write("learning rate," + str(learn_rate) + '\n')
	wfil.write("number epochs," + str(num_epochs) + '\n')
	wfil.write("batch size," + str(batch_size) + '\n')
	wfil.write("optimizer," + str(learn_rate) + '\n')
	wfil.write("momentum," + str(momentum) + '\n')
	wfil.write("early stopping threshold," + str(early_stopping_thresh) + '\n')
	wfil.write("early stopping number of epochs necessary," + str(early_stopping_num_epochs) + '\n')
	wfil.write("dropout rate," + str(dropout_rate) + '\n')
	wfil.write("regularization weight," + str(l2_lambda) + '\n')
	wfil.write("using batch norm," + "yes" + '\n')
	wfil.write("start time," + str(start_time) + '\n')


# weights are initialized randomly by default
# pytorch  has some built in initialization options which can be found
# additionally, if you want to write your own initialization scheme you can follow the examples in the second answer on this stackoverflow post https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch 

# initialize network
model = Network(dropout_rate = dropout_rate)

# create loaders to feed in data to the network in batches
train_set = mydata("./snf_train.csv", 'SNF')
trainloader = torch.utils.data.DataLoader( dataset = train_set , batch_size= batch_size , shuffle = True)
valid_set = mydata("./snf_valid.csv", 'SNF')
validloader = torch.utils.data.DataLoader( dataset = valid_set , batch_size= batch_size , shuffle = True)

# initialize optimizer
# adding in the weight decay parameter to the optimizer turns l2 regularization on. The value this parameter takes is equal to it's lambda - larger lambda means more regularization
# adding the momentum term adds momentum to the model with the value provided
optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=l2_lambda)  #, momentum = momentum, weight_decay=l2_lambda)


# track best val loss to know when to save best weights
best_valid_loss = "unset"

# track stuff for early stopping
early_stopping_counter = 0
es_valid_loss = 0.0

with open(run_id + '/log_file.csv', 'w') as log_fil:
	# write headers for log file
	log_fil.write("epoch,epoch duration,train loss,valid loss,train accuracy, valid accuracy\n")

	for epoch in range(0, num_epochs):
		epoch_start = datetime.now()

		# track train and validation loss
		epoch_train_loss = 0.0
		epoch_valid_loss = 0.0

		# track train and validation accuracy
		epoch_train_accuracy = 0.0
		epoch_valid_accuracy = 0.0
		epoch_train_counter = 0.0
		epoch_valid_counter = 0.0

		for i, (images, labels) in enumerate(trainloader):

			# zero out gradients for every batch or they will accumulate
			optimizer.zero_grad()

			# undoes model.eval()
			model.train(True)

			# forward step
			outputs = model(images)

			# compute loss
			loss = loss_func(outputs, labels)

			# backwards step
			loss.backward()

			# update weights and biases
			# rembmer if using scheduler, you step the scheduler not the optimizer - see *torch.optim.lr_scheduler* https://pytorch.org/docs/stable/optim.html
			optimizer.step()

			# track training loss
			epoch_train_loss += loss.item()

			# calculate training accuracy
			for i in range(0,len(labels)):
				temp_label = labels[i]
				temp_pred = outputs[i,1]
				if temp_pred > loss_adj_conf_thresh:
					temp_pred = 1.0
				else:
					temp_pred = 0.0
				if float(temp_pred) - float(temp_label) == 0:
					epoch_train_accuracy += 1.0
				epoch_train_counter += 1.0

		epoch_train_accuracy = epoch_train_accuracy / epoch_train_counter

		# track valid loss - the torch.no_grad() unsures gradients will not be updated based on validation set
		with torch.no_grad():
			for i, (images, labels) in enumerate(validloader):
				# Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.
				model.eval()

				outputs = model(images)
				loss = loss_func(outputs, labels)
				epoch_valid_loss += loss.item()

				# calculate validation accuracy
				for i in range(0,len(labels)):
					temp_label = labels[i]
					temp_pred = outputs[i,1]
					if temp_pred > loss_adj_conf_thresh:
						temp_pred = 1.0
					else:
						temp_pred = 0.0
					if float(temp_pred) - float(temp_label) == 0:
						epoch_valid_accuracy += 1.0
					epoch_valid_counter += 1.0

		epoch_valid_accuracy = epoch_valid_accuracy / epoch_valid_counter

		# track total epoch time
		epoch_end = datetime.now()
		epoch_time = (epoch_end - epoch_start).total_seconds()

		# save best weights
		if best_valid_loss=="unset" or epoch_valid_loss < best_valid_loss:
			best_valid_loss = epoch_valid_loss
			torch.save(model, run_id + "/best_weights.pth")

		# save most recent weights
		torch.save(model, run_id + "/last_weights.pth")

		# save epoch results in log file
		log_fil.write( str(epoch) + ',' + str(epoch_time) + ',' + str(epoch_train_loss) + ',' + str(epoch_valid_loss) + ',' + str(epoch_train_accuracy) + ',' + str(epoch_valid_accuracy) + '\n' )
		
		# print out epoch level training details
		print("epoch: " + str(epoch) + " - ("+ str(epoch_time) + " seconds)" + "\n\ttrain loss: " + str(epoch_train_loss) + " - train accuracy: " + str(epoch_train_accuracy) + "\n\tvalid loss: " + str(epoch_valid_loss) + " - valid accuracy: " + str(epoch_valid_accuracy))

		# implement early stopping
		if es_valid_loss == 0.0:
			early_stopping_counter = 0
			es_valid_loss = epoch_valid_loss

		if es_valid_loss - epoch_valid_loss < early_stopping_thresh:
			early_stopping_counter += 1
		else:
			early_stopping_counter = 0
			es_valid_loss = epoch_valid_loss

		if early_stopping_counter >= early_stopping_num_epochs:
			print("Stopped Early")
			break

end_time = datetime.now()
with open(run_id + '/hyperparams.csv', 'a') as wfil:
	wfil.write("end time," + str(end_time) + '\n')

# Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.