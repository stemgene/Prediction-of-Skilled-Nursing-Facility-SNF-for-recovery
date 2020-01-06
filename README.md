# Prediction-of-Skilled-Nursing-Facility-SNF-for-recovery

After having a total joint surgery (either a total hip or total knee replacement), some patients must be discharged to a Skilled Nursing Facility (SNF) for recovery. These SNFs are very expensive and can be quite disruptive to patient’s lives. The goal of this project is to develop a model that can predict, before the surgery, where the patient will be discharged to. This is a binary prediction task: will the patient go to a snf or not.

The data file contains real medical data that has been deidentified. The outcome is in the column INDEX_DISCH_DISP_NM. Due to the privacy and security of data, I didn't upload the dataset. 

* Part 1 – Data Preprocessing

First preprocess the data, as part of the preprocessing, I construct three new variables based on the column ProcName1. The first variable indicates which side the surgery was performed on (left, right, both, or unknown). The second indicates if the surgery was on the knee or the hip. The third one indicates if the anterior approach was used. Assume based on domain knowledge that these three factors will all be important. 

* Part 2 – Modeling

To make the prediction of SNF outcome, I first implementated algorithm of logistic regression, random forest, and SVM, then wrote a neural network. 

* Part 3 – Hyperparameter Tuning

Using the validation set, tune the hyperparameters for random forest, SVM models, and neural network. This is a chance to play around not only with the “traditional hyperparameters” (such as learning rate, batch size, momentum), but also network architecture related things like network depth, number of nodes in each hidden layer, different types of layers, different activation functions, different loss functions, different optimizers. Obviously it is not possible to try out all possible network architectures, but I've tried at least a few different ones to get the feel for how results change as architecture changes. 
