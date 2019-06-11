"""In this version(SmartAM2), we train and test on a combination of 
origonal samples and adversarial samples,  for validation, we use unseen batch
of adversarial samples"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import roc_curve
import time

start_time = time.time() 
#****PART1---DATA PREPROCESSING************************************************
# Importing the datasets
dataset_org = pd.read_csv("MSGmalware_analysis_dataset_if.csv", delimiter=",")
dataset_adver = pd.read_csv('adver_dataset_if_train_1.csv', delimiter=",")
#concatenate original and adversarial samples
dataset_retrain = pd.concat([dataset_org, dataset_adver], axis=0)
# split into input (X) and output (Y) variables
X = dataset_retrain.iloc[:, 1:358].values
y = dataset_retrain.iloc[:, 358].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 0)

#***PART2----BUILDING THE ANN(SmartAM2)****************************************
# Initialising the ANN
model = Sequential()
# Adding the input layer and the first hidden layer 
model.add(Dense(units = 300, kernel_initializer = 'uniform', 
                activation = 'relu', input_dim = 357))
# Adding the second hidden layer
model.add(Dense(units = 200, kernel_initializer = 'uniform', 
                activation = 'relu'))
# Adding the third hidden layer
model.add(Dense(units = 80, kernel_initializer = 'uniform', 
                activation = 'relu'))
# Adding the output layer
model.add(Dense(units = 1, kernel_initializer = 'uniform', 
                activation = 'sigmoid'))
# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
              metrics = ['accuracy'])
# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 100, epochs = 800)


#****PART3----MAKING PREDICTIONS & EVALUATING THE MODEL************************
# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_org = confusion_matrix(y_test, y_pred)
# evaluate the model
scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#ROC
y_org_pred = model.predict(X_test).ravel()
fpr_org, tpr_org, thresholds_org = roc_curve(y_test, y_org_pred)

#AUC value can also be calculated like this.
from sklearn.metrics import auc
auc_org = auc(fpr_org, tpr_org)


#*****PART4----VALIDATING WITH UNSEEN ADVERSARIAL SAMPLES**********************
dataset_adver = pd.read_csv('adver_dataset_if_test_1.csv')
x_adver = dataset_adver.iloc[:, 1:-1].values
y_adver = dataset_adver.iloc[:, 358].values
# Predicting the Test set results for adver
y_adver_pred = model.predict(x_adver)
y_adver_pred = (y_adver_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_adver = confusion_matrix(y_adver, y_adver_pred)


#Now, let’s plot the ROC for the MODEL;
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_org, tpr_org, label='Original+Adversarial samples (area = {:.3f})'.format(auc_org))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()



end_time=time.time()
print('Execution time: '+str(round( end_time -start_time, 3))+'seconds')

#******SAVING THE MODEL********************************************************
#save json
model_json = model.to_json()
with open("SmartAM2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("SmartAM2.h5")
print("Saved model to disk")


#visualize the ann
from ann_visualizer.visualize import ann_viz
ann_viz(model, view=True, filename="SmartAM2.gv", 
        title="SmartAM2")

