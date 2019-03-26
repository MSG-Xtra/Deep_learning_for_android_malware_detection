# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 10:16:05 2019
@author: MSG
===============================================================================
****this script is to determine importance of each feature
such that less significant features can be ignored******
===============================================================================
"""
import json
import pandas as pd
import numpy as np

#**************POPULATE GENERAL FEATURES LISTS*********************************
#general lists upon which feature selection will be made basing on feature importance 

Glist_of_receivers_actions = []
Glist_of_permissions = []
Glist_of_apis = []
Gruntime_registered_receivers = []
Glist_of_fingerprints = []
#populate lists
with open("M_Bfeatures_jsons.json", "r") as d:
    json_dataset = json.load(d) 
    for key, value in json_dataset.items():
       list_of_receivers_actions = value['list_of_receivers_actions']
       list_of_permissions = value['list_of_permissions']      
       apis =   value['apis']
       runtime_registered_receivers =  value['runtime_registered_receivers']
       list_of_fingerprints = value['list_of_fingerprints']
      
        
       #POPULATE GENERAL LISTS UPON WHICH FEATURE PRESENCE CHECK WILL BE DONE
	   #list_of_receivers_actions
       for i in range(len(list_of_receivers_actions)):
           r = list_of_receivers_actions[i]
           if  r not in Glist_of_receivers_actions:
               if r.startswith("android.intent.action."):
                   Glist_of_receivers_actions.append(r)       
      
       #list_of_permissions
       for i in range(len(list_of_permissions)):
           s = list_of_permissions[i]
           if s  not in Glist_of_permissions:
               if s.startswith('android.permission.'):
                   Glist_of_permissions.append(s) 
                   
       #list_of_api_names
       for key in  apis.keys():
           if key not in Glist_of_apis:
               Glist_of_apis.append(key) 
        
       #registered_receivers
       for i in range(len(runtime_registered_receivers)):
           rt =  runtime_registered_receivers[i]
           if rt not in Gruntime_registered_receivers:
               if rt.startswith("android.intent.action."):
                   Gruntime_registered_receivers.append(rt) 
                
       #list_of_fingerprints
       for i in range(len(list_of_fingerprints)):
           if  list_of_fingerprints[i] not in Glist_of_fingerprints:
               Glist_of_fingerprints.append(list_of_fingerprints[i])       


others = ['malware']

all_features = (Glist_of_permissions + Glist_of_receivers_actions + Gruntime_registered_receivers+ 
                        Glist_of_fingerprints + Glist_of_apis +  others)
 
 
#****GET FEATURE IMPORTANCES ************************************************
# Importing the dataset
dataset = pd.read_csv("MSGmalware_analysis_dataset_all_features.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset.iloc[:, 1:714].values
y = dataset.iloc[:, 714].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)
## Import the random forest model.
from sklearn.ensemble import RandomForestClassifier 
## This line instantiates the model. 
rf = RandomForestClassifier() 
## Fit the model on your training data.
rf.fit(X_train, y_train) 
## And score it on your testing data.
rf.score(X_test, y_test)

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
important_features = []
important_features_scores = []
for f in range(X.shape[1]):
    print("%d. %-*s %f" % (f + 1, 0, all_features[indices[f]], importances[indices[f]]))
    #store important features
    if importances[indices[f]] > 0.000004:
        important_features.append(all_features[indices[f]])
        important_features_scores.append(all_features[indices[f]]+' >>> '+str(round(importances[indices[f]],6)))

#new general lists of important features(if) 
if_list_of_receivers_actions = []
if_list_of_permissions = []
if_list_of_apis = []
if_runtime_registered_receivers = []
if_list_of_fingerprints = []   
if_others = []    

#update if_list_of_receivers_actions
for i in range (len(Glist_of_receivers_actions)):
    if Glist_of_receivers_actions[i] in important_features:
        if_list_of_receivers_actions.append(Glist_of_receivers_actions[i])
#update if_others
for i in range(len(others)):
    if others[i] in important_features:
        if_others.append(others[i])      
#update if_list_of_permissions
for i in range(len(Glist_of_permissions)):
    if   Glist_of_permissions[i] in  important_features:
        if_list_of_permissions.append(Glist_of_permissions[i])
#update if_list_of_apis 
for i in range(len(Glist_of_apis)):
    if   Glist_of_apis[i] in  important_features:
        if_list_of_apis.append(Glist_of_apis[i])        
#update if_runtime_registered_receivers
for i in range(len(Gruntime_registered_receivers)):
    if   Gruntime_registered_receivers[i] in  important_features:
        if_runtime_registered_receivers.append(Gruntime_registered_receivers[i]) 
#update if_list_of_fingerprints
for i in range(len(Glist_of_fingerprints)):
    if  Glist_of_fingerprints[i] in  important_features:
        if_list_of_fingerprints.append(Glist_of_fingerprints[i]) 
"""******************************END****************************************""" 