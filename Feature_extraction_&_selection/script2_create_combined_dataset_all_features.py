
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 12:49:34 2018
@author: MSG
===============================================================================
***this script creates a dataset with all features***
===============================================================================
"""
import json
import pandas as pd
import time
#LOAD DATA
start_time = time.time() 
#load .json datasets for malware and benign
#concatenate them
with open("Mfeatures_jsons.json", "r") as mal:
    data_m = json.load(mal)
with open("Bfeatures_jsons.json", "r") as beg:
    data_b = json.load(beg)
    
data_m.update(data_b)

with open("M_Bfeatures_jsons.json", "w") as fo:
    json.dump(data_m, fo, indent = 2)

#POPULATE FEATURES LISTS
#general lists upon which feature presence check will be made 
#these are the features under investigation for each app

Glist_of_receivers_actions = []
Glist_of_permissions = []
Glist_of_apis = []
Gruntime_registered_receivers = []
Glist_of_fingerprints = []

#populate lists
with open("M_Bfeatures_jsons.json", "r") as d:
    json_dataset = json.load(d)
    
     
    for key, value in json_dataset.items():
        
        # here I push a list of data into a pandas DataFrame 
        # each key(sample) forms a row
        #this function will be transfered to the final script that
        #reads the json file created here
              
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


#CREATE PANDAS DATAFRAME
#here I define my pandas Dataframe with the columns I want to get from the json
#dataset for all samples(malware & benign)

others = ['malware']
    
#create dataset(1527 columns of features)

data = pd.DataFrame(columns = Glist_of_permissions
                       + Glist_of_receivers_actions + Gruntime_registered_receivers+ 
                        Glist_of_fingerprints + Glist_of_apis +  others)

#initialize  dataset with 0 values for all features
with open("M_Bfeatures_jsons.json", "r") as d:
    json_dataset = json.load(d)
  
    #add rows with only index values(md5)
    for key  in json_dataset.keys(): 
        data.loc[key] = 0 
       
#CHECK FOR Features and update corresponding  dataset values  
with open("M_Bfeatures_jsons.json", "r") as d:
    json_dataset = json.load(d)
    
    for key, value in json_dataset.items():
        
        # here I push append  data to a pandas DataFrame 
        # each key(sample) forms a row
                     
       list_of_permissions = value['list_of_permissions']
       list_of_receivers = value['list_of_receivers']
       list_of_receivers_actions = value['list_of_receivers_actions']
       runtime_registered_receivers =  value['runtime_registered_receivers']
       list_of_fingerprints = value['list_of_fingerprints']
       apis =   value['apis']
       malware = value['malware']
            
       #update presence of given permission for sample(key)
       for i in range(len(list_of_permissions)):
           m =  list_of_permissions[i]        
           if m in Glist_of_permissions:
               data.loc[key, m] = 1   
               
       #update presence of given receivers_action for sample(key)
       for i in range(len(list_of_receivers_actions)):
           m =  list_of_receivers_actions[i]        
           if m in Glist_of_receivers_actions:
               data.loc[key, m] = 1
               
       #update presence of given registered_receiver for sample(key)
       for i in range(len(runtime_registered_receivers)):
           m =  runtime_registered_receivers[i]        
           if m in Gruntime_registered_receivers:
               data.loc[key, m] = 1
        
       #update presence of given fingerprint for sample(key)
       for i in range(len(list_of_fingerprints)):
           m = list_of_fingerprints[i]        
           if m in Glist_of_fingerprints:
               data.loc[key, m] = 1 
               
       #update presence of a given api call for sample(key) 
       for api in apis.keys():
           m = api
           if m in Glist_of_apis:
                data.loc[key, m] = 1 

       #update  others features
       data.loc[key, 'malware'] = malware

#save dataset in  csv format
data.to_csv("MSGmalware_analysis_dataset_all_features.csv",  encoding='utf8')
end_time = time.time()
print('Execution time: '+str(round( end_time -start_time, 3))+'seconds')
"""******************************END****************************************"""
    

  
