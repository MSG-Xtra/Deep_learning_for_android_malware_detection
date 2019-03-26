# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 18:43:35 2019

@author: MSG
===============================================================================
***this script creates final dataset that considers most important features
===============================================================================
"""
import json
import pandas as pd
import time

start_time = time.time() 
#LOAD DATA
#import important feature lists 
from significant_features import (if_list_of_permissions, if_list_of_receivers_actions , 
if_runtime_registered_receivers ,if_list_of_fingerprints , if_list_of_apis)


#CREATE PANDAS DATAFRAME
# pandas Dataframe with the columns  from the json
#dataset for all samples(malware & benign)
other = ['malware']
data = pd.DataFrame(columns = if_list_of_permissions
                       + if_list_of_receivers_actions + if_runtime_registered_receivers+ 
                        if_list_of_fingerprints + if_list_of_apis +  other)

#initialize  dataset with 0 values for all features
with open("M_Bfeatures_jsons.json", "r") as d:
    json_dataset = json.load(d)
    #add rows with only index values(md5)
    for key  in json_dataset.keys(): 
        data.loc[key] = 0 
       
#check for features and update corresponding  dataset values  
with open("M_Bfeatures_jsons.json", "r") as d:
    json_dataset = json.load(d)    
    for key, value in json_dataset.items():        
        # append  data to a pandas DataFrame 
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
           if m in if_list_of_permissions:
               data.loc[key, m] = 1   
               
       #update presence of given receivers_action for sample(key)
       for i in range(len(list_of_receivers_actions)):
           m =  list_of_receivers_actions[i]        
           if m in if_list_of_receivers_actions:
               data.loc[key, m] = 1      
               
       #update presence of given registered_receiver for sample(key)
       for i in range(len(runtime_registered_receivers)):
           m =  runtime_registered_receivers[i]        
           if m in if_runtime_registered_receivers:
               data.loc[key, m] = 1    
               
       #update presence of given fingerprint for sample(key)
       for i in range(len(list_of_fingerprints)):
           m = list_of_fingerprints[i]        
           if m in if_list_of_fingerprints:
               data.loc[key, m] = 1 
               
       #update presence of a given api call for sample(key) 
       for api in apis.keys():
           m = api
           if m in if_list_of_apis:
                data.loc[key, m] = 1 
                
       #update  others features
       data.loc[key, 'malware'] = malware
       
#save dataset in  csv format
data.to_csv("MSGmalware_analysis_dataset_if.csv",  encoding='utf8')

end_time=time.time()
print('Execution time: '+str(round( end_time -start_time, 3))+'seconds')
"""******************************END****************************************"""
           
  
