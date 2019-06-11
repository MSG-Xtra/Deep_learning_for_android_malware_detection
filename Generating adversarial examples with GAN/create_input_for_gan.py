# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:39:22 2019
@author: MSG
===============================================================================
**********Prepare data for GAN usage**********
*******This script converts csv dataset to .npz format for GAN input*******
===============================================================================
"""

import pandas as pd
import numpy as np
import time

start_time =time.time()
dataset = pd.read_csv('MSGmalware_analysis_dataset_if.csv') 

#spliting dataset into x and y
dataset_b = dataset[dataset['malware']==0]
dataset_m = dataset[dataset['malware']==1]

xben = dataset_b.iloc[:, 1:358].values
yben = dataset_b.iloc[:, 358].values

xmal = dataset_m.iloc[:, 1:358].values
ymal = dataset_m.iloc[:, 358].values

#Save data in .npz format......under test

np.savez('dataset_if.npz', xmal=xmal, ymal=ymal, xben=xben, yben=yben)
end_time=time.time()

print('Execution time: '+str(round( end_time -start_time, 3))+'seconds')
"""******************************END****************************************"""
           
  
