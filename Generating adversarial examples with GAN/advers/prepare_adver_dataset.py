# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 23:30:46 2019

@author: MSG
"""
import numpy as np
import pandas as pd

data = np.load('adverV7.npz')
adver_dataset = data['xmal_adver']
adver_dataset= pd.DataFrame(adver_dataset)
#adver_dataset to csv
adver_dataset.to_csv("adver_dataset_test.csv",  encoding='utf8')


#load adver_dataset.csv
adver_dataset = pd.read_csv('adver_dataset_test.csv')
#assign 1 to dependent variable 'malware'
adver_dataset = adver_dataset.replace(np.nan, 1)
#for each feature value > 0.5, ssign it 1 else 0
adver_dataset_if = adver_dataset.where(adver_dataset<.5, 1)
adver_dataset_if = adver_dataset_if.where(adver_dataset_if>.5, 0)

#adver_dataset_if to csv
adver_dataset_if.to_csv("adver_dataset_if_test_2.csv",  encoding='utf8')
#adver_dataset_if.to_csv("adver_dataset_if_train2.csv",  encoding='utf8')
"""******************************END****************************************"""
  