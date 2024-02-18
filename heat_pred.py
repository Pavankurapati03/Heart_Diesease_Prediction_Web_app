# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 01:37:05 2024

@author: Pavan
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


loadmodel = pickle.load(open('C:/Users\Pavan/Downloads/Heart/trained_new_model.sav', 'rb'))

input_data = (51,0,2,130,256,0,0,149,0,0.5,2,0,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')