#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init()


# In[2]:


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle


# In[3]:


spark = SparkSession.builder.appName('Model Training').getOrCreate()


# In[4]:


df_fe_data = spark.read.parquet(r'..\data\feature_engineering_output.parquet').select('Date','Volume','vol_moving_avg', 'adj_close_rolling_med')


# In[5]:


data = df_fe_data.toPandas()

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)


# In[ ]:


(data.isnull().sum()/data.shape[0])*100


# In[ ]:


data.dropna(inplace=True)


# In[ ]:


features = ['vol_moving_avg', 'adj_close_rolling_med']
target = 'Volume'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Create a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)


# In[ ]:


# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate the Mean Absolute Error and Mean Squared Error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print('Mean Absolute Error is {} and Mean squared Error is {}'.format(mae,mse))


# In[ ]:


# Saving trained model to disk
filename = r'..\trained_model\random_forest_regressor_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)


# In[ ]:




