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


# In[3]:


spark = SparkSession.builder.appName('Feature Engineering').getOrCreate()


# In[4]:


df_market_data = spark.read.parquet(r'..\data\processed_output.parquet') 


# In[5]:


df_market_data = df_market_data.withColumn('unix_time',F.unix_timestamp('Date','yyyy-MM-dd'))


# In[6]:


w = (Window().partitionBy('Symbol').orderBy('Unix_time').rowsBetween(-30,0))

sum_udf = F.udf(lambda x: int(np.sum(x)),IntegerType())

df_market_data = df_market_data.withColumn('prev_vols_vals',F.array(F.collect_list('Volume').over(w)))                .withColumn('prev_vals_size',F.size(F.collect_list('Volume').over(w)))                .withColumn('vol_moving_avg',F.when(F.col('prev_vals_size') > 30, F.round(sum_udf('prev_vols_vals')/30)))


# In[7]:


def calc_median(x):
    x = np.sort(x)
    return float(sum(x[0][13:15]))

calc_median_udf = F.udf(lambda x: calc_median(x), FloatType())

df_market_data = df_market_data.withColumn('prev_adj_close_vals',F.array(F.collect_list('Adj_Close').over(w)))                .withColumn('adj_close_rolling_med',F.when(F.col('prev_vals_size') > 30, F.round(calc_median_udf('prev_adj_close_vals')/2,4))) 


# In[8]:


df_market_data    .select('Symbol','Security_Name','Date','Open','High','Low','Close','Adj_Close','Volume','vol_moving_avg','adj_close_rolling_med')    .coalesce(1)    .write    .format('parquet')    .mode('overwrite')    .save(r'..\data\feature_engineering_output.parquet')


# In[9]:


# df_market_data.select('Date','adj_close_rolling_med','vol_moving_avg').show(35)


# In[ ]:




