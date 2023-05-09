#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init()


# In[2]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *


# In[3]:


spark = SparkSession.builder.appName('Pre-processing').getOrCreate()


# In[4]:


regex_str = "([^\/]+)$"
schema = StructType([                     StructField('Date',DateType(),True),                     StructField('Open',FloatType(),True),                     StructField('High',FloatType(),True),                     StructField('Low',FloatType(),True),                     StructField('Close',FloatType(),True),                     StructField('Adj_Close',FloatType(),True),                     StructField('Volume',IntegerType(),True),                     StructField('Symbol',StringType(),True)
                    ])


# In[5]:


df_stocks = spark.read.format('csv').option('header',True).schema(schema).load(r'..\data\stocks\*.csv')                      .withColumn('Symbol',regexp_replace(regexp_extract(input_file_name(),regex_str,1),'.csv',''))   
# df_stocks.show(5)

df_etfs = spark.read.format('csv').option('header',True).schema(schema).load(r'..\data\etfs\*.csv')                      .withColumn('Symbol',regexp_replace(regexp_extract(input_file_name(),regex_str,1),'.csv',''))   
# df_etfs.show(5)

df_market_data = df_stocks.unionAll(df_etfs)


# In[6]:


df_symbols = spark.read.csv(r'..\data\symbols_valid_meta.csv',header=True).select('Symbol','Security Name')                .withColumnRenamed('Security Name','Security_Name')
# df_symbols.show(5)


# In[7]:


df_processed = df_market_data.join(broadcast(df_symbols), how = 'inner', on = df_stocks.Symbol == df_symbols.Symbol)                .select(df_stocks.Symbol,
                        df_symbols.Security_Name,
                        df_stocks.Date,
                        df_stocks.Open,
                        df_stocks.High,
                        df_stocks.Low,
                        df_stocks.Close,
                        df_stocks.Adj_Close,
                        df_stocks.Volume)
# df_processed.show(5)


# In[8]:


df_processed.coalesce(1).write.mode('overwrite').save(r'..\data\processed_output.parquet')

