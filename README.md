# RiskThinking

Link to the problem statement: https://github.com/RiskThinking/work-samples/blob/main/Data-Engineer.md

Assumptions:
  1. All the neccessary softwares (python, spark and airflow) are already installed
  2. Data from kaggle is already downloaded and saved on machine
  

Problem 1: File code/Raw_To_processed.py contains the pyspark code for this particular problem statement. It reads the data from a local directory (assumption: kaggle dataset is already downloaded on the machine) and stores it in parquet format in a seperate directory

Problem 2: File code/Feature_Engineering.py contains the pyspark code for this problem statement. This code reads the parquet file generateda as part of output from Problem 1 and derives 2 new columns using custom functions. Finally the new data is stores in parquet as well as csv format in a seperate directries.

Problem 3: File code/Model_Training_Function.py contains the code for training the rf regressor model. This code reads the csv file generated in Problem 3 and trains the random forest model using that data. Model training parameter 'n_estimators' is reduced from 100 to 40 because the size of model after saving it on disk was approximately 17Gb's and it was difficult to host this model on gcf. The trained model is stored in .pkl file in a seperate directory

Problem 4: code/prediction_api.py contains the code which serves the trained ml model. This api hosted as a google cloud function.

Orchestration:
  The entire pipeline is orchestrated using Airflow. File airflow/dags/DAG_Raw_to_processed.py contains the code wherein all the 3 python code files for Problems 1-3 are executed. 
  The airflow execution logs are present in airflow/logs folder
  
Api Link: https://us-central1-rt-de-sample-task.cloudfunctions.net/market-data 
