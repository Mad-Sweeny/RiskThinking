import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from sklearn.preprocessing import StandardScaler

def train_model():
    #data = pd.read_csv(r'/home/ebony_maw_0/data/model_training_data/*.csv')
    print('reading csv data')
    data = pd.read_csv(r'/home/ebony_maw_0/data/model_training_data/training_data.csv')
    
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    print('Dropping null values')
    data.dropna(inplace=True)

    features = ['vol_moving_avg', 'adj_close_rolling_med']
    target = 'Volume'

    scaler_x=StandardScaler()
    scaler_y=StandardScaler()

    X = pd.DataFrame(data[features])
    y = pd.DataFrame(data[target])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_scaled = scaler_x.fit_transform(X_train.values)
    X_train_scaled = pd.DataFrame(X_train_scaled,index=X_train.index,columns=features)

    X_test_scaled = scaler_x.transform(X_test.values)
    X_test_scaled = pd.DataFrame(X_test_scaled,index=X_test.index,columns=features)

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1))
    y_train['scaled_volume'] = y_train_scaled

    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1))
    y_test['scaled_volume'] = y_test_scaled
    

    print('Training rf regressor model')
    # Create a RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=40, random_state=42)

    # Train the model
    model.fit(X_train_scaled, y_train['scaled_volume'])

    # Make predictions on test data
    y_pred = model.predict(X_test_scaled)

    # Calculate the Mean Absolute Error and Mean Squared Error
    mae = mean_absolute_error(y_test['scaled_volume'], y_pred)
    mse = mean_squared_error(y_test['scaled_volume'], y_pred)

    print('Mean Absolute Error is {} and Mean squared Error is {}'.format(mae,mse))

    
    # Saving trained model to disk
    filename = r'/home/ebony_maw_0/trained_model/random_forest_regressor_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    
    #print('Model Trained Successfully and saved to disk')
    return('Model Trained Successfully and saved to disk.')





