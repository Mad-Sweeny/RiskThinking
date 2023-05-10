from flask import Flask, jsonify, request
import pickle
import numpy as np
import json

#app = Flask(__name__)
with open(r'trained_model/random_forest_regressor_model.pkl','rb') as f:
    model = pickle.load(f)
# model = pickle.load(r'..\trained_model\random_forest_regressor_model.pkl')


#@app.route('/predict')
def predict(requests):
    vol_moving_avg = request.args.get('vol_moving_avg')
    adj_close_rolling_med = request.args.get('adj_close_rolling_med')
    features = [vol_moving_avg,adj_close_rolling_med]
    
    predictions = model.predict([features])[0]
    
    return jsonify({'predictions' : predictions})   

#if __name__ == '__main__':
#    app.run(debug=True)
