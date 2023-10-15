import pickle
from flask import Flask
from flask import request
from flask import jsonify

import requests
import wget

app = Flask('credit_card')

    
model_file_path = r'D:\repolar\MachineLearningZoomCamp_puntoronto\ML_Zoomcamp\model1.bin'
dv_file_path = r'D:\repolar\MachineLearningZoomCamp_puntoronto\ML_Zoomcamp\dv.bin'
with open(model_file_path, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file_path, 'rb') as f_in:
    dv = pickle.load(f_in)

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    result = {
        "churn_probability": round(float(y_pred), 3)
    }
    return jsonify(result)

if __name__ == "main":
    app.run(debug=True, host='0.0.0.0', port=9696)

