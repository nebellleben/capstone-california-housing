import pickle
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any
import pandas as pd

app = FastAPI(title="California Housing Prediction Model")

with open('baseline_model.pkl', 'rb') as f:
    baseline_model = pickle.load(f)

with open('lasso_model.pkl', 'rb') as f:
    lasso_model = pickle.load(f)

with open('ridge_model.pkl', 'rb') as f:
    ridge_model = pickle.load(f)

with open('nn_model.pkl', 'rb') as f:
    nn_model = pickle.load(f)

def predict_single(housing_data):
    df = pd.DataFrame([housing_data])
    df_dict = df.to_dict(orient='records')
    X = nn_model.transform(df_dict)

    y_pred_baseline = float(baseline_model.predict(X)[0]) 
    y_pred_lasso = float(lasso_model.predict(X)[0]) 
    y_pred_ridge = float(ridge_model.predict(X)[0]) 
    y_pred_nn = float(nn_model.predict(X)[0]) 

    return [y_pred_baseline, y_pred_lasso, y_pred_ridge, y_pred_nn]


@app.post("/predict")
def predict(housing_data: Dict[str, Any]):
    results = predict_single(housing_data)

    return {
        "median house value (baseline model)": results[0],
        "median house value (lasso model)": results[1],
        "median house value (ridge model)": results[2],
        "median house value (neural network model)": results[3]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)