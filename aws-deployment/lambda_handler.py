import pickle
from fastapi import FastAPI
from mangum import Mangum
from typing import Dict, Any
import pandas as pd

app = FastAPI(title="California Housing Prediction Model")

# Load models at container startup (Lambda container reuse)
with open('model.pkl', 'rb') as f:
    dv, baseline_model, lasso_model, ridge_model, nn_model = pickle.load(f)

def predict_single(housing_data):
    df = pd.DataFrame([housing_data])
    df_dict = df.to_dict(orient='records')
    X = dv.transform(df_dict)

    y_pred_baseline = float(baseline_model.predict(X)[0]) 
    y_pred_lasso = float(lasso_model.predict(X)[0]) 
    y_pred_ridge = float(ridge_model.predict(X)[0]) 
    # Neural network prediction - suppress verbose output in Lambda
    y_pred_nn = float(nn_model.predict(X, verbose=0)[0]) 

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


@app.get("/")
def root():
    return {"message": "California Housing Prediction API", "status": "healthy"}


# Lambda handler - Mangum wraps FastAPI for Lambda
handler = Mangum(app, lifespan="off")

