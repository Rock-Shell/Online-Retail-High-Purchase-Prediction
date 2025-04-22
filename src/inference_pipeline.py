import joblib
import pandas as pd

model = joblib.load("models/xgboost_model.pkl")
feature_pipeline = joblib.load("models/feature_pipeline.pkl")

def predict_new_data(raw_data):
    processed = feature_pipeline.transform(raw_data)
    X = processed.drop("High Purchase") # might raise error, if already dropped
    predictions = model.predict(X)
    return predictions