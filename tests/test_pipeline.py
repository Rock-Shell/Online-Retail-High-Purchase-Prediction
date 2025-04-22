from src.inference_pipeline import predict_new_data
import pandas as pd

def test_prediction():
    df = pd.read_excel("data/raw_data.xlsx").head(5)
    preds = predict_new_data(df)
    assert len(preds) == len(df), "Prediction count mismatch"