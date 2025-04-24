import pandas as pd
from src.data_preprocessing import preprocess
from src.feature_eng import FeatureEngineering
from src.model_training import train_model

fe = FeatureEngineering

def main():
    preprocessed_df = preprocess(filepath="data/Online Retail.xlsx")
    df = fe.fit_transform(preprocessed_df.copy())
    train_model(df)

