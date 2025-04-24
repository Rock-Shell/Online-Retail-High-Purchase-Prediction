import pandas as pd

from src.data_preprocessing import preprocess
from src.feature_eng import FeatureEngineering
from src.model_training import train_model
# from src.inference_pipeline import predict_new_data


def main():
    preprocessed_df = preprocess(filepath="data\Online Retail.xlsx")

    # split into train and test
    df = preprocessed_df.copy()
    df["Year-month"] = pd.to_datetime(df["InvoiceDate"]).dt.to_period('M')

    # Train test split on last months data
    print(df[(df["Year-month"] == "2011-12") | (df["Year-month"]=="2011-11")]["InvoiceNo"].count()/df["InvoiceNo"].count()*100)
    train, test = df[df["Year-month"]<"2011-11"], df[df["Year-month"]>="2011-11"]
    
    train_df, test_df = FeatureEngineering.transform(train), FeatureEngineering.transform(test)
    train_x, train_y = train_df.drop("High Purchase", axis=1), train_df["High Purchase"]
    test_x, test_y = test_df.drop("High Purchase", axis=1), test_df["High Purchase"]
    
    train_model(train_x, train_y)
    
    # not evaluating on any metrics, just predicting and asserting the count of data is correct
    predict_new_data(test_x)
main()
