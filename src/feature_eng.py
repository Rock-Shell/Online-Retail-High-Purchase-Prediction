from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


def hour_bin(hr):
  if hr>= 5 and hr<= 10:
    return "Morning"
  elif hr>=11 and hr <= 16:
    return "Afternoon"
  elif hr>=17 and hr<= 21:
    return "Evening"
  else:
    return "Night"
  

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["Invoicehour"] = df["InvoiceDate"].dt.hour
        df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
        df["isWeekend"] = df["DayOfWeek"].isin([5,6]).astype(int)

        df["InvoicehourBin"] = df["Invoicehour"].apply(hour_bin)

        # add previous purchase count per customer
        sorted_df = df.sort_values(by=['CustomerID', 'InvoiceDate'])
        # Group and rank: For each customer, count the number of purchases till the previous transaction
        df['PreviousPurchaseCount'] = sorted_df.groupby('CustomerID').cumcount()

        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"]).dt.date

        # find country wise mean unit price, mean quanity bought
        df2 = df.groupby("Country").agg({"Quantity": "mean", "UnitPrice": "mean", "StockCode": "nunique"}).reset_index()
        df2 = df.merge(df2[["Country", "Quantity", "StockCode"]], on="Country", how="left", suffixes=("_x", "_country_avg"))
        
        session_data = df2.groupby(["InvoiceDate", "CustomerID", "Invoicehour"]).agg({"Quantity_x": "sum", "UnitPrice": "mean", "StockCode_x": "nunique", "Quantity_country_avg": "mean", "StockCode_country_avg": "mean"}).reset_index()
        session_data["High Purchase"] = (session_data["Quantity_x"] > 10).astype(int)
        
        # select features
        return session_data.drop(["CustomerID", "InvoiceDate"], axis=1)

