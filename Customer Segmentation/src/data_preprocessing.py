def preprocess(df):
  # # dropping null values
  print("Before dropping null values:", df.shape)
  df.dropna(inplace=True)
  print("After dropping null values:", df.shape)

  print(df.shape)
  df = df[(df["UnitPrice"]>0) & (df["Quantity"]>0)]
  print("After removing negative order:", df.shape)

  df["OrderValue"] = df["UnitPrice"] * df["Quantity"]
  sorted_df = df.sort_values(by=['CustomerID', 'InvoiceDate'])
  # Group and rank: For each customer, count the number of purchases till the previous transaction
  df['PreviousPurchaseCount'] = df.groupby('CustomerID').cumcount()

  cust_seg = df.groupby(["CustomerID"]).agg({"Quantity": "sum", "InvoiceNo": "count", "OrderValue": "mean", "PreviousPurchaseCount": "sum"}).reset_index()
  cust_seg.rename(columns={"InvoiceNo": "TotalOrders"}, inplace=True)
  cust_seg = cust_seg.merge(df[["CustomerID", "Country"]].drop_duplicates(), on="CustomerID", how="left")
  return cust_seg