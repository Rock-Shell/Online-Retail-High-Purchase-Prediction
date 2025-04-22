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
  

def preprocess(filepath):
    df = pd.read_excel(filepath)
    
    # dropping null values
    print("Before dropping null values:", df.shape)
    df.dropna(inplace=True)
    print("After dropping null values:", df.shape)

    df = df[(df["UnitPrice"]>0) & (df["Quantity"]>0)]
    print("After removing negative order:", df.shape)

    return df






