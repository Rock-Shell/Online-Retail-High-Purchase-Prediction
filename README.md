                                  ===========================================================
                                  Predicting whether customer will make high purchase or not
                                  ===========================================================


**DATA HANDLING**

  drop null values
  drop negative orders (return or cancelled)
  extract features from date (hour, day, isweekend)
  Validations
    Check if user places more than one order in a day
    check there is one to one mapping between invoice no and customerID
  Create hours bins, morning, evening..

**TRAIN TEST SPLIT**

  group by year-month
  calculate percentage for order of last 2 months (20%)
  use it as test DATA

**FURTHER PROCESSING**

  add previous purchase count for customer
  find mean unit price, mean quanity, and unique stocks per country
  Mark where quantity>10, as high purchase and use it as target variable
  Apply processing to both train and test DATA
  Check for imbalance
  Check for correlated features

**MODELLING**
  Use xgboost classifier for prediction, set scale_pos_weight as imbalance ratio
  use cross validation, and calculate average CV score
  Use gridsearchCV to try different parameters, choose best model
  
**Evaluation**

  evaluate training and test data, print confusion matrix and and classification report
  get shap values



![alt text](https://github.com/Rock-Shell/Online-Retail-High-Purchase-Prediction/blob/main/images/corr.png)



                                  ======================================================
                                  Unsupervised: Customer segmention
                                  ======================================================

**PREPROCESSING**

  dropping null values, and negative orders
  add OrderValue, total orders, count of orders, and mean of previous purchase per customer

**FEATURE ENGINEERING**

  Outlier detection using boxplot
  applying log transformation
  plot heatmap, to look for correlated features

  ![alt text](https://github.com/Rock-Shell/Online-Retail-High-Purchase-Prediction/blob/main/images/outliers.png)


**MODELLING**

  apply standard scaler
  use elbow method to find optimum number of clusters
  optimum k is where graph starts flattening
  use k means with this optimum k value

  ![alt text](https://github.com/Rock-Shell/Online-Retail-High-Purchase-Prediction/blob/main/images/elbow%20method.png)

**EVALUATION**

  Use Silhoutte score, ~1 Good clustering, non overlapping clusters

![alt text](https://github.com/Rock-Shell/Online-Retail-High-Purchase-Prediction/blob/main/images/customer%20segments.png)
