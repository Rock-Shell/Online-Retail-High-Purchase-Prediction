                                            ======================================================================
                                            Supervised: Predicting whether customer will make high purchase or not
                                            ======================================================================


DATA HANDLING
  drop null values
  drop negative orders (return or cancelled)
  extract features from date (hour, day, isweekend)
  Validations
    Check if user places more than one order in a day
    check there is one to one mapping between invoice no and customerID
  Create hours bins, morning, evening..

TRAIN TEST SPLIT
  group by year-month
  calculate percentage for order of last 2 months (20%)
  use it as test DATA

FURTHER PROCESSING
  add previous purchase count for customer
  find mean unit price, mean quanity, and unique stocks per country
  Mark where quantity>10, as high purchase and use it as target variable
  Apply processing to both train and test DATA
  Check for imbalance
  Check for correlated features

MODELLING
  Use xgboost classifier for prediction, set scale_pos_weight as imbalance ratio
  use cross validation, and calculate average CV score
  Use gridsearchCV to try different parameters, choose best model
  
Evaluation
  evaluate training and test data, print confusion matrix and and classification report
  get shap values
