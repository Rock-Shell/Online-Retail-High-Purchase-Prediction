import xgboost as xgb
import joblib


def train_model(x_train, y_train):
    best_params = {'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.7}
    model = xgb.XGBClassifier(
        objective="binary:logistic", 
        use_label_encoder=False, 
        eval_metric="logloss", 
        random_sate=42, 
        scale_pos_weight=38.27,
        **best_params
        )
    
    model.fit(x_train, y_train)
    joblib.dump(model, "models/xgboost_model.pkl")
