import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def XGBoost_bias(df, features, target='bias', test_size=0.2, random_state=42,
                          n_estimators=550, max_depth=5, learning_rate=0.01,
                          subsample=0.5, colsample_bytree=0.75):
    """
    Train an XGBoost model to predict bias and apply it to the simulated measurements.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data containing features and target.
    features : list
        List of column names to use as features.
    target : str
        Name of the target column (bias).
    test_size : float
        Fraction of data to use for testing.
    random_state : int
        Random seed for reproducibility.
    n_estimators, max_depth, learning_rate, subsample, colsample_bytree : float
        XGBoost hyperparameters.
    
    Returns
    -------
    model : xgb.XGBRegressor
        Trained XGBoost model.
    """
    # Prepare data
    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train model
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Test MSE:", mse)

    return model