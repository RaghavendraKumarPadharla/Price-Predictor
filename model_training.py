import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models"""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            tree_method='hist',
            enable_categorical=False,
            device='cpu'
        )
    }

    results = {}
    best_model = None
    best_r2 = -np.inf

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        accuracy = (1 - mape) * 100

        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r2,
            'MAPE': mape,
            'Accuracy': accuracy
        }

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    return results, best_model

def save_model(model, scaler, label_encoders, metrics):
    """Save the model and preprocessing objects"""
    if isinstance(model, XGBRegressor):
        model.save_model('price_predictor_model.json')  # XGBoost native format
    else:
        joblib.dump(model, 'price_predictor_model.joblib')

    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(label_encoders, 'label_encoders.joblib')
    joblib.dump(metrics, 'model_metrics.joblib')

if __name__ == "__main__":
    df = pd.read_csv('processed_data.csv')
    X = df.drop('Price (USD)', axis=1)
    y = df['Price (USD)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results, best_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    print("\nModel Evaluation Results:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    scaler = joblib.load('scaler.joblib')
    label_encoders = joblib.load('label_encoders.joblib')
    save_model(best_model, scaler, label_encoders, results)
