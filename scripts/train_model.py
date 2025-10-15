import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_recommender_model():
    """
    Loads the training data, trains separate models for temperature and
    pressure adjustments, and saves them for later use.
    """
    print("--- Starting Model Training ---")
    
    # 1. Load the generated training data
    try:
        df = pd.read_csv('data/processed/training_data.csv')
    except FileNotFoundError:
        print("Error: training_data.csv not found.")
        print("Please run 'python scripts/generate_training_data.py' first.")
        return

    # 2. Define the features (X) and the targets (y)
    # The features are the 'problem state' of the reactor
    features = ['Time', 'Current_NH3', 'Deviation', 'Current_Temp', 'Current_Press']
    X = df[features]
    
    # The targets are the 'solution adjustments' we want to predict
    y_temp = df['Temp_Adjustment']
    y_press = df['Press_Adjustment']

    # 3. Split the data into training and testing sets
    X_train, X_test, y_temp_train, y_temp_test, y_press_train, y_press_test = train_test_split(
        X, y_temp, y_press, test_size=0.2, random_state=42
    )

    # 4. Train the Temperature Adjustment Model
    print("Training Temperature model...")
    temp_model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=1000, 
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1 # Use all available CPU cores
    )
    # FIT METHOD MODIFIED FOR COMPATIBILITY: Removed early stopping arguments
    temp_model.fit(X_train, y_temp_train, verbose=False)

    # 5. Train the Pressure Adjustment Model
    print("Training Pressure model...")
    press_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    )
    # FIT METHOD MODIFIED FOR COMPATIBILITY: Removed early stopping arguments
    press_model.fit(X_train, y_press_train, verbose=False)

    # 6. Evaluate the models on the test set
    temp_preds = temp_model.predict(X_test)
    press_preds = press_model.predict(X_test)
    
    print("\n--- Model Performance ---")
    print(f"Temperature Model R²: {r2_score(y_temp_test, temp_preds):.3f}")
    print(f"Pressure Model R²:    {r2_score(y_press_test, press_preds):.3f}")

    # 7. Save the trained models for use in the dashboard
    model_dir = 'reactor_model'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(temp_model, os.path.join(model_dir, 'temp_model.joblib'))
    joblib.dump(press_model, os.path.join(model_dir, 'press_model.joblib'))
    print(f"\nModels saved successfully to the '{model_dir}' directory.")

if __name__ == "__main__":
    train_recommender_model()

