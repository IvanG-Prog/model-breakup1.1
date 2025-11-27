"""
This script performs hyperparameter tuning and training for all multi-target 
classification models. It uses Grid Search to optimize seven individual 
Random Forest Classifiers (one for each Target column: Target_1x to Target_20x) 
and saves the best-performing model for each target.
"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import time
import os

# --- Global Configuration ---
DATA_PATH = '../../data/processed/training_data.csv' 

# Base output path for saving models

MODEL_OUTPUT_PATH_BASE = '../../models/random_forest_multi_target_' 

# The 7 target columns to be trained (short to impulsive objectives)
TARGET_COLUMNS = ['Target_1x', 'Target_2x', 'Target_3x', 'Target_5x', 'Target_10x', 'Target_15x', 'Target_20x']

def tune_and_train_model():
    """
    Loads data, trains, and optimizes seven individual Random Forest models 
    (one per target) using Stratified Split and Grid Search for hyperparameter tuning.
    """
    start_time = time.time()
    
    # 1. Load the dataset
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {DATA_PATH}. Please run feature_engineering.py first.")
        return

    # 2. Prepare Features (X)
    feature_cols = [
        'ATR_14', 
        'RSI_14', 
        'Slope_50', 
        'Rejection_Power', 
        'BB_Position'
    ]
    X = df[feature_cols]

    # 3. Configure Grid Search and Hyperparameters
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [10, None],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [1, 5],
        # Class weighting is crucial for imbalanced targets (like 10x or 20x success)
        'class_weight': [
            None, 
            'balanced', 
            {0: 1, 1: 2}, 
            {0: 1, 1: 3}
        ]
    }
    
    total_combinations = (len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['class_weight']))
                          
    print("--- Starting Hyperparameter Tuning (Grid Search) ---")
    print(f"Testing {total_combinations} total combinations for each of the {len(TARGET_COLUMNS)} targets...")


    # --- Train a model for each Target ---
    for target_col in TARGET_COLUMNS:
        print(f"\n=======================================================")
        print(f"=== STARTING TRAINING FOR: {target_col} ===")
        print(f"=======================================================")
        
        Y = df[target_col]
        
        if Y.nunique() < 2:
            print(f"WARNING: Target '{target_col}' has only one unique value. Skipping.")
            continue
            
        # Split into training and test sets (Stratified to maintain class balance)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.25, random_state=42, stratify=Y
        )
        
        # Define the base model
        rf = RandomForestClassifier(random_state=42)

        # Define Grid Search (using F1-score as the primary optimization metric)
        grid_search = GridSearchCV(
            estimator=rf, 
            param_grid=param_grid, 
            cv=5, 
            scoring='f1', 
            verbose=1, 
            n_jobs=-1
        )

        # 4. Training
        grid_search.fit(X_train, Y_train)

        # 5. Tuning Results
        best_model = grid_search.best_estimator_
        print("\n--- Tuning Results ---")
        print(f"Best Parameters found: {grid_search.best_params_}")
        print(f"Best Cross-Validation Score (F1): {grid_search.best_score_:.4f}")

        # 6. Evaluate the Optimal Model on the Test Set
        Y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)

        print("\n--- Final Performance with Best Model (Test Set) ---")
        print(f"Test Accuracy ({target_col}): {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(Y_test, Y_pred))

        # 7. Model Persistence (save with unique name)
        output_file = f"{MODEL_OUTPUT_PATH_BASE}{target_col}.pkl"
        joblib.dump(best_model, output_file)
        print(f"\n--- Model Persistence ---")
        print(f"✅ Optimal model saved to: {output_file}")

    
    end_time = time.time()
    print(f"\nProcess completed in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    tune_and_train_model()