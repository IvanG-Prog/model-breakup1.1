"""
This script acts as the prediction engine (inference module). It loads all 7 
trained Random Forest models (one for each Target), predicts the success probability 
for a given signal (data point), and generates a detailed statistical report.
"""
import pandas as pd
import joblib
import os
import numpy as np

# --- Global Configuration ---
# Corregimos la ruta para usar ABSOLUTE PATH
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(BASE_PATH, 'models') 

TARGET_COLUMNS = ['Target_1x', 'Target_2x', 'Target_3x', 'Target_5x', 'Target_10x', 'Target_15x', 'Target_20x']

# --- REAL HISTORICAL PRECISION (Class 1) ---
# Used as the decision threshold for conservative trading.
HISTORICAL_PRECISION = {
    'Target_1x': 0.98, 'Target_2x': 0.96, 'Target_3x': 0.94, 
    'Target_5x': 0.89, 'Target_10x': 0.86, 'Target_15x': 0.84, 
    'Target_20x': 0.79
}

def load_all_models():
    """
    Loads all 7 trained Random Forest models from the MODELS_DIR directory.
    
    Returns:
        dict: A dictionary mapping target names to their loaded model objects.
    """
    models = {}
    print("Loading all 7 multi-target models...")
    for target in TARGET_COLUMNS:
        model_path = os.path.join(MODELS_DIR, f'random_forest_multi_target_{target}.pkl')
        try:
            models[target] = joblib.load(model_path)
        except FileNotFoundError:
            print(f"⚠️ ERROR: Model not found at {model_path}. Ensure tune_model.py has been executed.")
            return None
    return models

def predict_probabilities(data_point, models):
    """
    Predicts the probability of success (Class 1) for each target using the 
    corresponding trained model for a single input data point.
    
    Args:
        data_point (pd.DataFrame): DataFrame containing the input features for one signal.
        models (dict): Dictionary of loaded model objects.
        
    Returns:
        dict: A dictionary mapping target names to their success probability (%).
    """
    results = {}
    for target, model in models.items():
        # predict_proba returns [[Prob_Class_0, Prob_Class_1]]
        probability_of_success = model.predict_proba(data_point)[:, 1][0]
        results[target] = round(probability_of_success * 100, 2)
    return results

def calculate_break_even(target_value):
    """
    Calculates the minimum required success probability (Break Even) based on 
    the Risk-to-Reward (R:R) ratio for the given target value.
    
    Args:
        target_value (int): The ATR multiplier for the Take Profit (e.g., 5 for 5x ATR).
        
    Returns:
        float: The minimum required success probability (%) to break even.
    """
    
    # Define Stop Loss multiplier (1x or 3x)
    sl_multiplier = 1.0
    if target_value in [10, 15, 20]:
        sl_multiplier = 3.0
        
    # Break Even Formula: Risk / (Risk + Reward)
    be_needed = (sl_multiplier / (sl_multiplier + target_value)) * 100 
    return round(be_needed, 2)

def main():
    """
    Main function to load models, simulate a signal, generate predictions, 
    and output the statistical advantage report.
    """
    all_models = load_all_models()
    if not all_models:
        return
    
    print("\n--- 🧠 PREDICTION DIAGNOSTICS (Inference Engine) ---")
    
    # --- EXAMPLE USE CASE: Simulating a strong signal ---
    # The 5 features expected by the model (Note: event_type_encoded is missing here, 
    # assuming this example is for a specific trade type, or must be added.)
    # The final model must use all features used in tune_model.py
    
    # NOTE: The example features provided are ATR_14, RSI_14, Slope_50, Rejection_Power, BB_Position. 
    # We must ensure the model features match this set.
    ejemplo_features = {
        'ATR_14': [18.86], 
        'RSI_14': [32.60], 
        'Slope_50': [-3.59], 
        'Rejection_Power': [0.1285], 
        'BB_Position': [0.2699]
        # 'event_type_encoded': [1] # Assuming 'long' for the simulation
    }
    
    df_example = pd.DataFrame(ejemplo_features)

    # 2. Get success probabilities
    probabilities = predict_probabilities(df_example, all_models)

    print("\n📊 STATISTICAL ADVANTAGE REPORT")
    
    # Print the results in a concise, organized table
    print("| Target | R:R | P Prediction | SL | BE Minimum | **Net Advantage** | Suggested Decision |")
    print("|:------:|:---:|:------------:|:--:|:---------:|:------------------:|:------------------|")
    
    for target, prob in probabilities.items():
        target_value = int(target.split('_')[1].replace('x', ''))
        
        # Calculate Break Even and Net Advantage
        be_needed = calculate_break_even(target_value)
        
        # SL used for this target
        sl_used = '1x' if target_value <= 5 else '3x'
        
        # R:R used
        rr_value = round(target_value / float(sl_used.replace('x', '')), 2)
        
        # Net Advantage: how much the prediction exceeds the break-even minimum.
        advantage = round(prob - be_needed, 2)
        
        # Use Real Historical Precision as the decision threshold
        required_precision_hist = HISTORICAL_PRECISION.get(target, 0.0) * 100
        
        # The decision is based on whether the prediction exceeds the model's proven historical performance
        status = '🟢 OPEN' if prob > required_precision_hist else '🟡 AVOID'
        
        # Format Net Advantage
        advantage_str = f"+{advantage}%" if advantage > 0 else f"{advantage}%"
        
        print(f"| {target:^6} | {rr_value:^3} | {prob:^12.2f}% | {sl_used:^2} | {be_needed:^9.2f}% | **{advantage_str:^16}** | {status:^17} |")
        
    print("\n--- Multiple Position Strategy ---")
    print("✅ You open multiple separate trades (3x, 5x, 10x, etc.) only if the P Prediction exceeds the historical threshold.")

if __name__ == "__main__":
    main()