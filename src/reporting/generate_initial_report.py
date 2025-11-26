"""
This script loads the processed training data, trains a simple Random Forest 
Classifier as a baseline model, and generates a visual bar chart of the 
Precision, Recall, and F1-Score classification metrics for quick performance evaluation.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import os

# --- Configuration ---
TRAINING_DATA_PATH = '../../data/processed/training_data.csv' 

def plot_training_results():
    """
    Trains the baseline Random Forest model, evaluates its performance, 
    and generates a visual report of the classification metrics (Precision, 
    Recall, F1-Score) for classes 0 (SL/Miss) and 1 (TP/Hit).
    """
    
    print("--- Starting Training and Visual Report Generation ---")
    
    # 1. Load and Prepare Data
    try:
        df = pd.read_csv(TRAINING_DATA_PATH)
        df.dropna(inplace=True)
    except FileNotFoundError:
        print(f"Error: File not found at {TRAINING_DATA_PATH}. Terminating.")
        return
        
    TARGET_COLUMN = 'Target_1x' # Modify this if your target is named differently
    
    df['event_type_encoded'] = df['event_type'].apply(lambda x: 1 if x.lower() == 'long' else 0)
    
    # Features (X)
    features = ['ATR_14', 'RSI_14', 'Slope_50', 'event_type_encoded']
    X = df[features]
    
    # Target (Y)
    Y = df[TARGET_COLUMN] 

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=42, stratify=Y
    )
    
    # 2. Train the Model (Using initial parameters)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, Y_train)
    
    # 3. Predict and Get Classification Report
    Y_pred = model.predict(X_test)
    report = classification_report(Y_test, Y_pred, output_dict=True, zero_division=0)
    
    # Extract metrics for the graph
    df_report = pd.DataFrame(report).transpose()
    
    # We only care about the metrics for classes 0 and 1
    metrics_to_plot = df_report.loc[['0', '1'], ['precision', 'recall', 'f1-score']]

    # 4. Plotting the Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.25
    r1 = np.arange(len(metrics_to_plot.columns))
    r2 = [x + bar_width for x in r1]
    
    # Plot bars for Class 0 (Miss/SL) and Class 1 (Hit/TP)
    ax.bar(r1, metrics_to_plot.loc['0'].values, color='#ff7f0e', width=bar_width, edgecolor='grey', label='Class 0 (SL/Miss)') 
    ax.bar(r2, metrics_to_plot.loc['1'].values, color='#2ca02c', width=bar_width, edgecolor='grey', label='Class 1 (TP/Hit)') 
    
    # Add metrics values on top of bars
    for i in range(len(r1)):
        ax.text(r1[i], metrics_to_plot.loc['0'].values[i] + 0.01, f"{metrics_to_plot.loc['0'].values[i]:.2f}", ha='center', va='bottom')
        ax.text(r2[i], metrics_to_plot.loc['1'].values[i] + 0.01, f"{metrics_to_plot.loc['1'].values[i]:.2f}", ha='center', va='bottom')

    # Add labels, title, and custom x-axis tick labels
    ax.set_xlabel('Classification Metrics', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xticks([r + bar_width / 2 for r in r1])
    ax.set_xticklabels(metrics_to_plot.columns.str.capitalize())
    ax.set_title('Baseline Model Performance by Class', fontweight='bold')
    
    ax.legend()
    ax.set_ylim(0, 1.05) 
    
    plt.tight_layout()
    
    # Save the image
    OUTPUT_FILE = '../../data/processed/training_report_initial.png'
    plt.savefig(OUTPUT_FILE)
    print(f"✅ Classification Report Plot saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    plot_training_results()