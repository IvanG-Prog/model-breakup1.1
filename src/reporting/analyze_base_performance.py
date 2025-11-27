import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# --- Configuración de Rutas ---
TRAINING_DATA_PATH = '/home/ivang/Ivan/git/model-breakup1.1/data/processed/training_data.csv'

def train_and_evaluate_model():
    """Carga los datos, entrena y evalúa un modelo Random Forest."""
    
    print("--- Starting Model Training and Evaluation ---")
    
    # 1. Cargar Datos
    try:
        df = pd.read_csv(TRAINING_DATA_PATH)
        df.dropna(inplace=True) 
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {TRAINING_DATA_PATH}. Asegúrate de ejecutar feature_engineering.py primero.")
        return
        
    if len(df) < 50:
        # Esto ya no es un error, pero es una advertencia útil.
        print(f"Advertencia: Se encontraron solo {len(df)} eventos. El modelo no será estadísticamente fiable.")
        
    print(f"Total de muestras válidas para entrenamiento: {len(df)}")

    # 2. Preparación de Características
    
    # Codificar 'event_type' (short/long) a valores numéricos (0 o 1)
    # 0 = short, 1 = long
    df['event_type_encoded'] = df['event_type'].apply(lambda x: 1 if x.lower() == 'long' else 0)
    
    # Definir características (X) y la etiqueta (Y)
    # Excluimos Relative_Volume por ser 0.0 constante
    features = ['ATR_14', 'RSI_14', 'Slope_50', 'event_type_encoded']
    X = df[features]
    Y = df['Target_Y'] # La etiqueta de resultado (1=TP, 0=SL)
    
    # 3. División de Datos
    # Usamos 75% para entrenar y 25% para probar
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=42, stratify=Y
    )
    
    print(f"Tamaño de Entrenamiento (Train): {len(X_train)} muestras")
    print(f"Tamaño de Prueba (Test): {len(X_test)} muestras")


    # 4. Entrenamiento del Modelo (Random Forest)
    print("\n--- Training Random Forest Classifier ---")
    
    # Usamos Random Forest porque maneja bien la no-linealidad de los datos de trading
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, Y_train)
    
    
    # 5. Evaluación del Modelo
    Y_pred = model.predict(X_test)
    
    print("\n--- Performance Metrics (Test Set) ---")
    
    # Matriz de Confusión
    cm = confusion_matrix(Y_test, Y_pred)   
    print("Matriz de Confusión (Predicción vs Realidad):")
    print(cm)
    
    # Reporte de Clasificación (Precision, Recall, F1-Score)
    print("\nReporte de Clasificación:")
    # zero_division=0 evita errores si una clase no tiene predicciones
    print(classification_report(Y_test, Y_pred, zero_division=0))
    
    # 6. Importancia de las Características
    feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nImportancia de las Características (Para identificar patrones):")
    print(feature_importance)


if __name__ == '__main__':
    train_and_evaluate_model()