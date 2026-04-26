"""
Script de entrenamiento ejecutado durante el build en Render.
Modelo: CatBoost calibrado con Optuna — mejor modelo del proyecto
Métricas en validación: RMSE 8.73 | MAE 6.05 | R² 0.845
"""

import pandas as pd
import numpy as np
import joblib
import os
from catboost import CatBoostRegressor

TRAIN_URL = "https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2026/main/datasets/dataTrain_Spotify.csv"
MODEL_PATH = "model_pipeline.pkl"

# Variables exactas del notebook (Proyecto1_Popularidad_Canciones.ipynb)
NUMERICAS = [
    'duration_ms', 'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo'
]
CATEGORICAS = ['explicit', 'key', 'mode', 'time_signature', 'track_genre']
TEXTO       = ['artists', 'album_name', 'track_name']
PREDICTORES = NUMERICAS + CATEGORICAS + TEXTO

# Columnas que CatBoost manejará como categóricas (strings)
# key, mode, time_signature son int -> CatBoost los trata como numéricos
CAT_FEATURES = ['explicit', 'track_genre', 'artists', 'album_name', 'track_name']

# Mejores hiperparámetros encontrados con Optuna (Trial 12)
BEST_PARAMS = {
    "iterations":          3072,
    "depth":               9,
    "learning_rate":       0.049006456784407286,
    "l2_leaf_reg":         1.3950429102859259,
    "bagging_temperature": 1.1238363877194946,
    "random_strength":     0.7180695088113583,
    "loss_function":       "RMSE",
    "eval_metric":         "RMSE",
    "random_seed":         42,
    "verbose":             200,
}


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CAT_FEATURES:
        df[col] = df[col].fillna("missing").astype(str)
    return df[PREDICTORES]


def train():
    print("=" * 60)
    print("Descargando datos de entrenamiento...")
    data = pd.read_csv(TRAIN_URL, index_col=0)
    print(f"  Registros: {len(data):,} | Variables: {len(PREDICTORES)}")

    X = preprocess(data)
    y = data["popularity"]

    print("\nEntrenando CatBoost con hiperparámetros óptimos (Optuna)...")
    print(f"  iterations={BEST_PARAMS['iterations']} | depth={BEST_PARAMS['depth']} | lr={BEST_PARAMS['learning_rate']:.4f}")

    model = CatBoostRegressor(**BEST_PARAMS)
    model.fit(X, y, cat_features=CAT_FEATURES)

    artifact = {
        "model":       model,
        "predictores": PREDICTORES,
        "cat_features": CAT_FEATURES,
    }
    joblib.dump(artifact, MODEL_PATH)
    size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
    print(f"\nModelo guardado: '{MODEL_PATH}' ({size_mb:.2f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    train()
