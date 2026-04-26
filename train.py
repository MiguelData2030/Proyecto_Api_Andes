"""
Script de entrenamiento que se ejecuta durante el build en Render.
Descarga los datos, entrena el pipeline LightGBM y guarda model_pipeline.pkl.
"""

import pandas as pd
import numpy as np
import joblib
import os

from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

TRAIN_URL = "https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2026/main/datasets/dataTrain_Spotify.csv"
MODEL_PATH = "model_pipeline.pkl"

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_artists"]      = df["artists"].fillna("").str.count(";") + 1
    df["track_name_len"] = df["track_name"].fillna("").str.len()
    df["album_name_len"] = df["album_name"].fillna("").str.len()
    df["energy_dance"]   = df["energy"] * df["danceability"]
    df["loud_energy"]    = df["loudness"] * df["energy"]
    drop_cols = ["track_id", "artists", "album_name", "track_name"]
    return df.drop(columns=drop_cols, errors="ignore")


def train():
    print("Descargando datos de entrenamiento...")
    train_df = pd.read_csv(TRAIN_URL)
    train_df = train_df.drop(columns=["Unnamed: 0"], errors="ignore")

    print("Aplicando feature engineering...")
    train_proc = feature_engineering(train_df)

    X = train_proc.drop(columns=["popularity"])
    y = train_proc["popularity"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    model = LGBMRegressor(
        n_estimators     = 500,
        learning_rate    = 0.05,
        max_depth        = 6,
        num_leaves       = 50,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_samples= 20,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        random_state     = 42,
        n_jobs           = -1,
        verbose          = -1,
    )

    pipeline = Pipeline([("prep", preprocessor), ("model", model)])

    print("Entrenando modelo...")
    pipeline.fit(X, y)

    joblib.dump(pipeline, MODEL_PATH)
    size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
    print(f"Modelo guardado en '{MODEL_PATH}' ({size_mb:.2f} MB)")


if __name__ == "__main__":
    train()
