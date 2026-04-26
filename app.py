"""
API REST para predecir la popularidad de canciones de Spotify.
Modelo: CatBoost calibrado con Optuna
Métricas en validación: RMSE 8.73 | MAE 6.05 | R² 0.845
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH  = "model_pipeline.pkl"
model       = None
cat_features = None
predictores  = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, cat_features, predictores
    if os.path.exists(MODEL_PATH):
        artifact     = joblib.load(MODEL_PATH)
        model        = artifact["model"]
        cat_features = artifact["cat_features"]
        predictores  = artifact["predictores"]
        print(f"Modelo CatBoost cargado correctamente desde '{MODEL_PATH}'")
    else:
        print(f"ADVERTENCIA: No se encontró '{MODEL_PATH}'. Ejecuta train.py primero.")
    yield
    model = None


app = FastAPI(
    title="API Predicción Popularidad - Spotify",
    description=(
        "Predice la popularidad (0–100) de canciones de Spotify "
        "usando **CatBoost calibrado con Optuna** "
        "(RMSE: 8.73 | MAE: 6.05 | R²: 0.845 en validación)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Esquema de entrada
# ---------------------------------------------------------------------------
class SongFeatures(BaseModel):
    artists:          str   = Field(default="Unknown",  description="Nombre(s) del artista; varios separados por ';'")
    album_name:       str   = Field(default="Unknown",  description="Nombre del álbum")
    track_name:       str   = Field(default="Unknown",  description="Nombre de la pista")
    duration_ms:      float = Field(default=200000,     description="Duración en milisegundos")
    explicit:         bool  = Field(default=False,      description="True si tiene contenido explícito")
    danceability:     float = Field(default=0.5,  ge=0, le=1, description="Cuán bailable es (0–1)")
    energy:           float = Field(default=0.5,  ge=0, le=1, description="Energía perceptual (0–1)")
    key:              int   = Field(default=0,  ge=-1, le=11,  description="Tonalidad (-1 = no detectada)")
    loudness:         float = Field(default=-10.0,      description="Sonoridad en dB")
    mode:             int   = Field(default=1,  ge=0, le=1,    description="Modalidad: 1=mayor, 0=menor")
    speechiness:      float = Field(default=0.05, ge=0, le=1, description="Presencia de habla (0–1)")
    acousticness:     float = Field(default=0.5,  ge=0, le=1, description="Confianza de que es acústica (0–1)")
    instrumentalness: float = Field(default=0.0,  ge=0, le=1, description="Probabilidad de que sea instrumental (0–1)")
    liveness:         float = Field(default=0.1,  ge=0, le=1, description="Presencia de audiencia en vivo (0–1)")
    valence:          float = Field(default=0.5,  ge=0, le=1, description="Positividad musical (0–1)")
    tempo:            float = Field(default=120.0,      description="Tempo en BPM")
    time_signature:   int   = Field(default=4,  ge=0, le=7,   description="Firma de tiempo (compases por beat)")
    track_genre:      str   = Field(default="pop",      description="Género musical")

    class Config:
        json_schema_extra = {
            "example": {
                "artists":          "Hillsong Worship",
                "album_name":       "No Other Name",
                "track_name":       "No Other Name",
                "duration_ms":      440247,
                "explicit":         False,
                "danceability":     0.369,
                "energy":           0.598,
                "key":              7,
                "loudness":         -6.984,
                "mode":             1,
                "speechiness":      0.0304,
                "acousticness":     0.00511,
                "instrumentalness": 0.0,
                "liveness":         0.176,
                "valence":          0.0466,
                "tempo":            148.014,
                "time_signature":   4,
                "track_genre":      "world-music"
            }
        }


# ---------------------------------------------------------------------------
# Preprocesamiento (idéntico al notebook)
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in cat_features:
        df[col] = df[col].fillna("missing").astype(str)
    return df[predictores]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Info"])
def root():
    return {
        "api":                    "Predicción de Popularidad de Canciones - Spotify",
        "version":                "2.0.0",
        "modelo":                 "CatBoost + Optuna",
        "metricas_validacion":    {"RMSE": 8.73, "MAE": 6.05, "R2": 0.845},
        "estado":                 "activo",
        "modelo_cargado":         model is not None,
        "endpoints": {
            "GET  /":              "Información de la API",
            "GET  /health":        "Estado del servicio",
            "POST /predict":       "Predicción para una canción",
            "POST /predict_batch": "Predicción para varias canciones",
        },
        "documentacion_interactiva": "/docs",
    }


@app.get("/health", tags=["Info"])
def health():
    return {
        "status":         "ok" if model is not None else "modelo no disponible",
        "modelo_cargado": model is not None,
    }


@app.post("/predict", tags=["Predicción"])
def predict(song: SongFeatures):
    """
    Recibe las características de una canción y devuelve la popularidad predicha (0–100).
    Modelo: CatBoost + Optuna | RMSE: 8.73 | R²: 0.845
    """
    if model is None:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible. Verifica que '{MODEL_PATH}' existe.")
    try:
        df      = pd.DataFrame([song.dict()])
        df_proc = preprocess(df)
        pred    = float(np.clip(model.predict(df_proc)[0], 0, 100))
        return {
            "track_name":            song.track_name,
            "artists":               song.artists,
            "track_genre":           song.track_genre,
            "popularity_prediction": round(pred, 2),
            "escala":                "0 (menos popular) – 100 (más popular)",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {exc}")


@app.post("/predict_batch", tags=["Predicción"])
def predict_batch(songs: List[SongFeatures]):
    """
    Recibe una lista de canciones y devuelve la popularidad predicha para cada una.
    Modelo: CatBoost + Optuna | RMSE: 8.73 | R²: 0.845
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")
    try:
        results = []
        for song in songs:
            df      = pd.DataFrame([song.dict()])
            df_proc = preprocess(df)
            pred    = float(np.clip(model.predict(df_proc)[0], 0, 100))
            results.append({
                "track_name":            song.track_name,
                "artists":               song.artists,
                "track_genre":           song.track_genre,
                "popularity_prediction": round(pred, 2),
            })
        return {"predicciones": results, "total": len(results)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {exc}")
