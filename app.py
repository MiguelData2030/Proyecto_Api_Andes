"""
API REST para predecir la popularidad de canciones de Spotify.
Carga el pipeline entrenado (model_pipeline.pkl) y expone endpoints
de predicción individual y por lotes.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Carga del modelo al iniciar la aplicación
# ---------------------------------------------------------------------------
MODEL_PATH = "model_pipeline.pkl"
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Modelo cargado correctamente desde '{MODEL_PATH}'")
    else:
        print(f"ADVERTENCIA: No se encontró '{MODEL_PATH}'. "
              "Ejecuta primero el notebook para generar el archivo.")
    yield
    model = None


# ---------------------------------------------------------------------------
# Instancia FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API Predicción Popularidad - Spotify",
    description=(
        "Predice la popularidad (0–100) de canciones de Spotify "
        "usando un modelo LightGBM entrenado sobre el dataset de Spotify tracks."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Esquema de entrada
# ---------------------------------------------------------------------------
class SongFeatures(BaseModel):
    track_id:         Optional[str]   = Field(default="unknown",   description="ID de la pista (opcional)")
    artists:          str             = Field(default="Unknown",    description="Nombre(s) del artista; varios separados por ';'")
    album_name:       str             = Field(default="Unknown",    description="Nombre del álbum")
    track_name:       str             = Field(default="Unknown",    description="Nombre de la pista")
    duration_ms:      float           = Field(default=200000,       description="Duración en milisegundos")
    explicit:         bool            = Field(default=False,        description="True si tiene contenido explícito")
    danceability:     float           = Field(default=0.5,  ge=0, le=1, description="Cuán bailable es (0–1)")
    energy:           float           = Field(default=0.5,  ge=0, le=1, description="Energía perceptual (0–1)")
    key:              int             = Field(default=0,    ge=-1, le=11, description="Tonalidad (-1 = no detectada)")
    loudness:         float           = Field(default=-10.0,        description="Sonoridad en dB")
    mode:             int             = Field(default=1,    ge=0, le=1,  description="Modalidad: 1=mayor, 0=menor")
    speechiness:      float           = Field(default=0.05, ge=0, le=1, description="Presencia de habla (0–1)")
    acousticness:     float           = Field(default=0.5,  ge=0, le=1, description="Confianza de que es acústica (0–1)")
    instrumentalness: float           = Field(default=0.0,  ge=0, le=1, description="Probabilidad de que sea instrumental (0–1)")
    liveness:         float           = Field(default=0.1,  ge=0, le=1, description="Presencia de audiencia en vivo (0–1)")
    valence:          float           = Field(default=0.5,  ge=0, le=1, description="Positividad musical (0–1)")
    tempo:            float           = Field(default=120.0,        description="Tempo en BPM")
    time_signature:   int             = Field(default=4,    ge=3, le=7,  description="Firma de tiempo (3–7)")
    track_genre:      str             = Field(default="pop",        description="Género musical")

    class Config:
        json_schema_extra = {
            "example": {
                "track_id": "6KwkVtXm8OUp2XffN5k7lY",
                "artists": "Hillsong Worship",
                "album_name": "No Other Name",
                "track_name": "No Other Name",
                "duration_ms": 440247,
                "explicit": False,
                "danceability": 0.369,
                "energy": 0.598,
                "key": 7,
                "loudness": -6.984,
                "mode": 1,
                "speechiness": 0.0304,
                "acousticness": 0.00511,
                "instrumentalness": 0.0,
                "liveness": 0.176,
                "valence": 0.0466,
                "tempo": 148.014,
                "time_signature": 4,
                "track_genre": "world-music"
            }
        }


# ---------------------------------------------------------------------------
# Feature engineering (idéntico al notebook)
# ---------------------------------------------------------------------------
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["explicit"]       = df["explicit"].astype(int)   # bool → int (compatibilidad sklearn)
    df["n_artists"]      = df["artists"].fillna("").str.count(";") + 1
    df["track_name_len"] = df["track_name"].fillna("").str.len()
    df["album_name_len"] = df["album_name"].fillna("").str.len()
    df["energy_dance"]   = df["energy"] * df["danceability"]
    df["loud_energy"]    = df["loudness"] * df["energy"]
    drop_cols = ["track_id", "artists", "album_name", "track_name"]
    return df.drop(columns=drop_cols, errors="ignore")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Info"])
def root():
    return {
        "api": "Predicción de Popularidad de Canciones - Spotify",
        "version": "1.0.0",
        "estado": "activo",
        "modelo_cargado": model is not None,
        "endpoints": {
            "GET  /":               "Información de la API",
            "GET  /health":         "Estado del servicio",
            "POST /predict":        "Predicción para una canción",
            "POST /predict_batch":  "Predicción para varias canciones",
        },
        "documentacion_interactiva": "/docs",
    }


@app.get("/health", tags=["Info"])
def health():
    return {
        "status": "ok" if model is not None else "modelo no disponible",
        "modelo_cargado": model is not None,
    }


@app.post("/predict", tags=["Predicción"])
def predict(song: SongFeatures):
    """
    Recibe las características de una canción y devuelve la popularidad predicha (0–100).
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no disponible. Verifica que '{MODEL_PATH}' existe en el servidor.",
        )
    try:
        df = pd.DataFrame([song.dict()])
        df_proc = apply_feature_engineering(df)
        raw_pred = model.predict(df_proc)
        popularity = float(np.clip(raw_pred[0], 0, 100))
        return {
            "track_name":            song.track_name,
            "artists":               song.artists,
            "track_genre":           song.track_genre,
            "popularity_prediction": round(popularity, 2),
            "escala":                "0 (menos popular) – 100 (más popular)",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {exc}")


@app.post("/predict_batch", tags=["Predicción"])
def predict_batch(songs: List[SongFeatures]):
    """
    Recibe una lista de canciones y devuelve la popularidad predicha para cada una.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no disponible. Verifica que '{MODEL_PATH}' existe en el servidor.",
        )
    try:
        results = []
        for song in songs:
            df = pd.DataFrame([song.dict()])
            df_proc = apply_feature_engineering(df)
            raw_pred = model.predict(df_proc)
            popularity = float(np.clip(raw_pred[0], 0, 100))
            results.append({
                "track_name":            song.track_name,
                "artists":               song.artists,
                "track_genre":           song.track_genre,
                "popularity_prediction": round(popularity, 2),
            })
        return {"predicciones": results, "total": len(results)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {exc}")
