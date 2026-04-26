# API de Predicción de Popularidad de Canciones — Spotify

API REST desarrollada con **FastAPI** para predecir la popularidad (0–100) de canciones de Spotify, como parte del **Proyecto 1 de Machine Learning y NLP — MIAD Uniandes**.

---

## API en Producción

| Recurso | URL |
|---------|-----|
| **API base** | https://proyecto-api-andes.onrender.com |
| **Documentación interactiva (Swagger)** | https://proyecto-api-andes.onrender.com/docs |
| **Estado del servicio** | https://proyecto-api-andes.onrender.com/health |
| **Predicción individual** | https://proyecto-api-andes.onrender.com/docs#/Predicción/predict_predict_post |
| **Predicción por lotes** | https://proyecto-api-andes.onrender.com/docs#/Predicci%C3%B3n/predict_batch_predict_batch_post |

> **Nota:** El plan gratuito de Render entra en reposo tras 15 minutos sin actividad. La primera petición puede tardar ~30 segundos. Simplemente espera y vuelve a intentar.

---

## Cómo hacer predicciones desde el navegador

### Paso 1 — Abrir la documentación interactiva

Ingresa a: **https://proyecto-api-andes.onrender.com/docs**

Verás la interfaz Swagger con todos los endpoints disponibles.

---

### Paso 2 — Verificar que el modelo está cargado (`GET /health`)

1. Clic en **`GET /health`**
2. Clic en **"Try it out"**
3. Clic en **"Execute"**
4. Debes ver en la respuesta:
```json
{
  "status": "ok",
  "modelo_cargado": true
}
```

---

### Paso 3 — Predicción de una canción (`POST /predict`)

1. Clic en **`POST /predict`**
2. Clic en **"Try it out"**
3. El cuerpo viene pre-llenado con un ejemplo. Puedes usarlo tal cual o modificarlo
4. Clic en **"Execute"**
5. La respuesta tendrá la forma:
```json
{
  "track_name": "No Other Name",
  "artists": "Hillsong Worship",
  "track_genre": "world-music",
  "popularity_prediction": 42.18,
  "escala": "0 (menos popular) – 100 (más popular)"
}
```

---

### Paso 4 — Predicción de dos canciones (`POST /predict_batch`)

1. Clic en **`POST /predict_batch`**
2. Clic en **"Try it out"**
3. Reemplaza el contenido del body con el siguiente JSON (observaciones reales del set de test):

```json
[
  {
    "artists": "Hillsong Worship",
    "album_name": "No Other Name",
    "track_name": "No Other Name",
    "duration_ms": 440247,
    "explicit": false,
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
  },
  {
    "artists": "Bryan Adams",
    "album_name": "All I Want For Christmas Is You",
    "track_name": "Merry Christmas",
    "duration_ms": 151387,
    "explicit": false,
    "danceability": 0.683,
    "energy": 0.511,
    "key": 6,
    "loudness": -5.598,
    "mode": 1,
    "speechiness": 0.0279,
    "acousticness": 0.406,
    "instrumentalness": 0.000197,
    "liveness": 0.111,
    "valence": 0.598,
    "tempo": 109.991,
    "time_signature": 3,
    "track_genre": "rock"
  }
]
```

4. Clic en **"Execute"**
5. La respuesta tendrá la forma:
```json
{
  "predicciones": [
    {
      "track_name": "No Other Name",
      "artists": "Hillsong Worship",
      "track_genre": "world-music",
      "popularity_prediction": 42.18
    },
    {
      "track_name": "Merry Christmas",
      "artists": "Bryan Adams",
      "track_genre": "rock",
      "popularity_prediction": 55.73
    }
  ],
  "total": 2
}
```

---

## Descripción

El servicio expone un modelo de Machine Learning entrenado sobre el [dataset de Spotify Tracks](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) que abarca **125 géneros musicales** y más de **79 000 canciones**. A partir de características de audio (energía, bailabilidad, tempo, etc.) el modelo predice qué tan popular es una canción en una escala de 0 a 100.

---

## Arquitectura

```
Cliente (curl / Postman / Python)
        │
        ▼
  FastAPI  (app.py)
        │
        ▼
  Feature Engineering
  (n_artists, energy_dance, loud_energy, …)
        │
        ▼
  Pipeline scikit-learn
  ├── ColumnTransformer
  │   ├── SimpleImputer  (numéricas)
  │   └── OneHotEncoder  (categóricas)
  └── LGBMRegressor  (500 estimators)
        │
        ▼
  Popularidad predicha  [0 – 100]
```

---

## Modelo

| Parámetro | Valor |
|-----------|-------|
| Algoritmo | LightGBM Regressor |
| n_estimators | 500 |
| learning_rate | 0.05 |
| max_depth | 6 |
| num_leaves | 50 |
| Métrica de evaluación | RMSE (5-Fold CV) |
| Preprocesamiento | SimpleImputer + OneHotEncoder |

**Features de entrada (19 variables originales + 5 ingeniadas):**

| Feature | Tipo | Descripción |
|---------|------|-------------|
| `duration_ms` | float | Duración de la pista en milisegundos |
| `explicit` | bool | Si tiene contenido explícito |
| `danceability` | float 0–1 | Cuán bailable es |
| `energy` | float 0–1 | Intensidad perceptual |
| `key` | int -1–11 | Tonalidad (-1 = no detectada) |
| `loudness` | float | Sonoridad en dB |
| `mode` | int 0–1 | Modalidad: 1=mayor, 0=menor |
| `speechiness` | float 0–1 | Presencia de habla |
| `acousticness` | float 0–1 | Confianza de que es acústica |
| `instrumentalness` | float 0–1 | Probabilidad de que sea instrumental |
| `liveness` | float 0–1 | Presencia de audiencia en vivo |
| `valence` | float 0–1 | Positividad musical |
| `tempo` | float | Tempo en BPM |
| `time_signature` | int 3–7 | Firma de tiempo |
| `track_genre` | string | Género musical |
| `n_artists` | *Ingeniada* | Número de artistas |
| `track_name_len` | *Ingeniada* | Longitud del nombre de la pista |
| `album_name_len` | *Ingeniada* | Longitud del nombre del álbum |
| `energy_dance` | *Ingeniada* | energy × danceability |
| `loud_energy` | *Ingeniada* | loudness × energy |

---

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Información general de la API |
| `GET` | `/health` | Estado del servicio y del modelo |
| `POST` | `/predict` | Predicción para **una** canción |
| `POST` | `/predict_batch` | Predicción para **varias** canciones |
| `GET` | `/docs` | Documentación interactiva (Swagger UI) |

---

## Estructura del Repositorio

```
Proyecto_Api_Andes/
├── app.py              # API REST (FastAPI)
├── train.py            # Script de entrenamiento del modelo
├── requirements.txt    # Dependencias Python
├── render.yaml         # Configuración de despliegue en Render.com
└── README.md           # Este archivo
```

---

## Ejecución Local

### 1. Clonar el repositorio

```bash
git clone https://github.com/MiguelData2030/Proyecto_Api_Andes.git
cd Proyecto_Api_Andes
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Entrenar y guardar el modelo

```bash
python train.py
```

Esto descarga los datos desde GitHub, entrena el pipeline y genera `model_pipeline.pkl`.

### 4. Iniciar la API

```bash
uvicorn app:app --reload
```

La API queda disponible en `http://localhost:8000`.  
Documentación interactiva en `http://localhost:8000/docs`.

---

## Despliegue en Render.com

El archivo `render.yaml` automatiza el despliegue completo:

- **Build:** instala dependencias y entrena el modelo (`python train.py`)
- **Start:** lanza el servidor FastAPI con Uvicorn

### Pasos

1. Crear cuenta gratuita en [render.com](https://render.com)
2. **New → Web Service** → conectar este repositorio de GitHub
3. Render detecta `render.yaml` automáticamente
4. Hacer clic en **Deploy**

---

## Tecnologías

| Tecnología | Versión mínima | Uso |
|------------|----------------|-----|
| Python | 3.11 | Lenguaje base |
| FastAPI | 0.100 | Framework de la API REST |
| Uvicorn | 0.23 | Servidor ASGI |
| LightGBM | 4.0 | Algoritmo de predicción |
| scikit-learn | 1.3 | Pipeline de preprocesamiento |
| pandas | 2.0 | Manipulación de datos |
| numpy | 1.24 | Operaciones numéricas |
| joblib | 1.3 | Serialización del modelo |
| Render.com | — | Despliegue en la nube |

---

## Equipo

**Proyecto 1 — Machine Learning y Procesamiento de Lenguaje Natural**  
Maestría en Inteligencia Artificial y Datos — Uniandes (MIAD) · 2026

| Integrante | Rol |
|------------|-----|
| Winston & Adolfo | Preprocesamiento, Modelación y Calibración |
| **Miguel** | **Disponibilización (API)** |
| Gisell & Miguel | Documentación e Informe |

---

## Dataset

- **Fuente:** [Spotify Tracks Dataset — Hugging Face](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
- **Repositorio del curso:** [MIAD ML NLP 2026 — GitHub](https://github.com/davidzarruk/MIAD_ML_NLP_2026)
- **Registros de entrenamiento:** 79 800
- **Géneros musicales:** 125
- **Variable objetivo:** `popularity` (0–100)
