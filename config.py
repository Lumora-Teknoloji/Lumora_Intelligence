# config.py
"""Proje konfigürasyonu."""

# Z-Score eşikleri
ZSCORE_ERROR_THRESHOLD = 3.0      # |Z| > 3 → hatalı veri
ZSCORE_ANOMALY_THRESHOLD = 2.5    # |Z| > 2.5 → anomali (viral/düşüş)

# CatBoost
CATBOOST_ITERATIONS = 500
CATBOOST_LEARNING_RATE = 0.05
CATBOOST_DEPTH = 6
CATBOOST_FORECAST_DAYS = 90

# Kalman Filter
KALMAN_PROCESS_NOISE = 0.1        # Süreç gürültüsü
KALMAN_MEASUREMENT_NOISE = 1.0    # Ölçüm gürültüsü

# Prophet
PROPHET_CHANGEPOINT_SCALE = 0.05
PROPHET_SEASONALITY_MODE = "multiplicative"

# Change Point Detection
CPD_PENALTY = "bic"               # Bayesian Information Criterion
CPD_MIN_SIZE = 7                  # Minimum segment uzunluğu (gün)

# K-Prototypes
N_CLUSTERS = 4                    # Yıldız, Nakit İnek, Potansiyel, Düşen
MAX_ITER = 100

# CLIP
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# Ensemble ağırlıkları
ENSEMBLE_CATBOOST_WEIGHT = 0.6
ENSEMBLE_KALMAN_WEIGHT = 0.4

# Bayesian Optimization
OPTUNA_N_TRIALS = 50
