# config.py
"""
Lumora Intelligence — Mikro Servis Konfigürasyonu
"""
import os
from pathlib import Path

# ─── Ortam değişkenleri (.env dosyasından okunur) ──────────────────────────
def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

# ─── Servis Ayarları ────────────────────────────────────────────────────────
INTELLIGENCE_HOST = _env("INTELLIGENCE_HOST", "0.0.0.0")
INTELLIGENCE_PORT = int(_env("INTELLIGENCE_PORT", "8001"))
APP_ENV = _env("APP_ENV", "development")

# İç servis güvenlik anahtarı (LangChain Backend → Intelligence iletişimi)
INTERNAL_API_KEY = _env("INTERNAL_API_KEY", "lumora-internal-dev-key")

# Backend URL — Intelligence'ın callback göndereceği adres
BACKEND_URL          = _env("BACKEND_URL", "http://localhost:8000")
BACKEND_CALLBACK_URL = _env(
    "BACKEND_CALLBACK_URL",
    f"{_env('BACKEND_URL', 'http://localhost:8000')}/api/intelligence/callback"
)

# ─── PostgreSQL Bağlantısı ───────────────────────────────────────────────────
POSTGRESQL_HOST     = _env("POSTGRESQL_HOST", "localhost")
POSTGRESQL_PORT     = int(_env("POSTGRESQL_PORT", "5432"))
POSTGRESQL_DATABASE = _env("POSTGRESQL_DATABASE", "lumora_db")
POSTGRESQL_USERNAME = _env("POSTGRESQL_USERNAME", "postgres")
POSTGRESQL_PASSWORD = _env("POSTGRESQL_PASSWORD", "postgres")

DATABASE_URL = (
    f"postgresql://{POSTGRESQL_USERNAME}:{POSTGRESQL_PASSWORD}"
    f"@{POSTGRESQL_HOST}:{POSTGRESQL_PORT}/{POSTGRESQL_DATABASE}"
)

# ─── APScheduler Cron Zamanları ─────────────────────────────────────────────
NIGHTLY_BATCH_HOUR   = int(_env("NIGHTLY_BATCH_HOUR", "2"))    # 02:00 her gece
NIGHTLY_BATCH_MINUTE = int(_env("NIGHTLY_BATCH_MINUTE", "0"))
WEEKLY_RETRAIN_DAY   = _env("WEEKLY_RETRAIN_DAY", "sun")       # Pazar
WEEKLY_RETRAIN_HOUR  = int(_env("WEEKLY_RETRAIN_HOUR", "3"))   # 03:00

# ─── Engine Parametreleri ────────────────────────────────────────────────────
# Z-Score eşikleri
ZSCORE_ERROR_THRESHOLD   = 3.0
ZSCORE_ANOMALY_THRESHOLD = 2.5

# CatBoost
CATBOOST_ITERATIONS    = 500
CATBOOST_LEARNING_RATE = 0.05
CATBOOST_DEPTH         = 6
CATBOOST_FORECAST_DAYS = 90

# Kalman Filter
KALMAN_PROCESS_NOISE     = 0.1
KALMAN_MEASUREMENT_NOISE = 1.0

# Prophet
PROPHET_CHANGEPOINT_SCALE  = 0.05
PROPHET_SEASONALITY_MODE   = "multiplicative"

# Change Point Detection
CPD_PENALTY  = "bic"
CPD_MIN_SIZE = 7

# K-Prototypes
N_CLUSTERS = 4
MAX_ITER   = 100

# CLIP
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# Ensemble ağırlıkları
ENSEMBLE_CATBOOST_WEIGHT = 0.6
ENSEMBLE_KALMAN_WEIGHT   = 0.4

# Bayesian Optimization
OPTUNA_N_TRIALS = 50

# ─── Minimum veri gereksinimleri ─────────────────────────────────────────────
MIN_DAYS_FOR_CATBOOST = 30   # CatBoost retraining için min gün
MIN_DAYS_FOR_PROPHET  = 60   # Prophet için min gün

# ─── Tavily API (Araştırma Katmanı) ─────────────────────────────────────────
TAVILY_API_KEY     = _env("TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS = int(_env("TAVILY_MAX_RESULTS", "5"))
TAVILY_SEARCH_DEPTH = _env("TAVILY_SEARCH_DEPTH", "advanced")

# ─── Per-Category Profiller ─────────────────────────────────────────────────
# Her kategori tipi kendi algorithm parametrelerini taşır.
# CategoryAutoClassifier veri özelliklerine göre her kategoriyi bir profile atar.
CATEGORY_PROFILES = {
    "sparse": {
        "zscore_threshold":  None,    # Z-Score kullanılmaz — çok az veri
        "catboost_depth":    None,    # CatBoost kullanılmaz
        "catboost_iterations": None,
        "min_products":      0,       # Herhangi bir ürün sayısıyla çalışır
        "use_catboost":      False,   # Basit ortalama/median bazlı skor
        "score_weights": {"growth": 0.50, "velocity": 0.30, "demand": 0.20},
        "label_thresholds": {"trend": 60, "potansiyel": 35, "stabil": 15},
    },
    "premium": {
        "zscore_threshold":  2.0,     # Hassas — az ürünlü kategorilerde anomali erken yakalanmalı
        "catboost_depth":    4,       # Shallow tree → overfitting önle
        "catboost_iterations": 300,
        "min_products":      5,
        "use_catboost":      True,
        "score_weights": {"growth": 0.30, "velocity": 0.40, "demand": 0.30},
        "label_thresholds": {"trend": 65, "potansiyel": 35, "stabil": 15},
    },
    "mid_fashion": {
        "zscore_threshold":  3.0,
        "catboost_depth":    6,
        "catboost_iterations": CATBOOST_ITERATIONS,
        "min_products":      15,
        "use_catboost":      True,
        "score_weights": {"growth": 0.40, "velocity": 0.35, "demand": 0.25},
        "label_thresholds": {"trend": 70, "potansiyel": 40, "stabil": 20},
    },
    "heterogeneous": {
        "zscore_threshold":  5.0,     # Geniş tolerans — karışık ürün yelpazesi
        "catboost_depth":    8,       # Derin ağaç — karmaşık ilişkiler
        "catboost_iterations": CATBOOST_ITERATIONS,
        "min_products":      30,
        "use_catboost":      True,
        "score_weights": {"growth": 0.35, "velocity": 0.35, "demand": 0.30},
        "label_thresholds": {"trend": 75, "potansiyel": 45, "stabil": 25},
    },
}

# ─── Otomatik Sınıflandırma Kuralları ───────────────────────────────────────
# CategoryAutoClassifier bu kuralları kullanarak her kategoriyi bir profile atar.
AUTO_CLASSIFY_RULES = {
    "sparse_max_products":       5,     # < 5 ürün → sparse
    "premium_max_products":      15,    # < 15 ürün ve avg_price > threshold → premium
    "premium_min_avg_price":     300.0, # Ortalama fiyat eşiği (₺)
    "mid_fashion_max_products":  80,    # 15-80 ürün → mid_fashion
    "mid_fashion_max_price_cv":  0.5,   # Fiyat varyasyon katsayısı eşiği
    "heterogeneous_min_products": 80,   # 80+ ürün → heterogeneous
    "heterogeneous_min_price_cv": 0.7,  # VEYA price_cv > 0.7 → heterogeneous
}

# Bilinen kategorilerin profil eşlemesi (manuel override — auto-classify'dan önce gelir)
CATEGORY_TAG_MAP = {
    "crop":        "mid_fashion",
    "tayt":        "mid_fashion",
    "kadın abiye": "premium",
    "grup":        "heterogeneous",
}

# Bilinmeyen kategoriler için varsayılan profil
DEFAULT_CATEGORY_PROFILE = "mid_fashion"
