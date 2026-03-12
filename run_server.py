# run_server.py
"""Lumora Intelligence — Uvicorn server başlatıcı."""
import logging
import sys
import os

# Proje kökünü path'e ekle (engine, algorithms, vb. importlar için)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.INTELLIGENCE_HOST,
        port=config.INTELLIGENCE_PORT,
        reload=(config.APP_ENV == "development"),
        log_level="info",
    )
