# db/backfill_attributes.py
"""
One-shot script: products.attributes JSONB alanından
dominant_color, fabric_type, fit_type kolonlarını backfill eder.

Kullanım:
    python db/backfill_attributes.py --dry-run   # Kaç ürün etkilenecek
    python db/backfill_attributes.py              # Gerçek güncelleme
"""
import argparse
import json
import logging
import sys

from sqlalchemy import text
from db.connection import engine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# JSONB attributes'tan hangi key → hangi kolon eşlemesi
FIELD_MAP = {
    "dominant_color": [
        "Renk", "renk", "color", "Color", "Ana Renk",
    ],
    "fabric_type": [
        "Kumaş Tipi", "Kumaş", "kumaş", "fabric", "Fabric",
        "Materyal", "materyal", "Material",
    ],
    "fit_type": [
        "Kalıp", "kalıp", "Fit", "fit", "Beden Tipi",
        "Kesim", "kesim", "Siluet",
    ],
}


def find_value(attrs: dict, keys: list) -> str | None:
    """JSONB'den ilk bulunan key'in değerini döndürür."""
    if not attrs:
        return None
    for key in keys:
        if key in attrs:
            val = attrs[key]
            if isinstance(val, str) and val.strip():
                return val.strip()
            elif isinstance(val, list) and val:
                return str(val[0]).strip()
    return None


def run_backfill(dry_run: bool = True):
    """Tüm products'ları tarayıp JSONB'den kolon günceller."""

    # 1. attributes dolu ama dominant_color/fabric_type/fit_type boş olanları bul
    select_sql = text("""
        SELECT id, attributes
        FROM products
        WHERE attributes IS NOT NULL
          AND attributes != '{}'::jsonb
          AND (
              dominant_color IS NULL
              OR fabric_type IS NULL
              OR fit_type IS NULL
          )
    """)

    update_sql = text("""
        UPDATE products
        SET dominant_color = COALESCE(:dominant_color, dominant_color),
            fabric_type    = COALESCE(:fabric_type, fabric_type),
            fit_type       = COALESCE(:fit_type, fit_type)
        WHERE id = :pid
    """)

    with engine.connect() as conn:
        rows = conn.execute(select_sql).fetchall()
        logger.info(f"🔍 {len(rows)} ürün JSONB backfill'e uygun")

        if dry_run:
            # Dry-run: sadece ilk 10'u göster
            sample_updates = 0
            for row in rows[:10]:
                pid, attrs = row[0], row[1]
                if isinstance(attrs, str):
                    attrs = json.loads(attrs)
                color = find_value(attrs, FIELD_MAP["dominant_color"])
                fabric = find_value(attrs, FIELD_MAP["fabric_type"])
                fit = find_value(attrs, FIELD_MAP["fit_type"])
                if color or fabric or fit:
                    sample_updates += 1
                    logger.info(
                        f"  #{pid}: color={color}, fabric={fabric}, fit={fit}"
                    )
            logger.info(f"\n  {sample_updates}/10 örnekte değer bulundu")
            logger.info("  '--dry-run' kaldırarak gerçek güncelleme yapabilirsin")
            return

        # Gerçek güncelleme
        updated = 0
        with engine.begin() as tx_conn:
            for row in rows:
                pid, attrs = row[0], row[1]
                if isinstance(attrs, str):
                    attrs = json.loads(attrs)

                color = find_value(attrs, FIELD_MAP["dominant_color"])
                fabric = find_value(attrs, FIELD_MAP["fabric_type"])
                fit = find_value(attrs, FIELD_MAP["fit_type"])

                if color or fabric or fit:
                    tx_conn.execute(update_sql, {
                        "pid": pid,
                        "dominant_color": color,
                        "fabric_type": fabric,
                        "fit_type": fit,
                    })
                    updated += 1

        logger.info(f"✅ {updated}/{len(rows)} ürün güncellendi")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSONB → kolon backfill")
    parser.add_argument("--dry-run", action="store_true", help="Gerçek güncelleme yapma")
    args = parser.parse_args()
    run_backfill(dry_run=args.dry_run)
