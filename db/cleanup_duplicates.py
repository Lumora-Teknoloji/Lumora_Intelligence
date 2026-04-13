# db/cleanup_duplicates.py
"""
Lumora Intelligence — Daily_metrics duplicate temizleme scripti.

Aynı product_id + aynı gün + aynı search_term için birden fazla kayıt varsa,
en son kaydı tutar ve diğerlerini siler.

Kullanım:
  python db/cleanup_duplicates.py             # Gerçek silme
  python db/cleanup_duplicates.py --dry-run   # Sadece count göster
"""
import sys
import logging
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Proje kök dizininden import edebilmek için
sys.path.insert(0, ".")
from db.connection import engine


def count_duplicates() -> int:
    """Duplicate kayıt sayısını döndürür."""
    sql = text("""
        SELECT COUNT(*) FROM (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY product_id, DATE(recorded_at), search_term
                       ORDER BY recorded_at DESC
                   ) AS rn
            FROM daily_metrics
        ) sub
        WHERE rn > 1
    """)
    with engine.connect() as conn:
        return conn.execute(sql).scalar() or 0


def delete_duplicates() -> int:
    """
    Duplicate kayıtları siler.
    Aynı (product_id, DATE(recorded_at), search_term) için en son kaydı tutar.

    Returns:
        Silinen kayıt sayısı
    """
    sql = text("""
        DELETE FROM daily_metrics
        WHERE id IN (
            SELECT id FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY product_id, DATE(recorded_at), search_term
                           ORDER BY recorded_at DESC
                       ) AS rn
                FROM daily_metrics
            ) sub
            WHERE rn > 1
        )
    """)
    with engine.begin() as conn:
        result = conn.execute(sql)
        return result.rowcount


def main():
    dry_run = "--dry-run" in sys.argv

    count = count_duplicates()
    logger.info(f"🔍 Tespit edilen duplicate kayıt: {count}")

    if count == 0:
        logger.info("✅ Duplicate kayıt yok — temizleme gerekmiyor")
        return

    if dry_run:
        logger.info("🏷️  --dry-run modu: silme yapılmadı")
        return

    deleted = delete_duplicates()
    logger.info(f"🗑️  {deleted} duplicate kayıt silindi")

    # Kontrol
    remaining = count_duplicates()
    if remaining == 0:
        logger.info("✅ Tüm duplicate'lar temizlendi")
    else:
        logger.warning(f"⚠️  Hala {remaining} duplicate var (iç içe kayıtlar olabilir)")


if __name__ == "__main__":
    main()
