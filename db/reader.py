# db/reader.py
"""
Lumora Intelligence — DB okuma katmanı.
lumora_db'den daily_metrics ve products verilerini Pandas DataFrame olarak çeker.
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import text

from db.connection import engine

logger = logging.getLogger(__name__)


def get_daily_metrics(
    category: str | None = None,
    days: int = 90,
    product_ids: list[int] | None = None,
) -> pd.DataFrame:
    """
    daily_metrics tablosundan veri çeker.

    Args:
        category: Filtre uygulanacak search_term (None → tümü)
        days: Kaç günlük veri (bugünden geriye doğru)
        product_ids: Belirli ürün id'leri filtresi

    Returns:
        DataFrame: engine.predictor.train() uyumlu format
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    conditions = ["dm.recorded_at >= :cutoff"]
    params: dict = {"cutoff": cutoff}

    if category:
        conditions.append("dm.search_term = :category")
        params["category"] = category

    if product_ids:
        conditions.append("dm.product_id = ANY(:pids)")
        params["pids"] = product_ids

    where_clause = " AND ".join(conditions)

    sql = text(f"""
        SELECT
            dm.id,
            dm.product_id,
            p.name                  AS product_name,
            p.brand,
            p.category              AS product_category,
            dm.search_term          AS category,
            dm.recorded_at,
            dm.price,
            dm.discounted_price,
            dm.discount_rate,
            dm.cart_count,
            dm.favorite_count,
            dm.view_count,
            dm.rating_count,
            dm.avg_rating,
            dm.qa_count,
            dm.search_rank,
            dm.page_number,
            dm.absolute_rank,
            dm.engagement_score,
            dm.popularity_score,
            dm.sales_velocity,
            dm.demand_acceleration,
            dm.trend_direction,
            dm.velocity_score
        FROM daily_metrics dm
        JOIN products p ON p.id = dm.product_id
        WHERE {where_clause}
        ORDER BY dm.product_id, dm.recorded_at
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn, params=params)
        logger.info(f"DB'den {len(df)} daily_metrics satırı çekildi (category={category}, days={days})")
        return df
    except Exception as e:
        logger.error(f"daily_metrics okuma hatası: {e}")
        return pd.DataFrame()


def get_products(
    category: str | None = None,
    with_trend_score: bool = False,
) -> pd.DataFrame:
    """
    products tablosundan ürün bilgilerini çeker.

    Args:
        category: Filtre uygulanacak category_tag (None → tümü)
        with_trend_score: Sadece trend_score dolu olanları getir
    """
    conditions = []
    params: dict = {}

    if category:
        conditions.append("category_tag = :category")
        params["category"] = category

    if with_trend_score:
        conditions.append("trend_score IS NOT NULL")

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = text(f"""
        SELECT
            id,
            product_code,
            name,
            brand,
            seller,
            category,
            category_tag,
            last_price,
            last_discount_rate,
            last_engagement_score,
            avg_sales_velocity,
            trend_score,
            trend_direction,
            last_scored_at,
            attributes
        FROM products
        {where_clause}
        ORDER BY id
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn, params=params)
        logger.info(f"DB'den {len(df)} ürün çekildi (category={category})")
        return df
    except Exception as e:
        logger.error(f"products okuma hatası: {e}")
        return pd.DataFrame()


def get_categories() -> list[str]:
    """DB'deki aktif kategorileri (search_term) listeler."""
    sql = text("""
        SELECT DISTINCT search_term
        FROM daily_metrics
        WHERE search_term IS NOT NULL
          AND recorded_at >= NOW() - INTERVAL '30 days'
        ORDER BY search_term
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(sql)
            return [row[0] for row in result]
    except Exception as e:
        logger.error(f"Kategori listesi hatası: {e}")
        return []


def get_data_summary() -> dict:
    """Veri durumu özeti döndürür (health check için)."""
    sql = text("""
        SELECT
            COUNT(DISTINCT product_id)                        AS product_count,
            COUNT(*)                                          AS metric_count,
            MIN(recorded_at)::date                            AS oldest_date,
            MAX(recorded_at)::date                            AS newest_date,
            EXTRACT(DAY FROM NOW() - MIN(recorded_at))::int  AS data_days
        FROM daily_metrics
    """)
    try:
        with engine.connect() as conn:
            row = conn.execute(sql).fetchone()
            if row:
                return {
                    "product_count": row[0],
                    "metric_count":  row[1],
                    "oldest_date":   str(row[2]),
                    "newest_date":   str(row[3]),
                    "data_days":     row[4] or 0,
                }
    except Exception as e:
        logger.error(f"Veri özeti hatası: {e}")
    return {}
