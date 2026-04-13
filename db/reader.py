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


# ─── Category Auto-Classification ─────────────────────────────────────────────

def get_category_stats() -> list[dict]:
    """
    Her kategori için otomatik sınıflandırma istatistiklerini döndürür.
    CategoryAutoClassifier bu veriyi kullanarak profil tipini belirler.

    Returns:
        [{category, n_products, n_days, avg_price, price_cv, engagement_cv}, ...]
    """
    sql = text("""
        SELECT
            p.category,
            COUNT(DISTINCT dm.product_id)                             AS n_products,
            EXTRACT(DAY FROM MAX(dm.recorded_at) - MIN(dm.recorded_at))::int AS n_days,
            ROUND(AVG(dm.price)::numeric, 2)                          AS avg_price,
            CASE WHEN AVG(dm.price) > 0
                 THEN ROUND((STDDEV(dm.price) / AVG(dm.price))::numeric, 3)
                 ELSE 0 END                                           AS price_cv,
            CASE WHEN AVG(COALESCE(dm.engagement_score, 0)) > 0
                 THEN ROUND((STDDEV(COALESCE(dm.engagement_score, 0))
                        / AVG(COALESCE(dm.engagement_score, 0)))::numeric, 3)
                 ELSE 0 END                                           AS engagement_cv
        FROM daily_metrics dm
        JOIN products p ON p.id = dm.product_id
        WHERE p.category IS NOT NULL
        GROUP BY p.category
        ORDER BY n_products DESC
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql).fetchall()
            results = []
            for row in rows:
                results.append({
                    "category":      row[0],
                    "n_products":    row[1] or 0,
                    "n_days":        row[2] or 0,
                    "avg_price":     float(row[3]) if row[3] else 0.0,
                    "price_cv":      float(row[4]) if row[4] else 0.0,
                    "engagement_cv": float(row[5]) if row[5] else 0.0,
                })
            logger.info(f"📊 {len(results)} kategori istatistiği çekildi")
            return results
    except Exception as e:
        logger.error(f"Kategori istatistikleri hatası: {e}")
        return []


def get_category_profiles() -> dict[str, dict]:
    """
    categories_registry tablosundan kayıtlı profil bilgilerini döndürür.
    ModelRegistry DB-first profile lookup için kullanır.

    Returns:
        {category_name: {profile_type, lifecycle, data_days, total_products, group_name, overrides}, ...}
    """
    sql = text("""
        SELECT search_term, profile_type, lifecycle, data_days, total_products,
               group_name, overrides
        FROM categories_registry
        ORDER BY search_term
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql).fetchall()
            result = {}
            for row in rows:
                overrides = row[6] or {}
                if isinstance(overrides, str):
                    import json
                    try: overrides = json.loads(overrides)
                    except: overrides = {}
                result[row[0]] = {
                    "profile_type":   row[1],
                    "lifecycle":      row[2],
                    "data_days":      row[3] or 0,
                    "total_products": row[4] or 0,
                    "group_name":     row[5],
                    "overrides":      overrides,
                }
            logger.info(f"📋 {len(result)} kategori profili DB'den okundu")
            return result
    except Exception as e:
        logger.error(f"Kategori profilleri okuma hatası: {e}")
        return {}


# ─── Feedback Loop Okuma ──────────────────────────────────────────────────────

def get_production_decisions(
    category: str | None = None,
    product_id: int | None = None,
    limit: int = 100,
) -> list[dict]:
    """
    production_decisions tablosundan üretim kararlarını çeker.

    Args:
        category: search_term filtresi
        product_id: tek ürün filtresi
        limit: max kayıt sayısı
    """
    conditions = []
    params: dict = {"limit": limit}

    if category:
        conditions.append("pd.search_term = :category")
        params["category"] = category
    if product_id:
        conditions.append("pd.product_id = :pid")
        params["pid"] = product_id

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = text(f"""
        SELECT
            pd.id, pd.product_id, p.name AS product_name,
            pd.search_term, pd.predicted_score, pd.decision,
            pd.quantity, pd.notes, pd.decided_at
        FROM production_decisions pd
        LEFT JOIN products p ON p.id = pd.product_id
        {where_clause}
        ORDER BY pd.decided_at DESC
        LIMIT :limit
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                {
                    "id":              row[0],
                    "product_id":      row[1],
                    "product_name":    row[2],
                    "search_term":     row[3],
                    "predicted_score": float(row[4]) if row[4] else None,
                    "decision":        row[5],
                    "quantity":        row[6],
                    "notes":           row[7],
                    "decided_at":      str(row[8]),
                }
                for row in rows
            ]
    except Exception as e:
        logger.error(f"production_decisions okuma hatası: {e}")
        return []


def get_actual_sales(production_id: int | None = None, limit: int = 100) -> list[dict]:
    """
    actual_sales tablosundan gerçek satış sonuçlarını çeker.

    Args:
        production_id: belirli bir üretim kararına ait satışlar
        limit: max kayıt sayısı
    """
    conditions = []
    params: dict = {"limit": limit}

    if production_id:
        conditions.append("a.production_id = :prod_id")
        params["prod_id"] = production_id

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = text(f"""
        SELECT
            a.id, a.production_id,
            pd.product_id, p.name AS product_name,
            a.sold_quantity, a.produced_quantity, a.sell_through_rate,
            a.revenue, a.feedback_date, a.created_at
        FROM actual_sales a
        JOIN production_decisions pd ON pd.id = a.production_id
        LEFT JOIN products p ON p.id = pd.product_id
        {where_clause}
        ORDER BY a.feedback_date DESC
        LIMIT :limit
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                {
                    "id":                row[0],
                    "production_id":     row[1],
                    "product_id":        row[2],
                    "product_name":      row[3],
                    "sold_quantity":     row[4],
                    "produced_quantity": row[5],
                    "sell_through_rate": float(row[6]) if row[6] else None,
                    "revenue":          float(row[7]) if row[7] else None,
                    "feedback_date":    str(row[8]),
                    "created_at":       str(row[9]),
                }
                for row in rows
            ]
    except Exception as e:
        logger.error(f"actual_sales okuma hatası: {e}")
        return []


def get_feedback_summary() -> dict:
    """
    Feedback loop özetini döndürür — kaç karar, kaç satış, ortalama başarı.
    Health check ve dashboard için.
    """
    sql = text("""
        SELECT
            (SELECT COUNT(*) FROM production_decisions)              AS total_decisions,
            (SELECT COUNT(*) FROM actual_sales)                     AS total_sales,
            (SELECT ROUND(AVG(sell_through_rate)::numeric, 3)
             FROM actual_sales)                                     AS avg_sell_through,
            (SELECT COUNT(*) FROM actual_sales
             WHERE sell_through_rate >= 0.7)                        AS successful_sales
    """)
    try:
        with engine.connect() as conn:
            row = conn.execute(sql).fetchone()
            if row:
                total_sales = row[1] or 0
                successful = row[3] or 0
                return {
                    "total_decisions":   row[0] or 0,
                    "total_sales":       total_sales,
                    "avg_sell_through":  float(row[2]) if row[2] else None,
                    "successful_sales":  successful,
                    "success_rate":      round(successful / max(total_sales, 1), 3),
                    "ready_for_catboost": total_sales >= 30,
                }
    except Exception as e:
        logger.error(f"Feedback özeti hatası: {e}")
    return {"total_decisions": 0, "total_sales": 0, "ready_for_catboost": False}
