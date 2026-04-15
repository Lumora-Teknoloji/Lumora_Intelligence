# db/writer.py
"""
Lumora Intelligence — DB yazma katmanı.
intelligence_results, trend_alerts, products.trend_score güncelleme.
"""
import logging
from datetime import datetime, date
from sqlalchemy import text

from db.connection import engine

logger = logging.getLogger(__name__)


def save_predictions(predictions: list[dict]) -> int:
    """
    intelligence_results tablosuna tahmin sonuçlarını yazar (upsert).

    Args:
        predictions: [{product_id, trend_label, trend_score, confidence, category, ...}]

    Returns:
        Kaç satır yazıldı
    """
    if not predictions:
        return 0

    # Tablo yoksa oluştur
    _ensure_tables()

    upsert_sql = text("""
        INSERT INTO intelligence_results
            (product_id, category, trend_label, trend_score, confidence, scored_at)
        VALUES
            (:product_id, :category, :trend_label, :trend_score, :confidence, NOW())
        ON CONFLICT (product_id)
        DO UPDATE SET
            trend_label  = EXCLUDED.trend_label,
            trend_score  = EXCLUDED.trend_score,
            confidence   = EXCLUDED.confidence,
            category     = EXCLUDED.category,
            scored_at    = NOW()
    """)

    # Aynı zamanda products.trend_score güncelle
    product_update_sql = text("""
        UPDATE products
        SET
            trend_score     = :trend_score,
            trend_direction = :trend_label,
            last_scored_at  = NOW()
        WHERE id = :product_id
    """)

    written = 0
    try:
        with engine.begin() as conn:
            for row in predictions:
                conn.execute(upsert_sql, {
                    "product_id":  row.get("product_id"),
                    "category":    row.get("category", ""),
                    "trend_label": row.get("trend_label", ""),
                    "trend_score": row.get("trend_score"),
                    "confidence":  row.get("confidence"),
                })
                conn.execute(product_update_sql, {
                    "trend_score": row.get("trend_score"),
                    "trend_label": row.get("trend_label", ""),
                    "product_id":  row.get("product_id"),
                })
                written += 1
        logger.info(f"{written} tahmin kaydedildi / products.trend_score güncellendi")
    except Exception as e:
        logger.error(f"Tahmin kaydetme hatası: {e}")

    return written


def save_alert(alert: dict) -> bool:
    """
    trend_alerts tablosuna yeni bir uyarı kaydeder.

    Args:
        alert: {type, product_id?, category, message, extra_data?}

    Returns:
        True → başarılı
    """
    _ensure_tables()

    sql = text("""
        INSERT INTO trend_alerts
            (alert_type, product_id, category, message, extra_data, created_at, is_read)
        VALUES
            (:alert_type, :product_id, :category, :message, :extra_data::jsonb, NOW(), FALSE)
    """)

    import json
    try:
        with engine.begin() as conn:
            conn.execute(sql, {
                "alert_type": alert.get("type", "unknown"),
                "product_id": alert.get("product_id"),
                "category":   alert.get("category", ""),
                "message":    alert.get("message", ""),
                "extra_data": json.dumps(alert.get("extra_data", {})),
            })
        logger.info(f"Alert kaydedildi: {alert.get('type')} / {alert.get('category')}")
        return True
    except Exception as e:
        logger.error(f"Alert kaydetme hatası: {e}")
        return False


def mark_alerts_read(alert_ids: list[int]) -> int:
    """Belirtilen alert ID'lerini okundu olarak işaretler."""
    if not alert_ids:
        return 0
    sql = text("""
        UPDATE trend_alerts
        SET is_read = TRUE
        WHERE id = ANY(:ids)
    """)
    try:
        with engine.begin() as conn:
            result = conn.execute(sql, {"ids": alert_ids})
            return result.rowcount
    except Exception as e:
        logger.error(f"Alert güncelleme hatası: {e}")
        return 0


def get_alerts(unread_only: bool = False, limit: int = 50) -> list[dict]:
    """trend_alerts tablosundan alertleri okur."""
    _ensure_tables()

    condition = "WHERE is_read = FALSE" if unread_only else ""
    sql = text(f"""
        SELECT id, alert_type, product_id, category, message, extra_data, created_at, is_read
        FROM trend_alerts
        {condition}
        ORDER BY created_at DESC
        LIMIT :limit
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"limit": limit}).fetchall()
            return [
                {
                    "id":         row[0],
                    "type":       row[1],
                    "product_id": row[2],
                    "category":   row[3],
                    "message":    row[4],
                    "extra_data": row[5],
                    "created_at": str(row[6]),
                    "is_read":    row[7],
                }
                for row in rows
            ]
    except Exception as e:
        logger.error(f"Alert okuma hatası: {e}")
        return []


def _ensure_tables():
    """
    Gerekli tabloları oluşturur (yoksa).
    products tablosuna Intelligence kolonlarını ekler (yoksa).
    
    Backend'in sync_schema() ile çakışmaz:
    - Backend: products tablosuna kendi kolonlarını ekler
    - Intelligence: trend_score, trend_direction, last_scored_at ekler
    - Her iki taraf da ALTER TABLE IF NOT EXISTS kullanır → idempotent
    """
    ddl_statements = [
        # Intelligence sonuçları
        """
        CREATE TABLE IF NOT EXISTS intelligence_results (
            id          SERIAL PRIMARY KEY,
            product_id  INTEGER UNIQUE REFERENCES products(id) ON DELETE CASCADE,
            category    VARCHAR(100),
            trend_label VARCHAR(30),
            trend_score FLOAT,
            confidence  FLOAT,
            scored_at   TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        # Trend uyarıları
        """
        CREATE TABLE IF NOT EXISTS trend_alerts (
            id           SERIAL PRIMARY KEY,
            alert_type   VARCHAR(50)  NOT NULL,
            product_id   INTEGER      REFERENCES products(id) ON DELETE SET NULL,
            category     VARCHAR(100),
            message      TEXT,
            extra_data   JSONB        DEFAULT '{}',
            created_at   TIMESTAMPTZ  DEFAULT NOW(),
            is_read      BOOLEAN      DEFAULT FALSE
        )
        """,
        # products tablosuna Intelligence kolonları (idempotent)
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS trend_score     FLOAT",
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS trend_direction VARCHAR(30)",
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS last_scored_at  TIMESTAMPTZ",
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS dominant_color  VARCHAR(50)",
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS fabric_type     VARCHAR(50)",
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS fit_type        VARCHAR(50)",
        # daily_metrics rank momentum kolonları (Intelligence Faz 1)
        "ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS rank_change_1d  INTEGER",
        "ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS rank_change_3d  INTEGER",
        "ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS rank_velocity   FLOAT",
        "ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS momentum_score  FLOAT",
        "ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS is_new_entrant  BOOLEAN DEFAULT FALSE",
        "ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS stock_depth     INTEGER",
        # Kategori günlük sinyal tablosu
        """
        CREATE TABLE IF NOT EXISTS category_daily_signals (
            id              SERIAL PRIMARY KEY,
            signal_date     DATE        NOT NULL DEFAULT CURRENT_DATE,
            search_term     VARCHAR(100) NOT NULL,
            total_products  INTEGER,
            rising_count    INTEGER,
            falling_count   INTEGER,
            new_entrants    INTEGER,
            avg_fav_change  FLOAT,
            avg_rank_change FLOAT,
            category_heat   FLOAT,
            is_hot          BOOLEAN DEFAULT FALSE,
            is_cold         BOOLEAN DEFAULT FALSE,
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT uq_category_signal_date_term UNIQUE (signal_date, search_term)
        )
        """,
        # ── Feedback Loop Tabloları ─────────────────────────────────────────
        # production_decisions: Intelligence TREND dediği ürünler için üretim kararları
        """
        CREATE TABLE IF NOT EXISTS production_decisions (
            id              SERIAL PRIMARY KEY,
            product_id      INTEGER REFERENCES products(id) ON DELETE CASCADE,
            search_term     VARCHAR(100),
            predicted_score FLOAT,
            decision        VARCHAR(20) DEFAULT 'produce',
            quantity        INTEGER,
            notes           VARCHAR(500),
            decided_at      TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        # actual_sales: gerçek satış sonuçları → Kalman/CatBoost feedback
        """
        CREATE TABLE IF NOT EXISTS actual_sales (
            id                SERIAL PRIMARY KEY,
            production_id     INTEGER REFERENCES production_decisions(id) ON DELETE CASCADE,
            sold_quantity     INTEGER,
            produced_quantity INTEGER,
            sell_through_rate FLOAT,
            revenue           FLOAT,
            feedback_date     DATE DEFAULT CURRENT_DATE,
            created_at        TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        # ── C1: Style Trends  ────────────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS style_trends (
            id              SERIAL PRIMARY KEY,
            signal_date     DATE NOT NULL DEFAULT CURRENT_DATE,
            search_term     VARCHAR(200) NOT NULL,
            attribute_key   VARCHAR(100) NOT NULL,
            attribute_value VARCHAR(200) NOT NULL,
            product_count   INTEGER DEFAULT 0,
            pct             NUMERIC(5,2) DEFAULT 0,
            prev_pct        NUMERIC(5,2),
            pct_change      NUMERIC(5,2),
            is_trending     BOOLEAN DEFAULT FALSE,
            UNIQUE(signal_date, search_term, attribute_key, attribute_value)
        )
        """,
        # ── C2: Model Versions  ──────────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS model_versions (
            id              SERIAL PRIMARY KEY,
            category        VARCHAR(200) NOT NULL,
            version         INTEGER NOT NULL DEFAULT 1,
            profile_type    VARCHAR(50),
            train_rows      INTEGER,
            train_products  INTEGER,
            r2_score        NUMERIC(6,4),
            mae             NUMERIC(10,4),
            mape            NUMERIC(6,2),
            feature_count   INTEGER,
            top_features    JSONB,
            trained_at      TIMESTAMPTZ DEFAULT NOW(),
            model_path      TEXT,
            notes           TEXT
        )
        """,
        # ── C3: Categories Registry  ─────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS categories_registry (
            id              SERIAL PRIMARY KEY,
            search_term     VARCHAR(200) UNIQUE NOT NULL,
            profile_type    VARCHAR(50) DEFAULT 'mid_fashion',
            lifecycle       VARCHAR(20) DEFAULT 'COLD',
            data_days       INTEGER DEFAULT 0,
            total_products  INTEGER DEFAULT 0,
            feedback_count  INTEGER DEFAULT 0,
            last_trained_at TIMESTAMPTZ,
            last_scored_at  TIMESTAMPTZ,
            kalman_state    JSONB,
            config_override JSONB,
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            updated_at      TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        # İndeksler
        "CREATE INDEX IF NOT EXISTS idx_products_trend_score    ON products(trend_score DESC) WHERE trend_score IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS idx_products_dominant_color ON products(dominant_color) WHERE dominant_color IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS idx_intelligence_scored_at  ON intelligence_results(scored_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_daily_metrics_pid_date  ON daily_metrics(product_id, recorded_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_category_signals_date   ON category_daily_signals(signal_date DESC, search_term)",
        "CREATE INDEX IF NOT EXISTS idx_prod_decisions_pid      ON production_decisions(product_id, decided_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_actual_sales_prod_id    ON actual_sales(production_id)",
        "CREATE INDEX IF NOT EXISTS idx_style_trends_date       ON style_trends(signal_date DESC, search_term)",
        "CREATE INDEX IF NOT EXISTS idx_model_versions_cat      ON model_versions(category, trained_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_categories_lifecycle    ON categories_registry(lifecycle, search_term)",
        # ── categories_registry şema migrasyonları (idempotent) ──────────
        "ALTER TABLE categories_registry ADD COLUMN IF NOT EXISTS group_name VARCHAR(200)",
        "ALTER TABLE categories_registry ADD COLUMN IF NOT EXISTS overrides  JSONB DEFAULT '{}'",
    ]

    try:
        with engine.begin() as conn:
            for ddl in ddl_statements:
                conn.execute(text(ddl))
        logger.info("✅ Intelligence tabloları ve products kolonları kontrol edildi")
    except Exception as e:
        logger.error(f"Tablo oluşturma hatası: {e}")


# ─── Kategori Sinyal Yazıcı ───────────────────────────────────────────────────

def save_category_signal(signal: dict) -> bool:
    """
    category_daily_signals tablosuna günlük kategori sinyalini upsert eder.

    Args:
        signal: {
            search_term, total_products, rising_count, falling_count,
            new_entrants, avg_fav_change, avg_rank_change, category_heat
        }
    """
    _ensure_tables()
    sql = text("""
        INSERT INTO category_daily_signals
            (signal_date, search_term, total_products, rising_count, falling_count,
             new_entrants, avg_fav_change, avg_rank_change, category_heat, is_hot, is_cold)
        VALUES
            (CURRENT_DATE, :search_term, :total_products, :rising_count, :falling_count,
             :new_entrants, :avg_fav_change, :avg_rank_change, :category_heat,
             :category_heat > 0.8, :category_heat < -0.5)
        ON CONFLICT (signal_date, search_term)
        DO UPDATE SET
            total_products  = EXCLUDED.total_products,
            rising_count    = EXCLUDED.rising_count,
            falling_count   = EXCLUDED.falling_count,
            new_entrants    = EXCLUDED.new_entrants,
            avg_fav_change  = EXCLUDED.avg_fav_change,
            avg_rank_change = EXCLUDED.avg_rank_change,
            category_heat   = EXCLUDED.category_heat,
            is_hot          = EXCLUDED.is_hot,
            is_cold         = EXCLUDED.is_cold
    """)
    try:
        with engine.begin() as conn:
            conn.execute(sql, {
                "search_term":     signal.get("search_term", ""),
                "total_products":  signal.get("total_products", 0),
                "rising_count":    signal.get("rising_count", 0),
                "falling_count":   signal.get("falling_count", 0),
                "new_entrants":    signal.get("new_entrants", 0),
                "avg_fav_change":  signal.get("avg_fav_change", 0.0),
                "avg_rank_change": signal.get("avg_rank_change", 0.0),
                "category_heat":   signal.get("category_heat", 0.0),
            })
        logger.info(f"Kategori sinyali kaydedildi: {signal.get('search_term')} heat={signal.get('category_heat', 0):.2f}")
        return True
    except Exception as e:
        logger.error(f"Kategori sinyal hatası: {e}")
        return False


# ─── Rank Momentum Hesaplayıcı (Nightly) ─────────────────────────────────────

def update_rank_momentum() -> int:
    """
    daily_metrics tablosundaki rank_change_1d, rank_change_3d, rank_velocity,
    momentum_score, is_new_entrant kolonlarını günceller.

    Her gece nightly_batch() sonunda çağrılır.
    Mantık:
      - rank_change_1d  = dünkü absolute_rank - bugünkü absolute_rank  (+iyileşme)
      - rank_change_3d  = 3 gün önceki rank - bugünkü rank
      - rank_velocity   = EMA(rank_change_1d, alpha=0.3)
      - momentum_score  = tanh(rank_change_3d / 100) → [-1, +1]
      - is_new_entrant  = dün top100'de yoktu, bugün var

    Returns:
        Güncellenen satır sayısı
    """
    _ensure_tables()

    # Bugünkü metriklere dünkü ve 3 gün önceki rank'ı join et
    sql = text("""
        WITH today AS (
            SELECT DISTINCT ON (product_id)
                id, product_id, absolute_rank, recorded_at
            FROM daily_metrics
            WHERE recorded_at >= CURRENT_DATE
              AND absolute_rank IS NOT NULL
            ORDER BY product_id, recorded_at DESC
        ),
        yesterday AS (
            SELECT DISTINCT ON (product_id)
                product_id, absolute_rank AS rank_1d_ago
            FROM daily_metrics
            WHERE recorded_at >= CURRENT_DATE - INTERVAL '2 days'
              AND recorded_at < CURRENT_DATE
              AND absolute_rank IS NOT NULL
            ORDER BY product_id, recorded_at DESC
        ),
        three_days AS (
            SELECT DISTINCT ON (product_id)
                product_id, absolute_rank AS rank_3d_ago
            FROM daily_metrics
            WHERE recorded_at >= CURRENT_DATE - INTERVAL '4 days'
              AND recorded_at < CURRENT_DATE - INTERVAL '2 days'
              AND absolute_rank IS NOT NULL
            ORDER BY product_id, recorded_at DESC
        ),
        prev_velocity AS (
            SELECT DISTINCT ON (product_id)
                product_id, rank_velocity AS old_velocity
            FROM daily_metrics
            WHERE recorded_at < CURRENT_DATE
              AND rank_velocity IS NOT NULL
            ORDER BY product_id, recorded_at DESC
        )
        UPDATE daily_metrics dm
        SET
            rank_change_1d = COALESCE(y.rank_1d_ago, t.absolute_rank) - t.absolute_rank,
            rank_change_3d = COALESCE(td.rank_3d_ago, t.absolute_rank) - t.absolute_rank,
            rank_velocity  = ROUND(CAST(
                0.3 * (COALESCE(y.rank_1d_ago, t.absolute_rank) - t.absolute_rank)
                + 0.7 * COALESCE(pv.old_velocity, 0) AS NUMERIC), 2),
            momentum_score = TANH(
                CAST(COALESCE(td.rank_3d_ago, t.absolute_rank) - t.absolute_rank AS FLOAT) / 100.0
            ),
            is_new_entrant = (
                t.absolute_rank <= 100
                AND (y.rank_1d_ago IS NULL OR y.rank_1d_ago > 100)
            )
        FROM today t
        LEFT JOIN yesterday y  ON y.product_id  = t.product_id
        LEFT JOIN three_days td ON td.product_id = t.product_id
        LEFT JOIN prev_velocity pv ON pv.product_id = t.product_id
        WHERE dm.id = t.id
    """)

    try:
        with engine.begin() as conn:
            result = conn.execute(sql)
            count = result.rowcount
        logger.info(f"✅ Rank momentum güncellendi: {count} ürün")
        return count
    except Exception as e:
        logger.error(f"Rank momentum güncelleme hatası: {e}")
        return 0


# ─── Feedback Loop CRUD ───────────────────────────────────────────────────────

def save_production_decision(decision: dict) -> int | None:
    """
    production_decisions tablosuna üretim kararı kaydeder.

    Args:
        decision: {
            product_id, search_term, predicted_score,
            decision ('produce'|'skip'|'wait'), quantity, notes
        }

    Returns:
        Oluşturulan kaydın id'si, hata olursa None
    """
    _ensure_tables()
    sql = text("""
        INSERT INTO production_decisions
            (product_id, search_term, predicted_score, decision, quantity, notes)
        VALUES
            (:product_id, :search_term, :predicted_score, :decision, :quantity, :notes)
        RETURNING id
    """)
    try:
        with engine.begin() as conn:
            result = conn.execute(sql, {
                "product_id":      decision.get("product_id"),
                "search_term":     decision.get("search_term", ""),
                "predicted_score": decision.get("predicted_score"),
                "decision":        decision.get("decision", "produce"),
                "quantity":        decision.get("quantity", 0),
                "notes":           decision.get("notes", ""),
            })
            row = result.fetchone()
            new_id = row[0] if row else None
        logger.info(f"Üretim kararı kaydedildi: id={new_id}, product={decision.get('product_id')}")
        return new_id
    except Exception as e:
        logger.error(f"Üretim kararı kaydetme hatası: {e}")
        return None


def save_actual_sale(sale: dict) -> int | None:
    """
    actual_sales tablosuna gerçek satış verisi kaydeder.
    sell_through_rate otomatik hesaplanır.

    Args:
        sale: {
            production_id, sold_quantity, produced_quantity, revenue?
        }

    Returns:
        Oluşturulan kaydın id'si, hata olursa None
    """
    _ensure_tables()

    sold = sale.get("sold_quantity", 0)
    produced = sale.get("produced_quantity", 0)
    sell_through = round(sold / max(produced, 1), 4)

    sql = text("""
        INSERT INTO actual_sales
            (production_id, sold_quantity, produced_quantity, sell_through_rate, revenue)
        VALUES
            (:production_id, :sold_quantity, :produced_quantity, :sell_through_rate, :revenue)
        RETURNING id
    """)
    try:
        with engine.begin() as conn:
            result = conn.execute(sql, {
                "production_id":     sale.get("production_id"),
                "sold_quantity":     sold,
                "produced_quantity": produced,
                "sell_through_rate": sell_through,
                "revenue":          sale.get("revenue"),
            })
            row = result.fetchone()
            new_id = row[0] if row else None
        logger.info(
            f"Satış verisi kaydedildi: id={new_id}, "
            f"sold={sold}/{produced} (STR={sell_through:.1%})"
        )
        return new_id
    except Exception as e:
        logger.error(f"Satış verisi kaydetme hatası: {e}")
        return None


def get_feedback_stats() -> dict:
    """
    Feedback loop durumunu özetler.
    CatBoost retraining kararı bu istatistiklere bağlıdır.
    """
    _ensure_tables()
    sql = text("""
        SELECT
            (SELECT COUNT(*) FROM production_decisions)              AS total_decisions,
            (SELECT COUNT(*) FROM actual_sales)                     AS total_sales,
            (SELECT ROUND(AVG(sell_through_rate)::numeric, 3) FROM actual_sales) AS avg_sell_through,
            (SELECT MIN(decided_at)::date FROM production_decisions) AS first_decision,
            (SELECT MAX(feedback_date) FROM actual_sales)           AS last_feedback
    """)
    try:
        with engine.connect() as conn:
            row = conn.execute(sql).fetchone()
            if row:
                return {
                    "total_decisions":   row[0] or 0,
                    "total_sales":       row[1] or 0,
                    "avg_sell_through":  float(row[2]) if row[2] else None,
                    "first_decision":    str(row[3]) if row[3] else None,
                    "last_feedback":     str(row[4]) if row[4] else None,
                    "ready_for_catboost": (row[1] or 0) >= 30,
                }
    except Exception as e:
        logger.error(f"Feedback istatistik hatası: {e}")
    return {"total_decisions": 0, "total_sales": 0, "ready_for_catboost": False}


# ─── Model Version Kayıt ─────────────────────────────────────────────────────

def save_model_version(record: dict) -> int | None:
    """
    Her CatBoost eğitiminde model_versions tablosuna kayıt ekler.

    Args:
        record: {
            category, version, profile_type, train_rows, train_products,
            r2_score, mae, mape, feature_count, top_features (dict), model_path
        }
    """
    _ensure_tables()
    import json

    sql = text("""
        INSERT INTO model_versions
            (category, version, profile_type, train_rows, train_products,
             r2_score, mae, mape, feature_count, top_features, model_path)
        VALUES
            (:category, :version, :profile_type, :train_rows, :train_products,
             :r2_score, :mae, :mape, :feature_count, CAST(:top_features AS jsonb), :model_path)
        RETURNING id
    """)

    try:
        with engine.begin() as conn:
            top_feats = record.get("top_features", {})
            if isinstance(top_feats, dict):
                top_feats = json.dumps(top_feats)

            result = conn.execute(sql, {
                "category":       record.get("category"),
                "version":        record.get("version", 1),
                "profile_type":   record.get("profile_type"),
                "train_rows":     record.get("train_rows"),
                "train_products": record.get("train_products"),
                "r2_score":       record.get("r2_score"),
                "mae":            record.get("mae"),
                "mape":           record.get("mape"),
                "feature_count":  record.get("feature_count"),
                "top_features":   top_feats,
                "model_path":     record.get("model_path"),
            })
            row_id = result.scalar()
            logger.info(f"Model version kaydedildi: {record.get('category')} → #{row_id}")
            return row_id
    except Exception as e:
        logger.error(f"Model version kayıt hatası: {e}")
    return None


# ─── Categories Registry CRUD ─────────────────────────────────────────────────

def upsert_category_registry(data: dict):
    """
    categories_registry tablosuna kategori durumunu upsert eder.

    Args:
        data: {
            search_term, profile_type, lifecycle, data_days,
            total_products, feedback_count, kalman_state (dict),
            group_name (str|None), overrides (dict)
        }
    """
    _ensure_tables()
    import json

    sql = text("""
        INSERT INTO categories_registry
            (search_term, profile_type, lifecycle, data_days,
             total_products, feedback_count, last_scored_at,
             kalman_state, group_name, overrides, updated_at)
        VALUES
            (:search_term, :profile_type, :lifecycle, :data_days,
             :total_products, :feedback_count, NOW(),
             CAST(:kalman_state AS jsonb), :group_name,
             CAST(:overrides AS jsonb), NOW())
        ON CONFLICT (search_term)
        DO UPDATE SET
            profile_type   = EXCLUDED.profile_type,
            lifecycle      = EXCLUDED.lifecycle,
            data_days      = EXCLUDED.data_days,
            total_products = EXCLUDED.total_products,
            feedback_count = EXCLUDED.feedback_count,
            last_scored_at = NOW(),
            kalman_state   = EXCLUDED.kalman_state,
            group_name     = EXCLUDED.group_name,
            overrides      = EXCLUDED.overrides,
            updated_at     = NOW()
    """)

    try:
        with engine.begin() as conn:
            ks = data.get("kalman_state", {})
            if isinstance(ks, dict):
                ks = json.dumps(ks)
            ov = data.get("overrides", {})
            if isinstance(ov, dict):
                ov = json.dumps(ov)
            conn.execute(sql, {
                "search_term":    data.get("search_term"),
                "profile_type":   data.get("profile_type", "mid_fashion"),
                "lifecycle":      data.get("lifecycle", "COLD"),
                "data_days":      data.get("data_days", 0),
                "total_products": data.get("total_products", 0),
                "feedback_count": data.get("feedback_count", 0),
                "kalman_state":   ks,
                "group_name":     data.get("group_name"),
                "overrides":      ov,
            })
    except Exception as e:
        logger.error(f"Category registry upsert hatası: {e}")

