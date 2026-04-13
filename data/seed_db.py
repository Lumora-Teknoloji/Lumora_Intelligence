"""
seed_db.py — Local DB'ye test verisi yükler.
generate_dataset.py'den Kademe 1 / 2m veri üretip doğrudan lumora_db'ye INSERT eder.

Kullanım:
  python seed_db.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import json
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import create_engine, text

# ─── Config ─────────────────────────────────────────────────────────────
DB_URL = "postgresql://postgres:postgres@localhost:5432/lumora_db"
KADEME = 1       # Kristal berraklık — en net sinyal
N_DAYS = 60      # 2 aylık veri yeterli
START_DATE = date(2026, 1, 12)  # 2 ay öncesi
N_CATEGORIES = 5  # Hızlı test için 5 kategori
PRODUCTS_PER_PROFILE = 3  # 3 rising + 3 falling + 3 stable = 9/kategori = 45 ürün

# Import generate_dataset functions
from generate_dataset import (
    KADEME_CONFIGS, CATEGORIES, 
    generate_products, generate_daily_metrics
)

def main():
    print("=" * 60)
    print(" LUMORA — Local DB Seed Script")
    print("=" * 60)
    
    engine = create_engine(DB_URL)
    cfg = KADEME_CONFIGS[KADEME]
    seed = 42
    
    # Override CATEGORIES to use only N_CATEGORIES
    import generate_dataset as gd
    original_cats = gd.CATEGORIES
    gd.CATEGORIES = original_cats[:N_CATEGORIES]
    
    print(f"\n1. Ürünler üretiliyor ({N_CATEGORIES} kategori × {PRODUCTS_PER_PROFILE*3} ürün)...")
    products_df = generate_products(
        n_rising=PRODUCTS_PER_PROFILE,
        n_falling=PRODUCTS_PER_PROFILE,
        n_stable=PRODUCTS_PER_PROFILE,
        seed=seed, kademe=KADEME
    )
    print(f"   → {len(products_df)} ürün üretildi")
    
    print(f"\n2. Daily metrics üretiliyor ({N_DAYS} gün × {len(products_df)} ürün)...")
    metrics_df = generate_daily_metrics(
        products_df=products_df,
        n_days=N_DAYS,
        start_date=START_DATE,
        cfg=cfg,
        seed=seed
    )
    print(f"   → {len(metrics_df)} satır üretildi")
    
    # Restore
    gd.CATEGORIES = original_cats
    
    # ─── DB'ye yükle ──────────────────────────────────────────────────────
    print("\n3. Mevcut verileri temizleme...")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM intelligence_results"))
        conn.execute(text("DELETE FROM category_daily_signals"))
        conn.execute(text("DELETE FROM daily_metrics"))
        conn.execute(text("DELETE FROM products"))
    print("   → Temizlendi")
    
    print("\n4. products tablosuna yükleme...")
    prod_rows = []
    for _, row in products_df.iterrows():
        prod_rows.append({
            "id": int(row["product_id"]),
            "product_code": f"SB-{int(row['product_id']):06d}",
            "name": row["name"],
            "brand": row["brand"],
            "category": row["category"],
            "category_tag": row["category"],
            "seller": f"Seller {np.random.randint(1, 20)}",
            "last_price": float(row["price"]),
            "last_discount_rate": float(row["discount_rate"]),
            "attributes": row["attributes"] if isinstance(row["attributes"], str) else json.dumps(row["attributes"], ensure_ascii=False),
        })
    
    with engine.begin() as conn:
        for p in prod_rows:
            conn.execute(text("""
                INSERT INTO products (id, product_code, name, brand, category, category_tag, seller, last_price, last_discount_rate, attributes)
                VALUES (:id, :product_code, :name, :brand, :category, :category_tag, :seller, :last_price, :last_discount_rate, :attributes)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    brand = EXCLUDED.brand,
                    category = EXCLUDED.category
            """), p)
    print(f"   → {len(prod_rows)} ürün yüklendi")
    
    print(f"\n5. daily_metrics tablosuna yükleme ({len(metrics_df)} satır)...")
    batch_size = 500
    loaded = 0
    for i in range(0, len(metrics_df), batch_size):
        batch = metrics_df.iloc[i:i+batch_size]
        rows = []
        for _, m in batch.iterrows():
            rows.append({
                "product_id": int(m["product_id"]),
                "recorded_at": m["date"],
                "search_term": m["category"],
                "price": float(m["price"]),
                "discounted_price": float(m["discounted_price"]),
                "discount_rate": float(m["discount_rate"]),
                "cart_count": int(m["cart_count"]),
                "favorite_count": int(m["favorite_count"]),
                "view_count": int(m["view_count"]),
                "rating_count": int(m["rating_count"]),
                "avg_rating": float(m["rating"]),
                "absolute_rank": int(m["absolute_rank"]),
                "search_rank": int(m["absolute_rank"]) % 48 + 1,
                "page_number": int(m["absolute_rank"]) // 48 + 1,
                "engagement_score": float(m["engagement_score"]),
                "popularity_score": float(m["engagement_score"]) * 100,
            })
        
        with engine.begin() as conn:
            for r in rows:
                conn.execute(text("""
                    INSERT INTO daily_metrics (
                        product_id, recorded_at, search_term, price, discounted_price,
                        discount_rate, cart_count, favorite_count, view_count,
                        rating_count, avg_rating, absolute_rank, search_rank,
                        page_number, engagement_score, popularity_score
                    ) VALUES (
                        :product_id, :recorded_at, :search_term, :price, :discounted_price,
                        :discount_rate, :cart_count, :favorite_count, :view_count,
                        :rating_count, :avg_rating, :absolute_rank, :search_rank,
                        :page_number, :engagement_score, :popularity_score
                    )
                """), r)
        loaded += len(rows)
        pct = loaded * 100 // len(metrics_df)
        print(f"   → {loaded}/{len(metrics_df)} ({pct}%)", end="\r")
    
    print(f"\n   → {loaded} satır yüklendi")
    
    # ─── Doğrulama ─────────────────────────────────────────────────────
    print("\n6. Doğrulama...")
    with engine.connect() as conn:
        p_count = conn.execute(text("SELECT count(*) FROM products")).scalar()
        m_count = conn.execute(text("SELECT count(*) FROM daily_metrics")).scalar()
        cats = conn.execute(text("SELECT DISTINCT search_term FROM daily_metrics")).fetchall()
        date_range = conn.execute(text("SELECT MIN(recorded_at)::date, MAX(recorded_at)::date FROM daily_metrics")).fetchone()
    
    print(f"   products:      {p_count}")
    print(f"   daily_metrics: {m_count}")
    print(f"   kategoriler:   {[r[0] for r in cats]}")
    print(f"   tarih aralığı: {date_range[0]} → {date_range[1]}")
    
    print("\n" + "=" * 60)
    print(" ✅ SEED TAMAMLANDI!")
    print("=" * 60)

if __name__ == "__main__":
    main()
