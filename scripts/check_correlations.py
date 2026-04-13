import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np

# k3/6m: en gerçekci orta kademe
dm = pd.read_csv("data/datasets/k3/6m/daily_metrics.csv", parse_dates=["date"])
pr = pd.read_csv("data/datasets/k3/6m/products.csv",
                  usecols=["product_id", "_trend_profile"])
dm = dm.merge(pr.rename(columns={"_trend_profile": "prof"}), on="product_id")

cols = ["absolute_rank", "favorite_count", "cart_count",
        "view_count", "rating_count", "engagement_score"]

# ── 1. Korelasyon matrisi ────────────────────────────────────────────────────
corr = dm[cols].corr().round(2)
print("=== KORELASYON MATRİSİ (k3/6m) ===")
print(corr.to_string())
print()

# ── 2. Sepet → Rating gecikmeli korelasyon ───────────────────────────────────
dm_s = dm.sort_values(["product_id", "date"])
dm_s["cart_1d_later"]    = dm_s.groupby("product_id")["cart_count"].shift(-1)
dm_s["cart_7d_later"]    = dm_s.groupby("product_id")["cart_count"].shift(-7)
dm_s["rating_7d_later"]  = dm_s.groupby("product_id")["rating_count"].shift(-7)
dm_s["rating_14d_later"] = dm_s.groupby("product_id")["rating_count"].shift(-14)

print("=== SEPET → RATİNG GECİKMELİ KORELASYON ===")
lag_cols = ["cart_count", "cart_7d_later", "rating_count",
            "rating_7d_later", "rating_14d_later"]
print(dm_s[lag_cols].corr().round(2).to_string())
print()

# ── 3. Rising vs Stable vs Falling karşılaştırması ──────────────────────────
print("=== PROFİL KARŞILAŞTIRMASI (ortalama değerler) ===")
print(f"{'':12}  {'fav':>8}  {'cart':>8}  {'view':>8}  {'rating':>8}  {'engagement':>10}")
for prof in ["rising", "stable", "falling"]:
    sub = dm[dm["prof"] == prof]
    print(f"  {prof:<12}"
          f"  {sub['favorite_count'].mean():>8.0f}"
          f"  {sub['cart_count'].mean():>8.1f}"
          f"  {sub['view_count'].mean():>8.0f}"
          f"  {sub['rating_count'].mean():>8.0f}"
          f"  {sub['engagement_score'].mean():>10.3f}")
print()

# ── 4. Fav artışı → cart gecikmesi testi ────────────────────────────────────
print("=== FAV ARTIŞI → CART ARTIŞI ANLIKKLIĞI ===")
dm_s["fav_delta_7d"] = dm_s.groupby("product_id")["favorite_count"].diff(7)
dm_s["cart_delta_7d"] = dm_s.groupby("product_id")["cart_count"].diff(7)

cor_same = dm_s["fav_delta_7d"].corr(dm_s["cart_delta_7d"])
cor_lag3  = dm_s["fav_delta_7d"].corr(
    dm_s.groupby("product_id")["cart_count"].diff(7).shift(-3))

print(f"  fav_delta_7d ↔ cart_delta_7d (aynı an): r={cor_same:.2f}")
print(f"  fav_delta_7d → cart_delta (3 gün sonra): r={cor_lag3:.2f}")
print()

# ── 5. Zincir analizi: rank → fav → cart → rating ────────────────────────────
print("=== ZİNCİR ANALİZİ (rank → fav → cart → rating) ===")
chain = {
    "rank_reach → fav_count":        dm["rank_reach_mult"].corr(dm["favorite_count"]),
    "rank_reach → cart_count":       dm["rank_reach_mult"].corr(dm["cart_count"]),
    "rank_reach → rating_count":     dm["rank_reach_mult"].corr(dm["rating_count"]),
    "fav_count → cart_count":        dm["favorite_count"].corr(dm["cart_count"]),
    "fav_count → rating_count":      dm["favorite_count"].corr(dm["rating_count"]),
    "cart_count → rating_count":     dm["cart_count"].corr(dm["rating_count"]),
    "engagement → rating_count":     dm["engagement_score"].corr(dm["rating_count"]),
}
for label, r in chain.items():
    strength = "GÜÇLİ" if abs(r) > 0.7 else ("ORTA" if abs(r) > 0.4 else "ZAYIF")
    print(f"  {label:<40}: r={r:+.2f}  [{strength}]")
print()

# ── 6. Kademeler arası korelasyon özeti ─────────────────────────────────────
print("=== KADEMELER ARASI SEPET-RATING KORELASYONU ===")
print(f"  {'Kademe':<20}  {'fav-cart r':>10}  {'fav-rating r':>12}  {'cart-rating r':>13}")
for k in range(1, 6):
    try:
        dm_k = pd.read_csv(f"data/datasets/k{k}/6m/daily_metrics.csv")
        r_fc = dm_k["favorite_count"].corr(dm_k["cart_count"])
        r_fr = dm_k["favorite_count"].corr(dm_k["rating_count"])
        r_cr = dm_k["cart_count"].corr(dm_k["rating_count"])
        knames = {1:"Kristal",2:"Net",3:"Orta",4:"Gürültülü",5:"Kaotik"}
        print(f"  k{k} [{knames[k]:<12}]  {r_fc:>10.2f}  {r_fr:>12.2f}  {r_cr:>13.2f}")
    except Exception as e:
        print(f"  k{k}: HATA - {e}")
