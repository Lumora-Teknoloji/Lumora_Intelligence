"""
run_csv_test.py
---------------
data/output_v2/daily_metrics.csv dosyasını okuyup
PredictionEngine'e besler, sonuçları ekrana yazar ve
results/csv_test_results.csv'ye kaydeder.

Adımlar:
  1. CSV oku
  2. Engine'in beklediği kolonlara adapt et (fabric, color vs.)
  3. engine.train(df)
  4. engine.predict()
  5. Sonuçları yaz

Kullanım:
  python run_csv_test.py
  python run_csv_test.py --no-prophet --categories "crop top" "elbise yazlık"
  python run_csv_test.py --sample 30   # her kategoriden 30 ürün örnekle
"""

import sys, io, os, argparse, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from engine.predictor import PredictionEngine


# ──────────────────────────────────────────────────────────────────────────────
# CSV → Engine format adapter
# ──────────────────────────────────────────────────────────────────────────────
def adapt_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    generate_csv_v2 çıktısını engine.train() formatına dönüştürür.

    Engine'in zorunlu kolonları:
      product_id, category, date, cart_count, favorite_count,
      engagement_score, fabric, color, brand, price, rating

    v2 CSV'de mevcut: hepsi var.
    Eksik olanlar türetilir: fabric → 'Pamuk' default, color → 'Siyah' default
    """
    df = df.copy()

    # date → datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # fabric / color — CSV'de yok, category'den basit inference
    if "fabric" not in df.columns:
        CAT_FABRIC = {
            "tayt": "Polyester Likra", "spor": "Performans Kumaş",
            "mont": "Şişme Kumaş", "kaban": "Yün Karışım",
            "kazak": "Pamuk Akrilik", "hırka": "Pamuk",
            "abiye": "Saten", "elbise": "Şifon", "bluz": "Viskon",
            "gömlek": "Pamuk", "şort": "Denim", "etek": "Viskon",
            "pijama": "Saten", "tesettür": "Modal", "sweatshirt": "Pamuk",
            "hoodie": "Pamuk Polar", "jean": "Denim", "tunik": "Keten",
        }
        def infer_fabric(cat):
            cat_l = str(cat).lower()
            for key, fab in CAT_FABRIC.items():
                if key in cat_l:
                    return fab
            return "Pamuk"
        df["fabric"] = df["category"].apply(infer_fabric)

    if "color" not in df.columns:
        COLORS = ["Siyah", "Beyaz", "Ekru", "Lacivert", "Bej", "Haki", "Gri",
                  "Bordo", "Pembe", "Kırmızı", "Yeşil", "Mavi", "Kahverengi"]
        rng = np.random.default_rng(42)
        df["color"] = rng.choice(COLORS, size=len(df))

    # search_term — engine bazı yerlerde kullanır
    if "search_term" not in df.columns:
        df["search_term"] = df["category"]

    # engagement_score — None → NaN dönüşümü (engine robust ama güvenlik için)
    df["engagement_score"] = pd.to_numeric(df["engagement_score"], errors="coerce")

    # Engine bazı feature'lar için bu kolonları bekler
    for col in ["discount_rate", "view_count", "rating_count"]:
        if col not in df.columns:
            df[col] = 0

    return df


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CSV → PredictionEngine test")
    parser.add_argument("--csv",        default="data/output_v2/daily_metrics.csv")
    parser.add_argument("--no-prophet", action="store_true")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Filtrelenecek kategoriler (boşsa tümü)")
    parser.add_argument("--sample",     type=int, default=None,
                        help="Kategori başına max ürün sayısı (büyük dataseti küçültmek için)")
    parser.add_argument("--top",        type=int, default=30,
                        help="Gösterilecek en iyi N ürün (varsayılan: 30)")
    args = parser.parse_args()

    csv_path = ROOT / args.csv
    if not csv_path.exists():
        print(f"[HATA] CSV bulunamadı: {csv_path}")
        sys.exit(1)

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   📊 CSV → PredictionEngine Test Runner                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── 1. CSV Yükle ────────────────────────────────────────────────────────
    print(f"[1/4] CSV yükleniyor: {csv_path.name}")
    t0 = time.time()
    df_raw = pd.read_csv(csv_path, low_memory=False)
    print(f"  ✓  {len(df_raw):,} satır, {df_raw['product_id'].nunique():,} ürün, "
          f"{df_raw['category'].nunique()} kategori  "
          f"({time.time()-t0:.1f}s)")

    # Kategori filtresi
    if args.categories:
        df_raw = df_raw[df_raw["category"].isin(args.categories)]
        if df_raw.empty:
            print(f"  [HATA] Belirtilen kategoriler bulunamadı!")
            sys.exit(1)
        print(f"  Filtre: {df_raw['category'].nunique()} kategori seçildi")

    # Ürün örnekleme (büyük dataset için)
    if args.sample:
        sampled_pids = (
            df_raw.groupby("category")["product_id"]
            .apply(lambda x: x.drop_duplicates().sample(
                min(args.sample, x.nunique()), random_state=42))
            .explode().values
        )
        df_raw = df_raw[df_raw["product_id"].isin(sampled_pids)]
        print(f"  Örnekleme: {df_raw['product_id'].nunique():,} ürün "
              f"(kategori başına max {args.sample})")

    print()

    # ── 2. Adapt et ─────────────────────────────────────────────────────────
    print("[2/4] Veri formatı dönüştürülüyor...")
    df = adapt_csv(df_raw)
    print(f"  ✓  {len(df.columns)} kolon hazır")
    print()

    # ── 3. Engine Eğit ──────────────────────────────────────────────────────
    print("[3/4] Engine eğitiliyor...")
    t1 = time.time()
    engine = PredictionEngine(
        use_prophet=not args.no_prophet,
        use_clip=False
    )
    engine.train(df, verbose=True)
    print(f"\n  ✓  Eğitim tamamlandı ({time.time()-t1:.1f}s)")
    print()

    # ── 4. Tahmin yap ───────────────────────────────────────────────────────
    print("[4/4] Tahminler hesaplanıyor...")
    t2 = time.time()
    predictions = engine.predict()
    print(f"  ✓  {len(predictions):,} ürün için tahmin ({time.time()-t2:.1f}s)")
    print()

    if predictions.empty:
        print("  [HATA] Tahmin yapılamadı!")
        sys.exit(1)

    # ── SONUÇLAR ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  📈 TREND SONUÇLARI")
    print("=" * 70)

    # Etiket dağılımı
    label_counts = predictions["trend_label"].value_counts()
    total = len(predictions)
    print("\n  Etiket Dağılımı:")
    for lbl, cnt in label_counts.items():
        bar = "█" * min(40, int(cnt / total * 40))
        print(f"    {lbl:<15}: {cnt:>5,} ürün ({cnt/total*100:5.1f}%)  {bar}")

    # Kategori başına TREND sayısı (ilk 20)
    print("\n  Kategori Başına TREND Ürün Sayısı (ilk 20):")
    cat_trend = (predictions[predictions["trend_label"] == "TREND"]
                 .groupby("category").size()
                 .sort_values(ascending=False))
    for cat, cnt in cat_trend.head(20).items():
        print(f"    {cat:<35}: {cnt:>3} TREND ürün")

    # Top N ürünler
    print(f"\n  🏆 En Yüksek Trend Skoru (Top {args.top}):")
    cols_show = ["product_id", "category", "trend_score", "trend_label",
                 "confidence", "cart_growth_pct", "kalman_prod_velocity",
                 "ensemble_demand"]
    cols_avail = [c for c in cols_show if c in predictions.columns]
    top_df = predictions.head(args.top)[cols_avail]
    print(top_df.to_string(index=False))

    # TREND label'lı ürünlerin gerçek profil doğruluğu (CSV'de _trend_profile varsa)
    if "_trend_profile" in df_raw.columns:
        print("\n  🎯 Gerçek Profil vs Tahmin Doğruluğu:")
        true_profiles = (df_raw.groupby("product_id")["_trend_profile"]
                         .first().reset_index())
        check = predictions.merge(true_profiles, on="product_id", how="left")
        if not check.empty:
            # TREND tahmin → gerçekte rising olan ürün yüzdesi
            trend_preds = check[check["trend_label"] == "TREND"]
            if len(trend_preds) > 0:
                actual_rising = (trend_preds["_trend_profile"] == "rising").sum()
                pct = actual_rising / len(trend_preds) * 100
                print(f"    TREND tahmin edilen {len(trend_preds)} üründen "
                      f"{actual_rising} tanesi gerçekten rising → "
                      f"Precision: %{pct:.1f}")

            # DUSEN tahmin → gerçekte falling olan ürün yüzdesi
            dusen_preds = check[check["trend_label"] == "DUSEN"]
            if len(dusen_preds) > 0:
                actual_falling = (dusen_preds["_trend_profile"] == "falling").sum()
                pct2 = actual_falling / len(dusen_preds) * 100
                print(f"    DUSEN tahmin edilen {len(dusen_preds)} üründen "
                      f"{actual_falling} tanesi gerçekten falling → "
                      f"Precision: %{pct2:.1f}")

    # ── Kaydet ──────────────────────────────────────────────────────────────
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "csv_test_results.csv"
    predictions.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  💾 Sonuçlar kaydedildi: {out_path}")

    # Motor durumu
    status = engine.status()
    print()
    print("=" * 70)
    print("  ⚙️  MOTOR DURUMU")
    print("=" * 70)
    print(f"  CatBoost eğitildi: {'✅' if status['catboost_trained'] else '❌'}")
    print(f"  Prophet aktif    : {'✅' if status['prophet_enabled'] else '❌'}")
    print(f"  Kalman ürün bazlı: {len(status.get('kalman_category_states', {})):,} state")

    if status.get("catboost_features"):
        print(f"\n  🏆 Feature Importance (Top 8):")
        for i, (feat, imp) in enumerate(list(status["catboost_features"].items())[:8]):
            bar = "█" * max(1, int(imp / 3))
            print(f"    {i+1:2}. {feat:<30}: {imp:5.1f}%  {bar}")

    total_time = time.time() - t0
    print()
    print("═" * 70)
    print(f"  ✅ Tamamlandı!  Toplam süre: {total_time:.1f}s")
    print("═" * 70)


if __name__ == "__main__":
    main()
