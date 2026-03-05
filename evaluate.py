# evaluate.py
"""
🎯 Lumora Intelligence — Tahmin Doğruluğu Değerlendirmesi

Yöntem:
  1. Gerçek trend etiketleri olan veri üret (rising/viral/falling/stable)
  2. İlk 45 günde eğit
  3. Son 15 günde test et (sistem bu günleri "görmemişti")
  4. Sistem kararı vs gerçek etiket karşılaştır

Metrikler:
  - Trend yakalama oranı (Rising + Viral → TREND tespit edildi mi?)
  - Düşüş uyarı oranı (Falling → DUSEN tespit edildi mi?)
  - Viral tespit hızı (Kaçıncı günde fark etti?)
  - CatBoost: MAE, R², composite score korelasyonu
  - Kalman: velocity yön doğruluğu
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.sample_data import generate_products, generate_daily_metrics
from engine.predictor import PredictionEngine

# ──────────────────────────────────────────────────────────────
# Gerçek → Beklenen etiket eşleştirmesi
# ──────────────────────────────────────────────────────────────
TRUE_TO_EXPECTED = {
    "viral":               ["TREND"],
    "rising":              ["TREND", "POTANSIYEL"],
    "seasonal":            ["TREND", "POTANSIYEL", "STABIL"],  # orta güven
    "new_entrant":         ["POTANSIYEL", "STABIL"],
    "stable":              ["STABIL"],
    "falling":             ["DUSEN"],
    "competitor_dominated":["DUSEN"],
}

# Hangi profil yükseliş sinyali vermeli?
SHOULD_RISE = {"viral", "rising"}
SHOULD_FALL = {"falling", "competitor_dominated"}
SHOULD_STABLE = {"stable"}


def evaluate(n_products=150, train_days=45, test_days=15, verbose=True):
    total_days = train_days + test_days
    now = datetime.now()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   🎯 LUMORA INTELLIGENCE — DOĞRULUK DEĞERLENDİRMESİ    ║")
    print(f"║   Eğitim: {train_days} gün | Test: {test_days} gün | {n_products} ürün          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # ── 1. VERİ ÜRET ────────────────────────────────────────────
    print("📦 Veri üretiliyor...")
    products = generate_products(n=n_products,
                                  categories=["crop", "tayt", "grup", "kadın abiye"])
    all_metrics = generate_daily_metrics(products, days=total_days, scraper_gaps=False)

    # Gerçek profil etiketlerini al (her ürünün ilk profili)
    profile_per_product = all_metrics.groupby("product_id")["_trend_profile"].first().reset_index()
    profile_per_product.columns = ["product_id", "true_profile"]
    true_labels = profile_per_product

    profile_dist = true_labels["true_profile"].value_counts()
    print(f"   Toplam kayıt: {len(all_metrics):,}")
    print(f"   Trend profil dağılımı:")
    for prof, cnt in profile_dist.items():
        print(f"     {prof:25s}: {cnt} ürün")
    print()

    # ── 2. EĞİTİM / TEST BÖLME ──────────────────────────────────
    cutoff = now - timedelta(days=test_days)
    train_data = all_metrics[all_metrics["date"] < cutoff].copy()
    test_data  = all_metrics[all_metrics["date"] >= cutoff].copy()

    print(f"📊 Veri bölünmesi:")
    print(f"   Eğitim seti: {len(train_data):,} kayıt ({train_data['product_id'].nunique()} ürün)")
    print(f"   Test seti:   {len(test_data):,} kayıt ({test_data['product_id'].nunique()} ürün)")
    print()

    # ── 3. MOTOR EĞİT ───────────────────────────────────────────
    print("🧠 Motor eğitiliyor (sadece eğitim verisiyle)...")
    engine = PredictionEngine(use_prophet=False, use_clip=False)
    engine.train(train_data, verbose=False)
    print("   ✅ Eğitim tamamlandı")
    print()

    # ── 4. TEST VERİSİNİ TAHMIN ET ──────────────────────────────
    print("🔮 Test verisi tahmin ediliyor...")
    predictions = engine.predict(test_data)

    if predictions.empty:
        print("❌ Tahmin yapılamadı!")
        return

    # Gerçek etiketlerle birleştir
    eval_df = predictions.merge(
        true_labels.reset_index()[["product_id", "true_profile"]],
        on="product_id", how="left"
    )

    total = len(eval_df)
    print(f"   Tahmin yapılan ürün: {total}")
    print()

    # ── 5. DOĞRULUK METRİKLERİ ──────────────────────────────────
    print("=" * 60)
    print("📈 DOĞRULUK METRİKLERİ")
    print("=" * 60)

    # 5.1 Etiket doğruluk matrisi
    print("\n🏷️  Tahmin Etiketleri vs. Gerçek Profil:")
    matrix = pd.crosstab(
        eval_df["true_profile"],
        eval_df["trend_label"],
        rownames=["Gerçek"],
        colnames=["Tahmin"],
        margins=True
    )
    print(matrix.to_string())
    print()

    # 5.2 Yükseliş yakalama oranı
    rising_products = eval_df[eval_df["true_profile"].isin(SHOULD_RISE)]
    if len(rising_products) > 0:
        caught_as_trend    = (rising_products["trend_label"] == "TREND").sum()
        caught_as_pot      = (rising_products["trend_label"] == "POTANSIYEL").sum()
        caught_as_stabil   = (rising_products["trend_label"] == "STABIL").sum()
        missed_as_dusen    = (rising_products["trend_label"] == "DUSEN").sum()

        print(f"📈 YÜKSELİŞ YAKALAMA (rising + viral = {len(rising_products)} ürün):")
        print(f"   TREND      olarak etiketlendi: {caught_as_trend:3d}  "
              f"({caught_as_trend/len(rising_products)*100:.1f}%)  ← İdeal")
        print(f"   POTANSIYEL olarak etiketlendi: {caught_as_pot:3d}  "
              f"({caught_as_pot/len(rising_products)*100:.1f}%)  ← Kabul edilebilir")
        print(f"   STABIL     olarak etiketlendi: {caught_as_stabil:3d}  "
              f"({caught_as_stabil/len(rising_products)*100:.1f}%)  ← Kaçırıldı")
        print(f"   DÜŞEN      olarak etiketlendi: {missed_as_dusen:3d}  "
              f"({missed_as_dusen/len(rising_products)*100:.1f}%)  ← ❌ Yanlış")

        hit_rate = (caught_as_trend + caught_as_pot) / len(rising_products) * 100
        print(f"   ➤ Yükseliş tespiti (TREND+POT): {hit_rate:.1f}%")
    print()

    # 5.3 Düşüş yakalama oranı
    falling_products = eval_df[eval_df["true_profile"].isin(SHOULD_FALL)]
    if len(falling_products) > 0:
        caught_as_dusen = (falling_products["trend_label"] == "DUSEN").sum()
        false_alarm     = (falling_products["trend_label"] == "TREND").sum()

        print(f"📉 DÜŞÜŞ YAKALAMA (falling + competitor_dominated = {len(falling_products)} ürün):")
        print(f"   DUSEN olarak etiketlendi: {caught_as_dusen:3d}  "
              f"({caught_as_dusen/len(falling_products)*100:.1f}%)  ← İdeal")
        print(f"   TREND olarak etiketlendi: {false_alarm:3d}  "
              f"({false_alarm/len(falling_products)*100:.1f}%)  ← ❌ Sahte alarm")
        drop_rate = caught_as_dusen / len(falling_products) * 100
        print(f"   ➤ Düşüş tespiti: {drop_rate:.1f}%")
    print()

    # 5.4 Viral tespit
    viral_products = eval_df[eval_df["true_profile"] == "viral"]
    if len(viral_products) > 0:
        viral_as_trend = (viral_products["trend_label"] == "TREND").sum()
        print(f"🔥 VİRAL TESPİT ({len(viral_products)} viral ürün):")
        print(f"   TREND olarak yakalandı: {viral_as_trend}/{len(viral_products)}  "
              f"({viral_as_trend/len(viral_products)*100:.0f}%)")
        if len(viral_products) > 0:
            viral_scores = viral_products["trend_score"].values
            print(f"   Viral ürün trend skorları: {[round(s,1) for s in sorted(viral_scores, reverse=True)]}")
    print()

    # 5.5 Yön Doğruluğu (Kalman velocity)
    print(f"🔄 KALMAN VELOCİTY YÖN DOĞRULUĞU:")
    kalman_correct = 0
    kalman_total   = 0
    for _, row in eval_df.iterrows():
        profile = row.get("true_profile")
        velocity = row.get("kalman_prod_velocity", 0) or 0
        if profile in SHOULD_RISE:
            kalman_total += 1
            if velocity > 0:
                kalman_correct += 1
        elif profile in SHOULD_FALL:
            kalman_total += 1
            if velocity < 0:
                kalman_correct += 1

    if kalman_total > 0:
        kalman_acc = kalman_correct / kalman_total * 100
        print(f"   Velocity yönü doğru: {kalman_correct}/{kalman_total} ({kalman_acc:.1f}%)")
    print()

    # 5.6 CatBoost composite score kalitesi
    print(f"🤖 CATBOOST COMPOSITE SCORE:")
    score_by_profile = eval_df.groupby("true_profile")["trend_score"].agg(["mean", "std", "count"])
    score_by_profile.columns = ["Ort. Trend Skoru", "Std", "Ürün Sayısı"]
    score_by_profile = score_by_profile.sort_values("Ort. Trend Skoru", ascending=False)
    print(score_by_profile.round(1).to_string())
    print()

    # 5.7 Özet skor tablosu
    print("=" * 60)
    print("📋 ÖZET DEĞERLENDİRME")
    print("=" * 60)

    # Genel doğruluk: sistem doğru etiket verdi mi?
    correct = 0
    for _, row in eval_df.iterrows():
        profile = row.get("true_profile", "unknown")
        label = row.get("trend_label", "STABIL")
        expected = TRUE_TO_EXPECTED.get(profile, ["STABIL"])
        if label in expected:
            correct += 1

    overall_acc = correct / total * 100 if total > 0 else 0

    print(f"\n   Toplam ürün:           {total}")
    print(f"   Doğru etiket:          {correct} ({overall_acc:.1f}%)")
    print(f"   Yanlış etiket:         {total - correct} ({100-overall_acc:.1f}%)")

    # Kategori bazlı
    print(f"\n   Kategori bazlı doğruluk:")
    for cat in eval_df["category"].dropna().unique():
        cat_df = eval_df[eval_df["category"] == cat]
        cat_correct = 0
        for _, row in cat_df.iterrows():
            profile = row.get("true_profile", "unknown")
            label   = row.get("trend_label", "STABIL")
            if label in TRUE_TO_EXPECTED.get(profile, ["STABIL"]):
                cat_correct += 1
        if len(cat_df) > 0:
            print(f"     {cat:20s}: {cat_correct}/{len(cat_df)} "
                  f"({cat_correct/len(cat_df)*100:.1f}%)")

    # Yükselen önerilerin precision'ı
    trend_predicted = eval_df[eval_df["trend_label"] == "TREND"]
    if len(trend_predicted) > 0:
        actually_rising = trend_predicted["true_profile"].isin(SHOULD_RISE).sum()
        precision = actually_rising / len(trend_predicted) * 100
        print(f"\n   Precision (TREND dedi → gerçekten yükselen): "
              f"{actually_rising}/{len(trend_predicted)} ({precision:.1f}%)")
        print(f"   (Bu '% yanlış üretim kararı' metriği)")

    # Recall
    all_rising = eval_df["true_profile"].isin(SHOULD_RISE).sum()
    if all_rising > 0:
        actually_caught_trend = eval_df[
            eval_df["true_profile"].isin(SHOULD_RISE) &
            (eval_df["trend_label"] == "TREND")
        ].shape[0]
        recall = actually_caught_trend / all_rising * 100
        print(f"   Recall (Yükselen ürünleri TREND diye etiketleme): "
              f"{actually_caught_trend}/{all_rising} ({recall:.1f}%)")
        print(f"   (Bu 'kaçan fırsat' metriği)")

    print()
    print("═" * 60)
    print(f"  ✅ Değerlendirme tamamlandı!")
    print("═" * 60)

    return {
        "overall_accuracy": overall_acc,
        "hit_rate": hit_rate if len(rising_products) > 0 else 0,
        "drop_rate": drop_rate if len(falling_products) > 0 else 0,
        "viral_detection": viral_as_trend / len(viral_products) * 100 if len(viral_products) > 0 else 0,
        "kalman_direction_acc": kalman_acc if kalman_total > 0 else 0,
    }


if __name__ == "__main__":
    results = evaluate(n_products=200, train_days=45, test_days=15)
