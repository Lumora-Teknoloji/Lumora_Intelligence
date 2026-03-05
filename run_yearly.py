# run_yearly.py
"""
📅 1 Yıllık Rolling Window Değerlendirmesi

Yöntem:
  Her ay için sliding window:
    - Eğitim: önceki 90 gün
    - Test:   sonraki 30 gün
  
  Ay 3  → eğitim: Gün 1-90,    test: Gün 91-120
  Ay 4  → eğitim: Gün 31-120,  test: Gün 121-150
  ...
  Ay 12 → eğitim: Gün 275-365, test: son 30 gün

Sonuç:
  - Ay ay yükseliş/düşüş tespiti
  - Mevsim geçişlerinde doğruluk
  - Sistemin zamanla öğrenme eğrisi
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from datetime import timedelta
from data.yearly_simulation import generate_yearly_data, SIM_START

SHOULD_RISE = {"viral", "rising", "seasonal"}
SHOULD_FALL = {"falling"}


def run_yearly_evaluation():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   📅 1 YILLIK ROLLING WINDOW DEĞERLENDİRMESİ              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("📦 1 yıllık veri üretiliyor...")
    prod_df, metr_df = generate_yearly_data()
    print()

    # Gerçek profiller
    profiles = metr_df.groupby("product_id")["_trend_profile"].first().reset_index()
    profiles.columns = ["product_id", "true_profile"]

    TRAIN_DAYS = 90    # eğitim penceresi
    TEST_DAYS  = 30    # test penceresi
    STEP_DAYS  = 30    # her ay bir adım

    monthly_results = []
    month_names = []

    start_month = 3   # ilk 90 günlük eğitim için ay 3'ten başla
    n_windows = (365 - TRAIN_DAYS - TEST_DAYS) // STEP_DAYS + 1

    print(f"📊 {n_windows} aylık pencere değerlendirilecek...")
    print(f"   Eğitim: {TRAIN_DAYS} gün | Test: {TEST_DAYS} gün | Adım: {STEP_DAYS} gün")
    print()

    from engine.predictor import PredictionEngine

    for w in range(n_windows):
        train_start_day = w * STEP_DAYS
        train_end_day   = train_start_day + TRAIN_DAYS
        test_end_day    = train_end_day + TEST_DAYS

        train_start = SIM_START + timedelta(days=train_start_day)
        train_end   = SIM_START + timedelta(days=train_end_day)
        test_end    = SIM_START + timedelta(days=test_end_day)

        # Pencere adı (ay bazlı)
        month_label = train_end.strftime("%b %Y")
        month_names.append(month_label)

        train = metr_df[(metr_df["date"] >= train_start) & (metr_df["date"] < train_end)].copy()
        test  = metr_df[(metr_df["date"] >= train_end)   & (metr_df["date"] < test_end)].copy()

        if len(train) < 100 or len(test) < 10:
            monthly_results.append(None)
            continue

        # Motor eğit
        engine = PredictionEngine(use_prophet=False, use_clip=False)
        engine.train(train, verbose=False)

        # Tahmin
        predictions = engine.predict(test)
        if predictions.empty:
            monthly_results.append(None)
            continue

        eval_df = predictions.merge(profiles, on="product_id", how="left")
        eval_df = eval_df.dropna(subset=["true_profile"])

        total = len(eval_df)
        if total == 0:
            monthly_results.append(None)
            continue

        # Metrikler
        r_df = eval_df[eval_df["true_profile"].isin(SHOULD_RISE)]
        f_df = eval_df[eval_df["true_profile"].isin(SHOULD_FALL)]
        viral_df = eval_df[eval_df["true_profile"] == "viral"]

        rise_rate = r_df["trend_label"].isin(["TREND","POTANSIYEL"]).mean() * 100 if len(r_df) > 0 else 0
        fall_rate = (f_df["trend_label"] == "DUSEN").mean() * 100 if len(f_df) > 0 else 0
        viral_rate = (viral_df["trend_label"] == "TREND").mean() * 100 if len(viral_df) > 0 else 0

        r_scores = eval_df[eval_df["true_profile"].isin(SHOULD_RISE)]["trend_score"].mean() if len(r_df) > 0 else 0
        f_scores = eval_df[eval_df["true_profile"].isin(SHOULD_FALL)]["trend_score"].mean() if len(f_df) > 0 else 0

        monthly_results.append({
            "month": month_label,
            "train_ürün": train["product_id"].nunique(),
            "train_kayıt": len(train),
            "test_ürün": total,
            "rise_rate": round(rise_rate, 1),
            "fall_rate": round(fall_rate, 1),
            "viral_rate": round(viral_rate, 1),
            "score_gap": round(r_scores - f_scores, 1),
            "n_rising": len(r_df),
            "n_falling": len(f_df),
            "n_viral": len(viral_df),
        })

        r_icon = "✅" if rise_rate >= 60 else ("⚠️" if rise_rate >= 40 else "❌")
        f_icon = "✅" if fall_rate >= 40 else ("⚠️" if fall_rate >= 20 else "❌")
        print(f"  [{month_label:>8}] Eğitim:{train['product_id'].nunique():>4} ürün | "
              f"Yükseliş: {r_icon}{rise_rate:>5.1f}% | "
              f"Düşüş: {f_icon}{fall_rate:>5.1f}% | "
              f"Gap: {r_scores-f_scores:>+5.1f}")

    # ── ÖZET TABLO ─────────────────────────────────────────────
    print()
    print("=" * 72)
    print("📊 ÖZET — AY AY DOĞRULUK")
    print("=" * 72)
    valid = [r for r in monthly_results if r]

    if valid:
        avg_rise = np.mean([r["rise_rate"] for r in valid])
        avg_fall = np.mean([r["fall_rate"] for r in valid])
        avg_gap  = np.mean([r["score_gap"] for r in valid])
        best_month = max(valid, key=lambda x: x["rise_rate"])
        worst_month = min(valid, key=lambda x: x["rise_rate"])

        print(f"\n  Ortalama Yükseliş Tespiti : {avg_rise:.1f}%  {'✅' if avg_rise >= 60 else '⚠️'}")
        print(f"  Ortalama Düşüş Tespiti    : {avg_fall:.1f}%  {'✅' if avg_fall >= 40 else '⚠️'}")
        print(f"  Ortalama Skor Gap         : {avg_gap:+.1f}")
        print(f"\n  En İyi Ay  : {best_month['month']} — Yükseliş {best_month['rise_rate']}%")
        print(f"  En Kötü Ay : {worst_month['month']} — Yükseliş {worst_month['rise_rate']}%")

        # Trend (öğreniyor mu zamanla?)
        rise_rates = [r["rise_rate"] for r in valid]
        first_half = np.mean(rise_rates[:len(rise_rates)//2])
        second_half = np.mean(rise_rates[len(rise_rates)//2:])
        learning_trend = second_half - first_half
        print(f"\n  Öğrenme Eğrisi:")
        print(f"    İlk 6 ay ortalaması  : {first_half:.1f}%")
        print(f"    Son 6 ay ortalaması  : {second_half:.1f}%")
        print(f"    Değişim              : {learning_trend:+.1f}% {'📈 İyileşiyor' if learning_trend > 2 else ('📉 Kötüleşiyor' if learning_trend < -2 else '→ Stabil')}")

    # ── KATEGORİ BAZLI ÖZET ─────────────────────────────────────
    print()
    print("=" * 72)
    print("📊 KATEGORİ BAZLI SON AY DOĞRULUĞU")
    print("=" * 72)

    if valid:
        # Son aya ait tam seti çek
        last = valid[-1]
        w = len(valid) - 1
        train_start = SIM_START + timedelta(days=w * STEP_DAYS)
        train_end   = SIM_START + timedelta(days=w * STEP_DAYS + TRAIN_DAYS)
        test_end    = SIM_START + timedelta(days=w * STEP_DAYS + TRAIN_DAYS + TEST_DAYS)

        train = metr_df[(metr_df["date"] >= train_start) & (metr_df["date"] < train_end)].copy()
        test  = metr_df[(metr_df["date"] >= train_end)   & (metr_df["date"] < test_end)].copy()

        engine = PredictionEngine(use_prophet=False, use_clip=False)
        engine.train(train, verbose=False)
        predictions = engine.predict(test)

        if not predictions.empty:
            eval_df = predictions.merge(profiles, on="product_id", how="left")
            eval_df = eval_df.dropna(subset=["true_profile"])

            print(f"\n  {'Kategori':<16} {'Ürün':>6} {'Rising':>8} {'Falling':>9} {'Gap':>7}")
            print("  " + "-" * 50)
            for cat in sorted(eval_df["category"].dropna().unique()):
                cat_df = eval_df[eval_df["category"] == cat]
                r = cat_df[cat_df["true_profile"].isin(SHOULD_RISE)]
                f = cat_df[cat_df["true_profile"].isin(SHOULD_FALL)]
                rr = r["trend_label"].isin(["TREND","POTANSIYEL"]).mean()*100 if len(r)>0 else 0
                fr = (f["trend_label"]=="DUSEN").mean()*100 if len(f)>0 else 0
                r_sc = r["trend_score"].mean() if len(r)>0 else 0
                f_sc = f["trend_score"].mean() if len(f)>0 else 0
                gap = r_sc - f_sc
                ri = "✅" if rr>=60 else ("⚠️" if rr>=40 else "❌")
                fi = "✅" if fr>=40 else ("⚠️" if fr>=20 else "❌")
                print(f"  {cat:<16} {len(cat_df):>6} "
                      f"{ri}{rr:>5.1f}%  {fi}{fr:>5.1f}%  {gap:>+6.1f}")

    print()
    print("═" * 72)
    print("  ✅ 1 Yıllık değerlendirme tamamlandı!")
    print("═" * 72)


if __name__ == "__main__":
    run_yearly_evaluation()
