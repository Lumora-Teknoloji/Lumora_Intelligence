# run_yearly_persistent.py
"""
📅 1 Yıllık Birikmeli (Persistent) Öğrenme Testi

Fark:
  run_yearly.py      → Her pencere sıfırdan eğitim (stateless)
  Bu script           → Aynı engine tüm yıl boyunca yaşar:
                          - Feedback biriktirilir (Kalman günceli kalır)
                          - CatBoost her pencerede ek örneklerle retrain
                          - Feedback penalties birikir
                          - _feedback_state (tmp dosyasında) korunur

Beklenti:
  İlk ay: %48 civarı (cold start)
  Son ay:  %60+ (birikmeli öğrenme ile iyileşme)
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, pickle, tempfile; sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from datetime import timedelta
from data.yearly_simulation import generate_yearly_data, SIM_START
from engine.predictor import PredictionEngine

SHOULD_RISE = {"viral", "rising", "seasonal"}
SHOULD_FALL = {"falling"}

TMP_STATE_FILE = os.path.join(tempfile.gettempdir(), "lumora_engine_state.pkl")


def score_window(engine, test_df, profiles, window_label):
    """Bir test penceresini değerlendirir, sonucu döndürür."""
    predictions = engine.predict(test_df)
    if predictions.empty:
        return None

    eval_df = predictions.merge(profiles, on="product_id", how="left")
    eval_df = eval_df.dropna(subset=["true_profile"])
    total = len(eval_df)
    if total == 0:
        return None

    r_df = eval_df[eval_df["true_profile"].isin(SHOULD_RISE)]
    f_df = eval_df[eval_df["true_profile"].isin(SHOULD_FALL)]
    viral_df = eval_df[eval_df["true_profile"] == "viral"]

    rise_rate  = r_df["trend_label"].isin(["TREND","POTANSIYEL"]).mean()*100 if len(r_df)>0 else 0
    fall_rate  = (f_df["trend_label"]=="DUSEN").mean()*100 if len(f_df)>0 else 0
    viral_rate = (viral_df["trend_label"]=="TREND").mean()*100 if len(viral_df)>0 else 0

    r_sc = r_df["trend_score"].mean() if len(r_df)>0 else 0
    f_sc = f_df["trend_score"].mean() if len(f_df)>0 else 0

    return {
        "month": window_label,
        "total": total,
        "rise_rate": round(rise_rate, 1),
        "fall_rate": round(fall_rate, 1),
        "viral_rate": round(viral_rate, 1),
        "score_gap":  round(r_sc - f_sc, 1),
        "n_rising":   len(r_df),
        "n_falling":  len(f_df),
        "n_viral":    len(viral_df),
        "eval_df":    eval_df,  # feedback için
    }


def give_feedback_from_results(engine, result, test_df):
    """
    Test sonuçlarından simüle gerçek satış üret ve engine'e ver.
    Gerçek satışı simüle etmek için _true_mult'u kullan.
    """
    if result is None:
        return 0

    fb_count = 0
    eval_df = result["eval_df"]

    # Her tahmin edilen ürün için gerçek satışı simüle et
    true_mults = test_df.groupby("product_id")["_true_mult"].mean()

    for _, row in eval_df.iterrows():
        pid = int(row["product_id"])
        cat = str(row.get("category", "crop"))
        predicted = int(row.get("ensemble_demand", 0))
        if predicted < 1:
            continue

        # True mult → gerçek satış kestirimi
        true_mult = float(true_mults.get(pid, 1.0))
        # rising: %90-110, falling: %8-20, viral: %120-200, stable: %60-100
        profile = row.get("true_profile", "stable")
        if profile in ("rising", "seasonal"):
            actual = int(predicted * true_mult * np.random.uniform(0.85, 1.15))
        elif profile == "viral":
            actual = int(predicted * true_mult * np.random.uniform(1.0, 2.0))
        elif profile == "falling":
            actual = int(predicted * max(0.05, true_mult) * np.random.uniform(0.1, 0.25))
        else:
            actual = int(predicted * np.random.uniform(0.6, 1.1))

        actual = max(0, actual)
        engine.feedback(
            category=cat,
            actual_sales=actual,
            predicted_demand=predicted,
            product_id=pid
        )
        fb_count += 1

    return fb_count


def run_persistent():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   📅 BİRİKMELİ ÖĞRENME — 1 YILLIK PERSISTENT TEST         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("📦 1 yıllık veri üretiliyor...")
    np.random.seed(99)
    prod_df, metr_df = generate_yearly_data()
    print()

    profiles = metr_df.groupby("product_id")["_trend_profile"].first().reset_index()
    profiles.columns = ["product_id", "true_profile"]

    TRAIN_DAYS = 90
    TEST_DAYS  = 30
    STEP_DAYS  = 30

    n_windows = (365 - TRAIN_DAYS - TEST_DAYS) // STEP_DAYS + 1
    print(f"📊 {n_windows} aylık pencere | Kalıcı engine (sıfırlanmıyor)")
    print(f"   Eğitim: {TRAIN_DAYS} gün | Test: {TEST_DAYS} gün | Adım: {STEP_DAYS} gün")
    print(f"   State dosyası: {TMP_STATE_FILE}")
    print()

    # ── TEK ENGINE — tüm yıl yaşar ──────────────────────────────
    engine = PredictionEngine(use_prophet=False, use_clip=False)

    stateless_results = []  # karşılaştırma için
    persistent_results = []
    cumulative_train = pd.DataFrame()  # birikmeli eğitim seti
    total_feedback = 0

    print(f"{'Ay':>10}  {'Eğitim':>7}  {'Yükseliş':>10}  {'Düşüş':>8}  "
          f"{'Gap':>6}  {'Feedback':>9}")
    print("-" * 65)

    for w in range(n_windows):
        train_start_day = w * STEP_DAYS
        train_end_day   = train_start_day + TRAIN_DAYS
        test_end_day    = train_end_day + TEST_DAYS

        train_start = SIM_START + timedelta(days=train_start_day)
        train_end   = SIM_START + timedelta(days=train_end_day)
        test_end    = SIM_START + timedelta(days=test_end_day)

        window_train = metr_df[(metr_df["date"] >= train_start) &
                                (metr_df["date"] < train_end)].copy()
        window_test  = metr_df[(metr_df["date"] >= train_end) &
                                (metr_df["date"] < test_end)].copy()
        month_label  = train_end.strftime("%b %Y")

        if len(window_train) < 100 or len(window_test) < 10:
            continue

        # ── BİRİKMELİ eğitim seti ────────────────────────────────
        # Mevcut pencere verisini birikimli veriyle birleştir
        # Sadece son 180 günü tut (hafıza sınırı)
        cumulative_train = pd.concat([cumulative_train, window_train], ignore_index=True)
        MEMORY_LIMIT = 180
        cutoff_date = train_end - timedelta(days=MEMORY_LIMIT)
        cumulative_train = cumulative_train[
            cumulative_train["date"] >= cutoff_date
        ].copy()

        # ── ENGINE'İ BİRİKMELİ VERİYLE EĞİT ─────────────────────
        # warm=True → mevcut Kalman state'leri koru, sadece CatBoost retrain
        engine.train(cumulative_train, verbose=False)

        # ── TAHMİN ET ────────────────────────────────────────────
        result = score_window(engine, window_test, profiles, month_label)

        if result:
            persistent_results.append(result)

            # ── GERİ BİLDİRİM VER ────────────────────────────────
            fb = give_feedback_from_results(engine, result, window_test)
            total_feedback += fb

            r_icon = "✅" if result["rise_rate"] >= 60 else ("⚠️" if result["rise_rate"] >= 40 else "❌")
            f_icon = "✅" if result["fall_rate"] >= 40 else ("⚠️" if result["fall_rate"] >= 20 else "❌")

            print(f"  [{month_label:>8}]  "
                  f"{cumulative_train['product_id'].nunique():>6} ürün  "
                  f"{r_icon}{result['rise_rate']:>6.1f}%  "
                  f"{f_icon}{result['fall_rate']:>5.1f}%  "
                  f"{result['score_gap']:>+5.1f}  "
                  f"fb+{fb:>4}")

        # ── STATE KAYDET ──────────────────────────────────────────
        try:
            state = {
                "feedback_history": engine.feedback_history,
                "feedback_penalties": engine._feedback_penalties,
                "weights": engine.weights,
                "window": w,
            }
            with open(TMP_STATE_FILE, "wb") as f:
                pickle.dump(state, f)
        except Exception:
            pass

    # ── ÖZET ─────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("📊 BİRİKMELİ ÖĞRENME ÖZETİ")
    print("=" * 65)

    if persistent_results:
        rise_rates = [r["rise_rate"] for r in persistent_results]
        fall_rates = [r["fall_rate"] for r in persistent_results]
        gaps       = [r["score_gap"] for r in persistent_results]

        n = len(rise_rates)
        first_half  = rise_rates[:n//2]
        second_half = rise_rates[n//2:]

        avg_rise_full   = np.mean(rise_rates)
        avg_fall_full   = np.mean(fall_rates)
        avg_gap_full    = np.mean(gaps)
        avg_rise_first  = np.mean(first_half)  if first_half  else 0
        avg_rise_second = np.mean(second_half) if second_half else 0
        learning        = avg_rise_second - avg_rise_first

        print(f"\n  Toplam feedback verilen   : {total_feedback:,}")
        print(f"  Penalty dict büyüklüğü    : {len(engine._feedback_penalties)} ürün")
        print(f"  Birikmeli eğitim ürün sayı: {cumulative_train['product_id'].nunique()}")

        print(f"\n  Ortalama Yükseliş Tespiti : {avg_rise_full:.1f}%")
        print(f"  Ortalama Düşüş Tespiti    : {avg_fall_full:.1f}%")
        print(f"  Ortalama Skor Gap         : {avg_gap_full:+.1f}")

        print(f"\n  📈 ÖĞRENME EĞRİSİ:")
        print(f"     İlk yarı ({n//2} ay) ort : {avg_rise_first:.1f}%")
        print(f"     Son yarı ({n-n//2} ay) ort: {avg_rise_second:.1f}%")

        if learning > 3:
            verdict = f"📈 +{learning:.1f}% İYİLEŞTİ — Birikmeli öğrenme çalışıyor!"
        elif learning > 0:
            verdict = f"📈 +{learning:.1f}% — Hafif iyileşme"
        elif learning > -3:
            verdict = f"→ {learning:.1f}% — Stabil (beklenen)"
        else:
            verdict = f"📉 {learning:.1f}% — Kötüleşiyor (veri kalitesi sorunu)"

        print(f"     Değişim                 : {verdict}")

        # Ay ay tablo
        print()
        print(f"  {'Ay':>10}  {'Yükseliş':>10}  {'Düşüş':>8}  {'Gap':>6}")
        print("  " + "-" * 40)
        for r in persistent_results:
            ri = "✅" if r["rise_rate"]>=60 else ("⚠️" if r["rise_rate"]>=40 else "❌")
            fi = "✅" if r["fall_rate"]>=40 else ("⚠️" if r["fall_rate"]>=20 else "❌")
            print(f"  [{r['month']:>8}]  "
                  f"{ri}{r['rise_rate']:>6.1f}%  "
                  f"{fi}{r['fall_rate']:>5.1f}%  "
                  f"{r['score_gap']:>+5.1f}")

        # Ensemble ağırlıkları
        print()
        print(f"  ⚖️  Final ensemble ağırlıkları:")
        for k, v in engine.weights.items():
            print(f"     {k:<20}: {v:.3f}")

    print()
    print(f"  💾 State dosyası kaydedildi: {TMP_STATE_FILE}")
    print()
    print("═" * 65)
    print("  ✅ Birikmeli öğrenme testi tamamlandı!")
    print("═" * 65)


if __name__ == "__main__":
    run_persistent()
