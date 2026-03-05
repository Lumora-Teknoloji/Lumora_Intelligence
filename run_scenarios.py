# run_scenarios.py
"""
🧪 Çoklu Senaryo Testi — 4 farklı pazar koşulunu karşılaştır

Her senaryo için:
  - Yükseliş tespiti (rising/viral → TREND+POT)
  - Düşüş tespiti (falling → DUSEN)
  - Viral erken tespit
  - Composite skor sıralaması doğru mu?
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from engine.predictor import PredictionEngine
from data.scenarios import (
    generate_sezonsal_pik,
    generate_soguk_baslangic,
    generate_fiyat_baskisi,
    generate_rakip_catismasi,
)
from data.sample_data import generate_products, generate_daily_metrics

SHOULD_RISE = {"viral", "rising", "seasonal_rising", "price_shock_resilient"}
SHOULD_FALL = {"falling", "seasonal_falling", "competitor_dominated", "price_shock_sensitive"}


def evaluate_scenario(name, products_df, metrics_df, train_days=45, test_days=15):
    """Tek senaryo için doğruluk değerlendirmesi."""
    from datetime import datetime, timedelta
    now = datetime.now()
    cutoff = now - timedelta(days=test_days)

    train = metrics_df[metrics_df["date"] < cutoff].copy()
    test  = metrics_df[metrics_df["date"] >= cutoff].copy()

    if len(train) < 50:
        return {"name": name, "error": "Yetersiz eğitim verisi"}

    engine = PredictionEngine(use_prophet=False, use_clip=False)
    engine.train(train, verbose=False)

    # Kategori kolonunu düzelt
    if "category_tag" in test.columns and "category" not in test.columns:
        test["category"] = test["category_tag"]

    predictions = engine.predict(test)
    if predictions.empty:
        return {"name": name, "error": "Tahmin boş"}

    # Gerçek profilleri al
    profiles = metrics_df.groupby("product_id")["_trend_profile"].first().reset_index()
    profiles.columns = ["product_id", "true_profile"]
    eval_df = predictions.merge(profiles, on="product_id", how="left")
    eval_df = eval_df.dropna(subset=["true_profile"])

    total = len(eval_df)
    if total == 0:
        return {"name": name, "error": "Eşleşme yok"}

    # Yükseliş tespiti
    r_df = eval_df[eval_df["true_profile"].isin(SHOULD_RISE)]
    rise_rate = 0
    if len(r_df) > 0:
        caught = r_df["trend_label"].isin(["TREND", "POTANSIYEL"]).sum()
        rise_rate = caught / len(r_df) * 100

    # Düşüş tespiti
    f_df = eval_df[eval_df["true_profile"].isin(SHOULD_FALL)]
    fall_rate = 0
    if len(f_df) > 0:
        caught_fall = (f_df["trend_label"] == "DUSEN").sum()
        fall_rate = caught_fall / len(f_df) * 100

    # Viral tespit
    viral_df = eval_df[eval_df["true_profile"] == "viral"]
    viral_rate = 0
    if len(viral_df) > 0:
        caught_viral = (viral_df["trend_label"] == "TREND").sum()
        viral_rate = caught_viral / len(viral_df) * 100

    # Composite skor sıralaması doğru mu?
    # Rising ürünler falling ürünlerden yüksek skor alıyor mu?
    rising_scores = eval_df[eval_df["true_profile"].isin(SHOULD_RISE)]["trend_score"].mean()
    falling_scores = eval_df[eval_df["true_profile"].isin(SHOULD_FALL)]["trend_score"].mean()
    rank_correct = rising_scores > falling_scores if (len(r_df) > 0 and len(f_df) > 0) else None

    # Viral skor vs ortalama
    viral_scores = eval_df[eval_df["true_profile"] == "viral"]["trend_score"].mean() if len(viral_df) > 0 else 0
    avg_score    = eval_df["trend_score"].mean()

    # Kategori bazlı skor farkı (ne kadar ayrışıyor)
    score_gap = rising_scores - falling_scores if (len(r_df) > 0 and len(f_df) > 0) else 0

    return {
        "name": name,
        "total": total,
        "n_rising": len(r_df),
        "n_falling": len(f_df),
        "n_viral": len(viral_df),
        "rise_rate": round(rise_rate, 1),
        "fall_rate": round(fall_rate, 1),
        "viral_rate": round(viral_rate, 1),
        "rising_score_avg": round(rising_scores, 1) if len(r_df) > 0 else 0,
        "falling_score_avg": round(falling_scores, 1) if len(f_df) > 0 else 0,
        "viral_score_avg": round(viral_scores, 1),
        "score_gap": round(score_gap, 1),
        "rank_order_correct": rank_correct,
        "error": None,
    }


def run_all():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   🧪 LUMORA INTELLIGENCE — ÇOKLU SENARYO TESTİ             ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    scenarios = []

    # ── Baseline (mevcut veri seti) ──────────────────────────────
    print("\n[1/5] Baseline senaryosu (karma kategoriler)...")
    bp = generate_products(n=150, categories=["crop", "tayt", "grup", "kadın abiye"])
    bm = generate_daily_metrics(bp, days=60, scraper_gaps=False)
    scenarios.append(evaluate_scenario("Baseline (karma)", bp, bm))

    # ── Senaryo 1: Sezonsal Pik ──────────────────────────────────
    print("[2/5] Sezonsal pik (kış başlangıcı)...")
    sp, sm = generate_sezonsal_pik(n=100, days=60)
    scenarios.append(evaluate_scenario("Sezonsal Pik (kış)", sp, sm))

    # ── Senaryo 2: Soğuk Başlangıç ───────────────────────────────
    print("[3/5] Soğuk başlangıç (yeni kategori + eski)...")
    cp, cm = generate_soguk_baslangic(n_new=40, n_old=80, days=60)
    scenarios.append(evaluate_scenario("Soğuk Başlangıç", cp, cm))

    # ── Senaryo 3: Fiyat Baskısı ─────────────────────────────────
    print("[4/5] Fiyat/Enflasyon baskısı...")
    fp, fm = generate_fiyat_baskisi(n=120, days=60)
    scenarios.append(evaluate_scenario("Fiyat Baskısı", fp, fm))

    # ── Senaryo 4: Rakip Çatışması ───────────────────────────────
    print("[5/5] Rakip çatışması + viral patlama...")
    rp, rm = generate_rakip_catismasi(n=100, days=60)
    scenarios.append(evaluate_scenario("Rakip Çatışması", rp, rm))

    # ── SONUÇLAR ─────────────────────────────────────────────────
    print()
    print("=" * 75)
    print("📊 SENARYO KARŞILAŞTIRMA TABLOSU")
    print("=" * 75)

    header = f"{'Senaryo':<24} {'Yükseliş':>9} {'Düşüş':>7} {'Viral':>7} {'Skor Gap':>9} {'Sıra OK':>8}"
    print(header)
    print("-" * 75)

    for r in scenarios:
        if r.get("error"):
            print(f"  {r['name']:<22} ❌ {r['error']}")
            continue

        skor_ok = "✅" if r["rank_order_correct"] else "❌" if r["rank_order_correct"] is False else "—"
        viral_str = f"{r['viral_rate']:.0f}%" if r["n_viral"] > 0 else "  —"

        rise_color = "✅" if r["rise_rate"] >= 60 else ("⚠️" if r["rise_rate"] >= 40 else "❌")
        fall_color = "✅" if r["fall_rate"] >= 40 else ("⚠️" if r["fall_rate"] >= 20 else "❌")

        print(f"  {r['name']:<22} "
              f"{rise_color}{r['rise_rate']:>5.1f}%  "
              f"{fall_color}{r['fall_rate']:>4.1f}%  "
              f"{viral_str:>6}  "
              f"{r['score_gap']:>+8.1f}  "
              f"{skor_ok:>7}")

    print()
    print("DETAY — Rising vs Falling Ortalama Skor:")
    print(f"  {'Senaryo':<24} {'Rising Skor':>12} {'Falling Skor':>13} {'Viral Skor':>11}")
    print("-" * 65)
    for r in scenarios:
        if r.get("error"):
            continue
        viral_s = f"{r['viral_score_avg']:.1f}" if r["n_viral"] > 0 else "  —"
        print(f"  {r['name']:<24} {r['rising_score_avg']:>12.1f} {r['falling_score_avg']:>13.1f} {viral_s:>11}")

    print()
    print("AÇIKLAMA:")
    print("  Yükseliş: rising/viral ürünlerin TREND+POTANSIYEL etiket alma oranı")
    print("  Düşüş:    falling ürünlerin DUSEN etiket alma oranı")
    print("  Skor Gap: rising_avg - falling_avg (pozitif = doğru sıralama)")
    print("  ✅ >= 60%  ⚠️ >= 40%  ❌ < 40%")
    print()
    print("═" * 75)
    print("  ✅ Tüm senaryolar tamamlandı!")
    print("═" * 75)

    return scenarios


if __name__ == "__main__":
    run_all()
