# feedback_demo.py
"""
🔄 Feedback Loop Demonu — "Yanıldın, bu ürün az sattı"

Senaryo:
  Sistem 5 ürüne TREND dedi, sen ürettin.
  3'ü çok sattı, 2'si az sattı.
  Feedback veriyorsun → sistem öğreniyor.

Sonuç:
  - Az satan ürünlerin Kalman demand'ı düşüyor
  - Uncertainty artıyor (güven azalıyor)
  - Ensemble ağırlıkları shift oluyor
  - Bir sonraki tahmin daha ihtiyatlı oluyor
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from data.sample_data import generate_products, generate_daily_metrics
from engine.predictor import PredictionEngine


def run_feedback_demo():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   🔄 FEEDBACK LOOP DEMO — Sistem Nasıl Öğrenir?        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── 1. SİSTEMİ EĞİT ────────────────────────────────────────────
    print("\n📦 Veri üretiliyor ve motor eğitiliyor...")
    products = generate_products(n=120, categories=["crop", "tayt", "grup", "kadın abiye"])
    metrics  = generate_daily_metrics(products, days=60, scraper_gaps=False)

    engine = PredictionEngine(use_prophet=False, use_clip=False)
    engine.train(metrics, verbose=False)
    print("   ✅ Motor eğitildi")

    # ── 2. İLK TAHMİN ──────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("📊 ADIM 1: İlk tahmin (feedback öncesi)")
    print("=" * 62)

    predictions = engine.predict()
    trend_products = predictions[predictions["trend_label"] == "TREND"].head(8)

    if trend_products.empty:
        trend_products = predictions.head(8)

    print(f"\n   Sistem {len(trend_products)} ürünü TREND olarak işaretledi:")
    print(f"   {'Ürün ID':>8} {'Kategori':>12} {'Trend Skoru':>12} "
          f"{'Ensemble Demand':>16} {'Confidence':>10}")
    print("   " + "-" * 62)

    for _, row in trend_products.iterrows():
        print(f"   {int(row['product_id']):>8} {str(row.get('category','?')):>12} "
              f"{row['trend_score']:>12.1f} {int(row['ensemble_demand']):>16} "
              f"{int(row.get('confidence', 50)):>10}%")

    # ── 3. ÜRETİM KARARI ───────────────────────────────────────────
    print(f"\n{'='*62}")
    print("🏭 ADIM 2: Üretim kararı — 5 ürün ürettim")
    print("=" * 62)

    # İlk 5 TREND ürünü seç
    produced = trend_products.head(5).copy()
    np.random.seed(42)

    # Simüle gerçek satışlar: 3 iyi, 2 kötü
    actual_sales_map = {}
    scenarios = []
    for i, (_, row) in enumerate(produced.iterrows()):
        predicted = int(row["ensemble_demand"])
        pid = int(row["product_id"])
        cat = str(row.get("category", "crop"))

        if i < 3:
            # İyi satanlar: tahminin %85-110'u
            actual = max(1, int(predicted * np.random.uniform(0.85, 1.10)))
            result = "✅ İyi sattı"
            success = True
        else:
            # Kötü satanlar: tahminin %8-15'i (sistem yanıldı!)
            actual = max(1, int(predicted * np.random.uniform(0.08, 0.15)))
            result = "❌ Az sattı (sistem yanıldı)"
            success = False

        actual_sales_map[pid] = {"cat": cat, "predicted": predicted,
                                  "actual": actual, "success": success}
        scenarios.append((pid, cat, predicted, actual, result))

    print(f"\n   {'Ürün ID':>8} {'Kategori':>12} {'Tahmin':>8} "
          f"{'Gerçek Satış':>13} {'Sonuç'}")
    print("   " + "-" * 65)
    for pid, cat, pred, actual, result in scenarios:
        err_pct = abs(actual - pred) / (pred + 1e-6) * 100
        print(f"   {pid:>8} {cat:>12} {pred:>8} {actual:>13}  {result}")
    print()

    # ── 4. KALİBRASYON — FEEDBACK ÖNCESİ DURUM ───────────────────
    print("=" * 62)
    print("📈 ADIM 3: Kalman durumu FEEDBACK ÖNCESİ")
    print("=" * 62)

    before_states = {}
    for pid, info in actual_sales_map.items():
        state = engine.kalman.get_product_state(pid)
        before_states[pid] = state.copy()
        print(f"   Ürün {pid:>4}  ({info['cat']:>12})  "
              f"demand={state['demand']:>8.1f}  "
              f"velocity={state['velocity']:>7.3f}  "
              f"uncertainty={state['uncertainty']:.2f}")

    # ── 5. FEEDBACK VER ─────────────────────────────────────────────
    print()
    print("=" * 62)
    print("💬 ADIM 4: Feedback veriliyor — 'bu ürünler az sattı'")
    print("=" * 62)
    print()

    for pid, info in actual_sales_map.items():
        result = engine.feedback(
            category=info["cat"],
            actual_sales=info["actual"],
            predicted_demand=info["predicted"],
            product_id=pid
        )
        status = "✅" if info["success"] else "❌"
        err_pct = abs(info["actual"] - info["predicted"]) / (info["predicted"] + 1e-6) * 100
        print(f"   {status}  Ürün {pid} → tahmin={info['predicted']}, "
              f"gerçek={info['actual']}, "
              f"hata=%{err_pct:.0f}")

    # ── 6. FEEDBACK SONRASI DURUM ────────────────────────────────────
    print()
    print("=" * 62)
    print("📉 ADIM 5: Kalman durumu FEEDBACK SONRASI")
    print("=" * 62)

    print(f"\n   {'Ürün':>6}  {'Sonuç':>6}  "
          f"{'Demand Önce':>12} {'Demand Sonra':>13} "
          f"{'Değişim':>9} {'Uncertainty Önce':>17} {'Sonra':>7}")
    print("   " + "-" * 75)

    for pid, info in actual_sales_map.items():
        state_after = engine.kalman.get_product_state(pid)
        state_before = before_states[pid]

        demand_change = state_after["demand"] - state_before["demand"]
        sign = "📉" if demand_change < -0.5 else ("📈" if demand_change > 0.5 else "→")
        result = "✅" if info["success"] else "❌"

        print(f"   {pid:>6}  {result:>6}  "
              f"{state_before['demand']:>12.1f} {state_after['demand']:>13.1f} "
              f" {sign}{demand_change:>+7.1f}  "
              f"{state_before['uncertainty']:>17.2f} {state_after['uncertainty']:>7.2f}")

    # ── 7. ENSEMBLE AĞIRLIK DEĞİŞİMİ ──────────────────────────────
    print()
    print("=" * 62)
    print("⚖️  ADIM 6: Ensemble ağırlık değişimi")
    print("=" * 62)

    status = engine.status()
    weights = status["ensemble_weights"]
    fb_count = status["feedback_count"]

    print(f"\n   Toplam feedback verilen: {fb_count}")
    print(f"\n   Güncel ağırlıklar:")
    print(f"   CatBoost       : {weights['catboost']:.2f}  "
          f"({'azaldı ⬇' if weights['catboost'] < 0.60 else 'sabit →'})")
    print(f"   Kalman (ürün)  : {weights['kalman_product']:.2f}  "
          f"({'arttı ⬆' if weights['kalman_product'] > 0.24 else 'sabit →'})")
    print(f"   Kalman (kategori): {weights['kalman_category']:.2f}")

    # ── 8. YENİDEN TAHMİN — Feedback sonrası ───────────────────────
    print()
    print("=" * 62)
    print("🔮 ADIM 7: YENİDEN TAHMİN (feedback sonrası, yeniden eğitmeden)")
    print("=" * 62)

    predictions_after = engine.predict()

    print(f"\n   Önceki TREND ürünlerinin yeni skorları:")
    print(f"\n   {'Ürün ID':>8} {'Kategori':>12} {'Skor Önce':>10} "
          f"{'Skor Sonra':>11} {'Değişim':>9} {'Yeni Etiket':>12}")
    print("   " + "-" * 68)

    for _, old_row in trend_products.iterrows():
        pid = int(old_row["product_id"])
        old_score = old_row["trend_score"]
        new_row = predictions_after[predictions_after["product_id"] == pid]
        if new_row.empty:
            continue
        new_score = new_row.iloc[0]["trend_score"]
        new_label = new_row.iloc[0]["trend_label"]
        delta = new_score - old_score
        arrow = "📉" if delta < -2 else ("📈" if delta > 2 else "→")
        success = actual_sales_map.get(pid, {}).get("success", None)
        result = "✅" if success else ("❌" if success is False else "  ")
        print(f"   {pid:>8} {str(old_row.get('category','?')):>12} "
              f"{old_score:>10.1f} {new_score:>11.1f} "
              f"{arrow}{delta:>+7.1f}   {new_label:>12}   {result}")

    # ── 9. ÖZET ─────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("📋 ÖZET — Feedback Loop Ne Sağladı?")
    print("=" * 62)

    poor_products = [pid for pid, info in actual_sales_map.items() if not info["success"]]
    good_products = [pid for pid, info in actual_sales_map.items() if info["success"]]

    poor_demand_change = []
    good_demand_change = []
    for pid in poor_products:
        b = before_states[pid]["demand"]
        a = engine.kalman.get_product_state(pid)["demand"]
        poor_demand_change.append(a - b)
    for pid in good_products:
        b = before_states[pid]["demand"]
        a = engine.kalman.get_product_state(pid)["demand"]
        good_demand_change.append(a - b)

    avg_poor_drop = np.mean(poor_demand_change) if poor_demand_change else 0
    avg_good_rise = np.mean(good_demand_change) if good_demand_change else 0

    print(f"""
   Az satan ürünler (❌):
     → Kalman demand ort. değişim: {avg_poor_drop:+.1f}
     → Uncertainty arttı (sistem bunlara daha az güveniyor)
     → Bir sonraki tahmin: daha düşük demand, daha ihtiyatlı

   İyi satan ürünler (✅):
     → Kalman demand ort. değişim: {avg_good_rise:+.1f}
     → Uncertainty düştü (sistem bunlara daha çok güveniyor)
     → Bir sonraki tahmin: güvenle yüksek demand

   Uzun vadede (haftalar sonra):
     → CatBoost bu gerçek satışlarla yeniden eğitilir
     → Hangi feature'ların yanıltıcı olduğunu öğrenir
     → Benzer ürünleri gelecekte daha doğru değerlendirir

   NOT: Bu tüm bunlar YENİDEN EĞİTİM OLMADAN gerçekleşti.
        Sadece Kalman online güncelleme ile öğrendi.
    """)

    print("═" * 62)
    print("  ✅ Feedback loop demosu tamamlandı!")
    print("═" * 62)


if __name__ == "__main__":
    run_feedback_demo()
