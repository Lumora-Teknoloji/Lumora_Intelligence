# main.py
"""
🧠 Trend Tahmin Motoru — Ana Çalıştırıcı
8 algoritmalı, kendini güncelleyen tahmin sistemi.

Kullanım:
    python main.py              → Tam demo (test verisiyle)
    python main.py --no-prophet → Prophet olmadan
    python main.py --no-clip    → CLIP olmadan
"""
import sys
import os

# Proje kök dizinini ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from data.sample_data import generate_products, generate_daily_metrics, generate_inventory
from engine.predictor import PredictionEngine


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   🧠 SELF-LEARNING TREND TAHMİN MOTORU                 ║")
    print("║   8 Algoritma · 6 Katman · Otomatik Güncelleme         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # ─── 1. VERİ OLUŞTUR ────────────────────────────────────────
    print("📦 1. Test verisi oluşturuluyor...")
    products = generate_products(
        n=100,
        categories=["crop", "tayt", "grup", "kadın abiye"]
    )
    metrics = generate_daily_metrics(products, days=60, scraper_gaps=True)
    inventory = generate_inventory()

    print(f"   Ürünler: {len(products)}")
    print(f"   Metrikler: {len(metrics)} kayıt ({len(products)} ürün × 60 gün)")
    print(f"   Envanter: {len(inventory)} malzeme")
    print()

    # ─── 2. MOTORU BAŞLAT ve EĞİT ──────────────────────────────
    use_prophet = "--no-prophet" not in sys.argv
    use_clip = "--no-clip" not in sys.argv and "--fast" not in sys.argv

    engine = PredictionEngine(use_prophet=use_prophet, use_clip=use_clip)
    engine.train(metrics, verbose=True)
    print()

    # ─── 3. TAHMİN YAP ─────────────────────────────────────────
    print("=" * 60)
    print("🔮 TAHMİN SONUÇLARI")
    print("=" * 60)
    predictions = engine.predict()

    if predictions.empty:
        print("  ❌ Tahmin yapılamadı!")
        return

    print(predictions.to_string(index=False, max_rows=15))
    print()

    # ─── 4. ENVANTER EŞLEŞTİRME ────────────────────────────────
    print("=" * 60)
    print("🏭 ENVANTER EŞLEŞTİRME")
    print("=" * 60)

    for _, inv in inventory.iterrows():
        material = inv["material"]
        color = inv.get("color")
        qty = inv["quantity_kg"]

        print(f"\n  📦 {qty} kg {color or ''} {material}:")

        result = engine.predict_for_inventory(material, color)

        if result.get("best_option"):
            best = result["best_option"]
            print(f"     ➤ En iyi: {best['category']} (tahmini talep: {best['predicted_demand']} adet)")
            top3 = result["recommendations"][:3]
            for r in top3:
                kalman = r.get("kalman_state", {})
                vel = kalman.get("velocity", 0)
                arrow = "↑" if vel > 0 else "↓" if vel < 0 else "→"
                print(f"       - {r['category']}: {r['predicted_demand']} adet {arrow}")

    # ─── 5. FEEDBACK LOOP DEMOsu ────────────────────────────────
    print()
    print("=" * 60)
    print("🔄 FEEDBACK LOOP DEMO")
    print("=" * 60)

    # Simüle: bir kategori için tahmin vs gerçek satış
    sample_cats = predictions["category"].unique()[:3]

    for cat in sample_cats:
        cat_pred = predictions[predictions["category"] == cat]
        if cat_pred.empty:
            continue

        predicted = cat_pred["ensemble_demand"].mean()
        actual = int(predicted * np.random.uniform(0.7, 1.3))  # Simüle gerçek satış

        engine.feedback(cat, actual, int(predicted))

    # Feedback sonrası yeni tahmin
    print("\n  📊 Feedback sonrası Kalman durumu:")
    states = engine.kalman.get_all_states()
    for cat, state in list(states.items())[:5]:
        arrow = "↑" if state["velocity"] > 0 else "↓" if state["velocity"] < 0 else "→"
        print(f"    {cat}: demand={state['demand']:.1f} {arrow} "
              f"(velocity={state['velocity']:.2f}, uncertainty={state['uncertainty']:.1f})")

    # ─── 6. CLIP DEMO (opsiyonel) ───────────────────────────────
    if use_clip:
        print()
        print("=" * 60)
        print("👁️ CLIP GÖRSEL EŞLEŞTİRME DEMO")
        print("=" * 60)

        try:
            from algorithms.clip_model import VisualMatcher
            clip = VisualMatcher()

            product_names = products["name"].tolist()[:10]
            query = "siyah pamuk oversize elbise"

            print(f"\n  Sorgu: \"{query}\"")
            matches = clip.demo_without_images(product_names, query)

            for m in matches[:5]:
                pct = int(m["similarity"] * 100)
                bar = "█" * (pct // 5)
                print(f"    {pct}% {bar} {m['product_name']}")

        except Exception as e:
            print(f"  ⚠ CLIP kullanılamadı: {e}")
            print(f"    → pip install open-clip-torch torch")

    # ─── 7. MOTOR DURUMU ────────────────────────────────────────
    print()
    print("=" * 60)
    print("📋 MOTOR DURUMU")
    print("=" * 60)

    status = engine.status()
    print(f"  CatBoost eğitildi: {'✅' if status['catboost_trained'] else '❌'}")
    print(f"  Prophet aktif: {'✅' if status['prophet_enabled'] else '❌'}")
    print(f"  CLIP aktif: {'✅' if status['clip_enabled'] else '❌'}")
    kalman_states = status.get('kalman_category_states', status.get('kalman_states', {}))
    print(f"  Kalman kategorileri: {len(kalman_states)}")

    if status["catboost_features"]:
        print(f"\n  🏆 Feature Importance (Top 5):")
        for i, (feat, imp) in enumerate(list(status["catboost_features"].items())[:5]):
            bar = "█" * int(imp / 2)
            print(f"    {i+1}. {feat}: {imp}% {bar}")

    if status["cluster_profiles"]:
        star = engine.clusterer.get_star_profile()
        if star:
            print(f"\n  ⭐ Yıldız Kümesi Profili:")
            for k, v in star.items():
                if k != "cluster_id":
                    print(f"    {k}: {v}")

    print()
    print("═" * 60)
    print("  ✅ Demo tamamlandı!")
    print("═" * 60)


if __name__ == "__main__":
    main()
