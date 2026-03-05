# demo_gercek_kullanim.py
"""
🎯 Gerçek Kullanım Senaryosu

Sen:  "Bu hafta hangi 5 ürünü üretmeliyim?"
Sistem: Top 5 TREND ürünü söyler

Sen: Onları üretirsin, sattıktan sonra:
  "Ürün A 2 sattı, B 45 sattı, C 8 sattı, D 30 sattı, E 3 sattı"

Sistem:
  - Az satan ürünleri → SAHTE TREND → bir daha TREND demez
  - Çok satanları → güven artar, benzer ürünlere daha fazla şans
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
from data.sample_data import generate_products, generate_daily_metrics
from engine.predictor import PredictionEngine

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   🎯 GERÇEK KULLANIM — Top 5 Trend + Satış Feedback    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Motor eğit
    print("\n📦 Motor eğitiliyor...")
    products = generate_products(n=150, categories=["crop","tayt","grup","kadın abiye"])
    metrics  = generate_daily_metrics(products, days=60, scraper_gaps=False)
    engine   = PredictionEngine(use_prophet=False, use_clip=False)
    engine.train(metrics, verbose=False)
    print("   ✅ Hazır\n")

    # ── HAFTA 1 ─────────────────────────────────────────────────
    print("=" * 58)
    print("📅 HAFTA 1 — Bu haftanın top 5 trend ürünü:")
    print("=" * 58)

    predictions = engine.predict()
    top5 = predictions.head(5)

    print(f"\n   {'Sıra':>4} {'Ürün ID':>8} {'Kategori':>12} "
          f"{'Trend Skoru':>12} {'Tahmin Talep':>13}")
    print("   " + "-" * 53)
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"   {rank:>4} {int(row['product_id']):>8} "
              f"{str(row.get('category','?')):>12} "
              f"{row['trend_score']:>12.1f} "
              f"{int(row['ensemble_demand']):>13} adet")

    # Gerçek satışları simüle et (kullanıcı bunları giriyor)
    top5_ids = [int(r["product_id"]) for _, r in top5.iterrows()]
    # Senaryo: 1. ve 3. ürün az sattı (sahte trend)
    simulated_sales = {
        top5_ids[0]: 45,   # ✅ iyi sattı
        top5_ids[1]: 2,    # ❌ çok az sattı (sahte trend)
        top5_ids[2]: 30,   # ✅ iyi sattı
        top5_ids[3]: 3,    # ❌ çok az sattı (sahte trend)
        top5_ids[4]: 18,   # ✅ kabul edilebilir
    }

    print(f"\n   Gerçek satışları giriyorsunuz:")
    for pid, sold in simulated_sales.items():
        icon = "✅" if sold >= 5 else "❌"
        print(f"   {icon} Ürün {pid:>4}: {sold} adet sattı")

    # ── FEEDBACK VER ─────────────────────────────────────────────
    print()
    print("=" * 58)
    print("💬 Feedback veriliyor...")
    print("=" * 58)

    result = engine.feedback_top_n(simulated_sales, fake_trend_threshold=5)

    print(f"\n   Toplam test: {result['total_tested']} ürün")
    print(f"   Gerçek trend: {result['real_trends']} ✅")
    print(f"   Sahte trend:  {result['fake_trends']} ❌\n")

    for d in result["details"]:
        print(f"   Ürün {d['product_id']:>4}  →  {d['verdict']}")
        print(f"            Tahmin: {d['predicted']} | Gerçek: {d['actual_sold']}")

    # ── HAFTA 2 ─────────────────────────────────────────────────
    print()
    print("=" * 58)
    print("📅 HAFTA 2 — Feedback sonrası yeni top 5:")
    print("=" * 58)

    predictions2 = engine.predict()
    top5_2 = predictions2.head(5)
    fake_ids = set(result["fake_trend_ids"])

    print(f"\n   {'Sıra':>4} {'Ürün ID':>8} {'Kategori':>12} "
          f"{'Skor':>6} {'Durum':>25}")
    print("   " + "-" * 60)
    for rank, (_, row) in enumerate(top5_2.iterrows(), 1):
        pid = int(row["product_id"])
        durum = "⚠️ Cezalı ürün!" if pid in fake_ids else "🆕 Yeni öneri"
        print(f"   {rank:>4} {pid:>8} "
              f"{str(row.get('category','?')):>12} "
              f"{row['trend_score']:>6.1f} "
              f"  {durum}")

    in_both = fake_ids & set(int(r["product_id"]) for _, r in top5_2.iterrows())
    if in_both:
        print(f"\n   ⚠️  {len(in_both)} sahte trend ürünü hâlâ top 5'te — "
              f"ceza henüz eşiği geçemedi.")
    else:
        print(f"\n   ✅ Sahte trend ürünleri top 5'ten çıktı — sistem öğrendi!")

    print()
    print("  📌 Özet:")
    print("  Sahte trend tespit edildiğinde sistem o ürünü cezalandırır.")
    print("  Aynı ürün bir sonraki hafta daha düşük skor alır.")
    print("  Birkaç haftada gerçek trend ürünleri öne çıkar.")
    print()
    print("  API kullanım özeti:")
    print("  ─────────────────────────────────────────────────────")
    print("  predictions = engine.predict()          # Bu haftanın listesi")
    print("  top5 = predictions.head(5)              # Top 5 al")
    print("  engine.feedback_top_n({                 # Satışları gir")
    print("      pid_A: 45,   # iyi sattı")
    print("      pid_B: 2,    # az sattı → sahte trend")
    print("  })")
    print()
    print("═" * 58)
    print("  ✅ Demo tamamlandı!")
    print("═" * 58)

if __name__ == "__main__":
    main()
