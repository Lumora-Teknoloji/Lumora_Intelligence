import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.insert(0, ".")
import pandas as pd, numpy as np
from data.sample_data import generate_products, generate_daily_metrics

products = generate_products(n=200, categories=["crop","tayt","grup","kadın abiye"])
metrics  = generate_daily_metrics(products, days=60, scraper_gaps=False)

total_products = metrics["product_id"].nunique()

print("=" * 65)
print("  VERİ DERİNLİĞİ VE KALİTE ANALİZİ")
print("=" * 65)

print(f"\nToplam kayıt : {len(metrics):,}")
print(f"Toplam ürün  : {total_products}")
print(f"Toplam gün   : {metrics['date'].nunique()}")

print()
print(f"{'Kategori':<20} {'Ürün':>6} {'Kayıt':>8} {'Ort.Gün':>8} {'Min.Gün':>8} {'Max.Gün':>8}")
print("-" * 65)
for cat in sorted(metrics["category"].unique()):
    cd = metrics[metrics["category"] == cat]
    days_per = cd.groupby("product_id")["date"].nunique()
    print(f"{cat:<20} {cd['product_id'].nunique():>6} {len(cd):>8} "
          f"{days_per.mean():>8.1f} {days_per.min():>8} {days_per.max():>8}")

print()
print("MODEL EĞİTİMİ İÇİN MİNİMUM GEREKSİNİMLER:")
print("  CatBoost eğitimi  : min 7 gün/ürün, min 30 ürün toplam")
print("  Kalman başlatma   : min 3 gün/ürün")
print("  Prophet           : min 30 gün/ürün, min 60 ürün")

days_per_product = metrics.groupby("product_id")["date"].nunique()
eligible_catboost = (days_per_product >= 7).sum()
eligible_kalman   = (days_per_product >= 3).sum()
eligible_prophet  = (days_per_product >= 30).sum()

print()
print(f"  CatBoost eğitilebilir: {eligible_catboost}/{total_products} ({eligible_catboost/total_products*100:.0f}%) {'✅' if eligible_catboost >= 30 else '❌'}")
print(f"  Kalman başlatılabilir: {eligible_kalman}/{total_products} ({eligible_kalman/total_products*100:.0f}%) {'✅' if eligible_kalman >= 10 else '❌'}")
print(f"  Prophet çalışabilir:   {eligible_prophet}/{total_products} ({eligible_prophet/total_products*100:.0f}%) {'✅' if eligible_prophet >= 60 else '❌'}")

print()
print("GÜN DAĞILIMI (ürün başına kaç gün var):")
for threshold in [3, 7, 14, 30, 45, 60]:
    count = (days_per_product >= threshold).sum()
    bar = "█" * int(count / total_products * 30)
    status = "✅" if count/total_products > 0.8 else ("⚠️" if count/total_products > 0.5 else "❌")
    print(f"  >= {threshold:2d} gün: {count:4d}/{total_products} ({count/total_products*100:5.1f}%) {status}  {bar}")

print()
print("KATEGORİ BAŞINA TREND PROFİL DAĞILIMI:")
for cat in sorted(metrics["category"].unique()):
    cd = metrics[metrics["category"] == cat]
    profiles = cd.groupby("product_id")["_trend_profile"].first().value_counts()
    print(f"  {cat}:")
    for prof, cnt in profiles.items():
        bar = "█" * cnt
        print(f"    {prof:<25}: {cnt:3d}  {bar}")

print()
print("BEKLENEN vs GERÇEK ÜRÜN SAYISI:")
expected = {"crop": 6449, "tayt": 3832, "grup": 2902, "kadın abiye": 169}
for cat in sorted(metrics["category"].unique()):
    actual_n = metrics[metrics["category"] == cat]["product_id"].nunique()
    exp = expected.get(cat, "?")
    ratio = actual_n / exp * 100 if isinstance(exp, int) else 0
    print(f"  {cat:<20}: Test={actual_n:4d}  Gerçek={exp}  "
          f"Oran=%{ratio:.1f} {'✅' if ratio > 1 else '⚠️ Az'}")

print()
print("SONUÇ:")
n60 = (days_per_product == 60).sum()
n45 = (days_per_product >= 45).sum()
n30 = (days_per_product >= 30).sum()
n7  = (days_per_product >= 7).sum()
print(f"  60 günlük ürünler: {n60} ({n60/total_products*100:.1f}%)")
print(f"  45+ gün:           {n45} ({n45/total_products*100:.1f}%)")
print(f"  30+ gün:           {n30} ({n30/total_products*100:.1f}%)")
print(f"   7+ gün:           {n7}  ({n7/total_products*100:.1f}%)")
