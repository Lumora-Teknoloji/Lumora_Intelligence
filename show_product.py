import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import pandas as pd
from data.sample_data import generate_products, generate_daily_metrics

products = generate_products(n=20, categories=["crop", "tayt", "grup"])
metrics  = generate_daily_metrics(products, days=45, scraper_gaps=False)

rising = metrics[metrics["_trend_profile"] == "rising"]["product_id"].unique()
pid = int(rising[0]) if len(rising) else int(metrics["product_id"].iloc[0])

sample = metrics[metrics["product_id"] == pid].sort_values("date").reset_index(drop=True)
row0 = sample.iloc[0]

# CSV olarak kaydet (Excel/VS Code'da acilabilir)
sample.to_csv("urun_verileri.csv", index=False, encoding="utf-8-sig")

# Terminale ozet yaz
print()
print("=" * 80)
print("URUN:", pid, "| Profil:", row0["_trend_profile"].upper(),
      "| Kategori:", row0["category"], "| Marka:", row0["brand"])
print("Toplam:", len(sample), "gunluk kayit")
print("=" * 80)
print()

# Satir satir onemli metrikler
header = f"{'Tarih':<12} {'Rank':>6} {'Sayfa':>5} {'Favori':>9} {'Sepet':>7} {'Goruntuleme':>12} {'Fiyat':>8} {'Indirim':>8} {'Rating':>7} {'YorumSay':>9} {'Mult':>6}"
print(header)
print("-" * len(header))
for _, row in sample.iterrows():
    fav = int(row["favorite_count"])
    cart = int(row["cart_count"])
    view = int(row["view_count"]) if row["view_count"] else 0
    print(f"{row['date']:<12} {int(row['absolute_rank']):>6} {int(row['page_number']):>5} {fav:>9,} {cart:>7,} {view:>12,} {row['price']:>8.0f} {row['discount_rate']:>8.1f} {row['rating']:>7.2f} {int(row['rating_count']):>9,} {row['_true_mult']:>6.2f}")

print()
print("-" * 80)
print("OZET:")
print("  Rank degisimi :", int(sample["absolute_rank"].iloc[0]), "->",
      int(sample["absolute_rank"].iloc[-1]), "(kucuk = iyi)")
print("  Favori degisim:", f"{int(sample['favorite_count'].iloc[0]):,}", "->",
      f"{int(sample['favorite_count'].iloc[-1]):,}")
print("  Sepet toplam  :", f"{int(sample['cart_count'].sum()):,}")
print("  Ort fiyat     :", f"{sample['price'].mean():.0f} TL")
print("  Ort rating    :", f"{sample['rating'].mean():.2f}", "|", int(sample["rating_count"].max()), "yorum")
print("  Ort mult      :", f"{sample['_true_mult'].mean():.3f}", "(engagement carpani)")
print()
print("CSV kaydedildi: urun_verileri.csv")
print("-" * 80)
