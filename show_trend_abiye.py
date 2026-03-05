import sys, io, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, ".")
import pandas as pd
import json

from data.sample_data import generate_products, generate_daily_metrics
from engine.predictor import PredictionEngine

# Sadece kadin abiye kategorisi
products = generate_products(n=60, categories=["kadın abiye"])
metrics  = generate_daily_metrics(products, days=60, scraper_gaps=False)

# Egit
engine = PredictionEngine(use_prophet=False, use_clip=False)
engine.train(metrics, verbose=False)

# Son 15 gun test
from datetime import datetime, timedelta
cutoff = datetime.now() - timedelta(days=15)
test = metrics[metrics["date"] >= cutoff].copy()
preds = engine.predict(test)

# TREND ve POTANSIYEL olanlar
trends = preds[preds["trend_label"].isin(["TREND", "POTANSIYEL"])].sort_values(
    ["trend_label", "trend_score"], ascending=[True, False]
)

if trends.empty:
    trends = preds.sort_values("trend_score", ascending=False)

top = trends.iloc[0]
pid = int(top["product_id"])

# O urunun tam verisini cek
urun_meta = products[products["product_id"] == pid].iloc[0]
son_gun   = metrics[metrics["product_id"] == pid].sort_values("date").iloc[-1]

attrs = urun_meta.get("attributes", {})
if isinstance(attrs, str):
    try:
        attrs = json.loads(attrs)
    except:
        attrs = {}

print()
print("=" * 70)
print("  EN TREND ABİYE")
print("=" * 70)
print(f"  Etiket    : {top['trend_label']}")
print(f"  Trend Skoru: {top['trend_score']:.1f} / 100")
print(f"  Guven     : %{top.get('confidence', 0)}")
print()
print(f"  Marka     : {urun_meta['brand']}")
print(f"  Talep Tahmini: ~{top.get('ensemble_demand', top.get('predicted_demand', 0)):.0f} birim")
print()
print("  URUN ÖZELLİKLERİ:")
print("  " + "-" * 50)
for k, v in attrs.items():
    print(f"    {k:<20}: {v}")
print()
print("  SON GUN METRİKLERİ:")
print("  " + "-" * 50)
print(f"    Rank         : {int(son_gun['absolute_rank']):,}  (sayfa {int(son_gun['page_number'])})")
print(f"    Favori       : {int(son_gun['favorite_count']):,}")
print(f"    Sepet        : {int(son_gun['cart_count']):,}")
print(f"    Fiyat        : {son_gun['price']:.0f} TL", end="")
if son_gun['discount_rate'] > 0:
    print(f"  → indirimli: {son_gun['discounted_price']:.0f} TL (%{son_gun['discount_rate']:.0f})")
else:
    print()
print(f"    Rating       : {son_gun['rating']:.1f} / 5.0  ({int(son_gun['rating_count'])} yorum)")
print(f"    Stok         : {int(son_gun.get('total_stock', 0))} adet ({son_gun.get('stock_depth','?')})")
if son_gun.get("available_sizes"):
    try:
        stok = json.loads(son_gun["available_sizes"])
        stok_str = "  |  ".join([f"{b}: {q}" for b, q in stok.items()])
        print(f"    Bedenler     : {stok_str}")
    except:
        pass
print()
print("  NEDEN TREND?")
print("  " + "-" * 50)

# Buyuyen mi?
fav_ilk = metrics[metrics["product_id"] == pid].sort_values("date").iloc[0]["favorite_count"]
fav_son  = son_gun["favorite_count"]
fav_deg  = (fav_son - fav_ilk) / (fav_ilk + 1) * 100

rank_ilk = metrics[metrics["product_id"] == pid].sort_values("date").iloc[0]["absolute_rank"]
rank_son  = son_gun["absolute_rank"]

print(f"    Favori degisimi : {int(fav_ilk):,} -> {int(fav_son):,}  (%{fav_deg:+.0f})")
print(f"    Rank degisimi   : {int(rank_ilk):,} -> {int(rank_son):,}", end="")
if rank_son < rank_ilk:
    print(f"  (↑ {int(rank_ilk - rank_son)} basamak yukseldi)")
else:
    print()

print("=" * 70)
print()
# Top 5 listele
print("  TOP 5 ABİYE (TREND Skoru):")
print("  " + "-" * 60)
show_cols = ["product_id", "trend_label", "trend_score", "confidence"]
available_cols = [c for c in show_cols if c in preds.columns]
top5 = preds.sort_values("trend_score", ascending=False).head(5)[available_cols]
for i, row in top5.iterrows():
    p = products[products["product_id"] == int(row["product_id"])].iloc[0]
    attrs2 = p.get("attributes", {})
    if isinstance(attrs2, str):
        try: attrs2 = json.loads(attrs2)
        except: attrs2 = {}
    renk = attrs2.get("Renk", "?")
    mat  = attrs2.get("Materyal", "?")[:12]
    yaka = attrs2.get("Yaka", "?")
    etek = attrs2.get("Etek Boyu", "?")
    print(f"    {p['brand']:<22} | {renk:<8} {mat:<14} | {etek:<16} | {row['trend_label']:<12} | Skor: {row['trend_score']:.1f}")
print()
