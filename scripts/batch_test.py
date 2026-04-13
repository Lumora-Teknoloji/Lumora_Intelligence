"""
batch_test.py
=============
20 benchmark dataset (k1-k5 × 2m/4m/6m/12m) üzerinde
algoritmanın precision/recall performansını ölçer.

Her dataset için:
  - TREND precision  (TREND dediği ürünlerin kaçı gerçekten rising?)
  - DUSEN precision  (DUSEN dediğinin kaçı gerçekten falling?)
  - Rising recall    (rising ürünlerin kaçını TREND yakaladı?)
  - Falling recall   (falling ürünlerin kaçını DUSEN yakaladı?)
  - Toplam süre

Sonunda 20 satırlık karşılaştırma tablosu yazdırır.
"""

import sys, io, os, time, warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from engine.predictor import PredictionEngine

DATASETS_DIR = ROOT / "data" / "datasets"
KADEMELER    = [1, 2, 3, 4, 5]
PERIYOTLAR   = ["2m", "4m", "6m", "12m"]
KADEME_NAMES = {1: "Kristal", 2: "Net", 3: "Orta", 4: "Gürültülü", 5: "Kaotik"}

# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER
# ─────────────────────────────────────────────────────────────────────────────
def adapt_df(dm: pd.DataFrame, pr: pd.DataFrame) -> pd.DataFrame:
    """Dataset'i engine formatına dönüştür."""
    df = dm.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Products'tan attributes çek
    if "attributes" in pr.columns:
        import json
        def extract_attr(row, key, default):
            try:
                return json.loads(row).get(key, default)
            except Exception:
                return default
        pr = pr.copy()
        pr["fabric"] = pr["attributes"].apply(lambda x: extract_attr(x, "Materyal", "Pamuk"))
        pr["color"]  = pr["attributes"].apply(lambda x: extract_attr(x, "Renk", "Siyah"))

    # Products merge
    merge_cols = ["product_id"]
    for c in ["fabric", "color", "brand"]:
        if c in pr.columns:
            merge_cols.append(c)
    if "_trend_profile" in pr.columns:
        merge_cols.append("_trend_profile")

    df = df.merge(pr[merge_cols], on="product_id", how="left")

    # Fallback defaults
    if "fabric" not in df.columns:
        df["fabric"] = "Pamuk"
    if "color" not in df.columns:
        df["color"] = "Siyah"
    if "brand" not in df.columns:
        df["brand"] = "Bilinmiyor"
    if "search_term" not in df.columns:
        df["search_term"] = df["category"]

    for col in ["discount_rate", "view_count", "rating_count", "engagement_score"]:
        if col not in df.columns:
            df[col] = 0
    df["engagement_score"] = pd.to_numeric(df["engagement_score"], errors="coerce").fillna(0)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# TEK TEST
# ─────────────────────────────────────────────────────────────────────────────
def run_single(kademe: int, period: str) -> dict:
    path = DATASETS_DIR / f"k{kademe}" / period
    dm_path = path / "daily_metrics.csv"
    pr_path = path / "products.csv"

    if not dm_path.exists():
        return {"error": "Dosya yok"}

    t0 = time.time()

    # Yükle
    dm = pd.read_csv(dm_path, low_memory=False)
    pr = pd.read_csv(pr_path, low_memory=False)

    # ── ÖRNEKLEME: tüm periyotlarda 1 ürün/profile/kategori ──────────────────
    # 120 kategori × 3 profil × 1 ürün = max 360 ürün
    # Bu algoritmanın O(ürün×gün) karmaşıklığını ~10x düşürür
    dm_t = dm  # daily_metrics zaten _trend_profile içeriyor
    pids = (
        dm_t.groupby(["category", "_trend_profile"])["product_id"]
        .apply(lambda x: x.drop_duplicates().sample(min(1, x.nunique()), random_state=42))
        .explode().values
    )
    dm = dm[dm["product_id"].isin(pids)]

    df = adapt_df(dm, pr)
    n_products = df["product_id"].nunique()
    n_rows     = len(df)

    # Engine
    engine = PredictionEngine(use_prophet=False, use_clip=False)
    engine.train(df, verbose=False)
    predictions = engine.predict()

    elapsed = time.time() - t0

    if predictions.empty:
        return {"error": "Tahmin boş"}

    # Ground truth
    true_profiles = (
        pr[["product_id", "_trend_profile"]]
        .drop_duplicates("product_id")
    )
    check = predictions.merge(true_profiles, on="product_id", how="left")

    # TREND precision
    trend_preds = check[check["trend_label"] == "TREND"]
    trend_prec  = 0.0
    if len(trend_preds) > 0:
        trend_prec = (trend_preds["_trend_profile"] == "rising").sum() / len(trend_preds) * 100

    # DUSEN precision
    dusen_preds = check[check["trend_label"].str.upper().isin(["DUSEN","DÜSEN"])]
    dusen_prec  = 0.0
    if len(dusen_preds) > 0:
        dusen_prec = (dusen_preds["_trend_profile"] == "falling").sum() / len(dusen_preds) * 100

    # POTANSIYEL → rising oranı
    pot_preds = check[check["trend_label"].str.upper().isin(["POTANSIYEL","POTANSİYEL"])]
    pot_rising = 0.0
    if len(pot_preds) > 0:
        pot_rising = (pot_preds["_trend_profile"] == "rising").sum() / len(pot_preds) * 100

    # Rising recall: rising olan ürünlerin kaçı TREND olarak etiketlendi?
    rising_total = (check["_trend_profile"] == "rising").sum()
    rising_caught = ((check["_trend_profile"] == "rising") &
                     (check["trend_label"] == "TREND")).sum()
    rising_recall = rising_caught / max(rising_total, 1) * 100

    # Falling recall
    falling_total = (check["_trend_profile"] == "falling").sum()
    falling_caught = ((check["_trend_profile"] == "falling") &
                      (check["trend_label"].str.upper().isin(["DUSEN","DÜSEN"]))).sum()
    falling_recall = falling_caught / max(falling_total, 1) * 100

    # Label dağılımı
    label_dist = check["trend_label"].value_counts().to_dict()

    return {
        "kademe":         kademe,
        "period":         period,
        "n_products":     n_products,
        "n_rows":         n_rows,
        "trend_prec":     trend_prec,
        "trend_count":    len(trend_preds),
        "dusen_prec":     dusen_prec,
        "dusen_count":    len(dusen_preds),
        "pot_rising":     pot_rising,
        "rising_recall":  rising_recall,
        "falling_recall": falling_recall,
        "elapsed":        elapsed,
        "label_dist":     label_dist,
        "error":          None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kademeler", nargs="+", type=int, default=[1,2,3,4,5])
    parser.add_argument("--periyotlar", nargs="+", default=["2m","4m","6m","12m"])
    args = parser.parse_args()
    kademeler = args.kademeler
    periyotlar = args.periyotlar

    print()
    print("╔" + "═" * 68 + "╗")
    print("║   LUMORA INTELLIGENCE — Batch Test                                ║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"  Kademeler : {kademeler}")
    print(f"  Periyotlar: {periyotlar}")
    print(f"  Dataset dizini: {DATASETS_DIR}")
    print()

    results = []
    total  = len(kademeler) * len(periyotlar)
    idx    = 0

    for k in kademeler:
        for p in periyotlar:
            idx += 1
            kname = KADEME_NAMES[k]
            print(f"  [{idx:2d}/{total}] k{k} [{kname:<9}] {p:>4}  ... ", end="", flush=True)
            try:
                r = run_single(k, p)
                if r.get("error"):
                    print(f"HATA: {r['error']}")
                else:
                    print(f"✅  TREND prec={r['trend_prec']:5.1f}%  "
                          f"DUSEN prec={r['dusen_prec']:5.1f}%  "
                          f"rising recall={r['rising_recall']:5.1f}%  "
                          f"({r['elapsed']:.0f}s)")
                results.append(r)
                # Anında CSV'ye yaz (test yarıda kesilirse kaybolmasın)
                if not r.get("error"):
                    out_dir = ROOT / "results"
                    out_dir.mkdir(exist_ok=True)
                    row_df = pd.DataFrame([{
                        "kademe": r["kademe"],
                        "kademe_name": KADEME_NAMES[r["kademe"]],
                        "period": r["period"],
                        "n_products": r["n_products"],
                        "trend_precision": round(r["trend_prec"], 1),
                        "trend_count": r["trend_count"],
                        "dusen_precision": round(r["dusen_prec"], 1),
                        "rising_recall": round(r["rising_recall"], 1),
                        "falling_recall": round(r["falling_recall"], 1),
                        "potansiyel_rising_pct": round(r["pot_rising"], 1),
                        "elapsed_sec": round(r["elapsed"], 1),
                    }])
                    out_path = out_dir / "batch_test_results.csv"
                    row_df.to_csv(out_path, mode="a",
                                  header=not out_path.exists(),
                                  index=False, encoding="utf-8-sig")
            except Exception as e:
                print(f"EXCEPTION: {e}")
                results.append({"kademe": k, "period": p, "error": str(e)})

    # ── ÖZET TABLO ─────────────────────────────────────────────────────────────
    print()
    print("═" * 100)
    print("  SONUÇ TABLOSU")
    print("═" * 100)
    header = (f"  {'Kademe':<10} {'Periyot':>6}  "
              f"{'TREND Prec':>10}  {'TREND#':>6}  "
              f"{'DUSEN Prec':>10}  {'DUSEN#':>6}  "
              f"{'POT→Rise':>8}  "
              f"{'Rise Recall':>11}  {'Fall Recall':>11}  "
              f"{'Süre':>5}")
    print(header)
    print("  " + "─" * 97)

    # Kademeler arası ortalama
    from collections import defaultdict
    by_kademe    = defaultdict(list)
    by_period    = defaultdict(list)

    for r in results:
        if r.get("error"):
            print(f"  k{r['kademe']}/{r['period']:>4}  HATA: {r['error']}")
            continue
        k = r["kademe"]
        p = r["period"]
        kname = KADEME_NAMES[k]

        row = (f"  k{k} [{kname:<8}]  {p:>4}   "
               f"{r['trend_prec']:>9.1f}%  {r['trend_count']:>6}  "
               f"{r['dusen_prec']:>9.1f}%  {r['dusen_count']:>6}  "
               f"{r['pot_rising']:>7.1f}%  "
               f"{r['rising_recall']:>10.1f}%  {r['falling_recall']:>10.1f}%  "
               f"{r['elapsed']:>4.0f}s")
        print(row)

        by_kademe[k].append(r)
        by_period[p].append(r)

        if p == periyotlar[-1] and k != kademeler[-1]:
            print("  " + "─" * 97)

    # Kademe ortalamaları
    print()
    print("═" * 100)
    print("  KADEME ORTALAMALARI (tüm periyotlar)")
    print("═" * 100)
    print(f"  {'Kademe':<15}  {'TREND Prec':>10}  {'DUSEN Prec':>10}  {'Rise Recall':>11}  {'Fall Recall':>11}")
    print("  " + "─" * 60)
    for k in KADEMELER:
        rs = [r for r in by_kademe[k] if not r.get("error")]
        if not rs:
            continue
        avg_tp = np.mean([r["trend_prec"] for r in rs])
        avg_dp = np.mean([r["dusen_prec"] for r in rs])
        avg_rr = np.mean([r["rising_recall"] for r in rs])
        avg_fr = np.mean([r["falling_recall"] for r in rs])
        kname  = KADEME_NAMES[k]
        print(f"  k{k} [{kname:<9}]  {avg_tp:>9.1f}%  {avg_dp:>9.1f}%  {avg_rr:>10.1f}%  {avg_fr:>10.1f}%")

    # Periyot ortalamaları
    print()
    print("═" * 100)
    print("  PERİYOT ORTALAMALARI (tüm kademeler)")
    print("═" * 100)
    print(f"  {'Periyot':<10}  {'TREND Prec':>10}  {'DUSEN Prec':>10}  {'Rise Recall':>11}  {'Fall Recall':>11}")
    print("  " + "─" * 55)
    for p in PERIYOTLAR:
        rs = [r for r in by_period[p] if not r.get("error")]
        if not rs:
            continue
        avg_tp = np.mean([r["trend_prec"] for r in rs])
        avg_dp = np.mean([r["dusen_prec"] for r in rs])
        avg_rr = np.mean([r["rising_recall"] for r in rs])
        avg_fr = np.mean([r["falling_recall"] for r in rs])
        print(f"  {p:<10}   {avg_tp:>9.1f}%  {avg_dp:>9.1f}%  {avg_rr:>10.1f}%  {avg_fr:>10.1f}%")

    # Sonuçları kaydet
    rows_out = []
    for r in results:
        if not r.get("error"):
            rows_out.append({
                "kademe": r["kademe"],
                "kademe_name": KADEME_NAMES[r["kademe"]],
                "period": r["period"],
                "n_products": r["n_products"],
                "trend_precision": round(r["trend_prec"], 1),
                "trend_count": r["trend_count"],
                "dusen_precision": round(r["dusen_prec"], 1),
                "rising_recall": round(r["rising_recall"], 1),
                "falling_recall": round(r["falling_recall"], 1),
                "potansiyel_rising_pct": round(r["pot_rising"], 1),
                "elapsed_sec": round(r["elapsed"], 1),
            })

    out = pd.DataFrame(rows_out)
    out_path = ROOT / "results" / "batch_test_results.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print()
    print(f"  💾 Tüm sonuçlar kaydedildi: {out_path}")
    print()


if __name__ == "__main__":
    main()
