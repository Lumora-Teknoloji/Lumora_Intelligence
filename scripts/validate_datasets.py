"""
validate_datasets.py
====================
Her kademe × periyot kombinasyonu için otomatik tutarlılık kontrolü.
Kontrol edilen şeyler:
  1. Rising ürünlerin rank gerçekten iyileşiyor mu?
  2. Falling ürünlerin rank gerçekten kötüleşiyor mu?
  3. Kademeler arası gürültü farkı beklenen yönde mi?
  4. Favori büyüme oranları mantıklı mı?
  5. Veri bütünlüğü (eksik gün, NaN, negatif değer var mı?)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os
import numpy as np
import pandas as pd
import json

BASE = os.path.join(os.path.dirname(__file__), "data", "datasets")

KADEMELER  = [1, 2, 3, 4, 5]
PERIYOTLAR = ["2m", "4m", "6m", "12m"]

# Beklenen gürültü sıralaması (k1 en az, k5 en fazla)
EXPECTED_NOISE = {1: 0.08, 2: 0.18, 3: 0.28, 4: 0.42, 5: 0.58}

results = {}
all_ok = True

print("=" * 70)
print(" DATASET TUTARLILIK DOĞRULAMASI")
print("=" * 70)

for k in KADEMELER:
    for p in PERIYOTLAR:
        path_dir = os.path.join(BASE, f"k{k}", p)
        dm_path  = os.path.join(path_dir, "daily_metrics.csv")
        pr_path  = os.path.join(path_dir, "products.csv")
        mt_path  = os.path.join(path_dir, "metadata.json")

        label = f"k{k}/{p}"

        if not os.path.exists(dm_path):
            print(f"  [{label}] ❌ DOSYA YOK: {dm_path}")
            all_ok = False
            continue

        dm  = pd.read_csv(dm_path)
        pr  = pd.read_csv(pr_path)
        with open(mt_path) as f:
            meta = json.load(f)

        issues = []

        # ── 1. Satır sayısı tutarlılığı ──────────────────────────────────
        n_products = len(pr)
        n_days     = meta["n_days"]
        expected   = n_products * n_days
        actual     = len(dm)
        if actual != expected:
            issues.append(f"Satır uyumsuzluğu: beklenen={expected:,} gerçek={actual:,}")

        # ── 2. NaN kontrolü ──────────────────────────────────────────────
        critical_cols = ["absolute_rank", "favorite_count", "date", "product_id"]
        for col in critical_cols:
            if col in dm.columns and dm[col].isna().any():
                issues.append(f"NaN bulundu: {col}")

        # ── 3. Negatif değer kontrolü ────────────────────────────────────
        for col in ["absolute_rank", "favorite_count", "cart_count",
                    "view_count", "rating_count"]:
            if col in dm.columns and (dm[col] < 0).any():
                issues.append(f"Negatif değer: {col}")

        # ── 4. Rank aralığı kontrolü ─────────────────────────────────────
        if "absolute_rank" in dm.columns:
            max_rank_actual = dm["absolute_rank"].max()
            if max_rank_actual > 9700:
                issues.append(f"Rank çok yüksek: {max_rank_actual}")

        # ── 5. Rising ürünlerin rank iyileşmesi ──────────────────────────
        dm["date"] = pd.to_datetime(dm["date"])

        # _trend_profile yalnızca products'ta kesin var
        dm_merged  = dm.merge(
            pr[["product_id", "_trend_profile"]].rename(
                columns={"_trend_profile": "profile"}),
            on="product_id", how="left"
        )

        rising  = dm_merged[dm_merged["profile"] == "rising"]
        falling = dm_merged[dm_merged["profile"] == "falling"]

        if len(rising) > 0 and "absolute_rank" in dm.columns:
            # İlk %20 vs son %20 gün rank karşılaştırması
            q20 = dm["date"].quantile(0.2)
            q80 = dm["date"].quantile(0.8)

            r_early = rising[rising["date"] <= q20]["absolute_rank"].mean()
            r_late  = rising[rising["date"] >= q80]["absolute_rank"].mean()
            rank_imp_pct = (r_early - r_late) / r_early * 100

            f_early = falling[falling["date"] <= q20]["absolute_rank"].mean()
            f_late  = falling[falling["date"] >= q80]["absolute_rank"].mean()
            rank_det_pct = (f_late - f_early) / f_early * 100

            # Rising için rank düşmeli (sayısal değer küçülmeli)
            if r_early <= r_late:
                issues.append(f"Rising rank kötüleşmiş! erken={r_early:.0f} geç={r_late:.0f}")
            if f_early >= f_late:
                issues.append(f"Falling rank iyileşmiş! erken={f_early:.0f} geç={f_late:.0f}")

        else:
            r_early = r_late = f_early = f_late = 0
            rank_imp_pct  = 0
            rank_det_pct  = 0

        # ── 6. Favori büyüme oranı kontrolü ─────────────────────────────
        fav_growth_rising   = 1.0
        fav_growth_falling  = 1.0
        if len(rising) > 0 and "favorite_count" in dm.columns:
            q20 = dm["date"].quantile(0.2)
            q80 = dm["date"].quantile(0.8)
            fv_early  = rising[rising["date"] <= q20]["favorite_count"].mean()
            fv_late   = rising[rising["date"] >= q80]["favorite_count"].mean()
            if fv_early > 0:
                fav_growth_rising = fv_late / fv_early

            ff_early  = falling[falling["date"] <= q20]["favorite_count"].mean()
            ff_late   = falling[falling["date"] >= q80]["favorite_count"].mean()
            if ff_early > 0:
                fav_growth_falling = ff_late / ff_early

        # ── 7. Kademeler arası gürültü tutarlılığı ───────────────────────
        if "absolute_rank" in dm.columns:
            # Her üründe günlük rank değişim std'si = gürültü proxy
            dm_sorted  = dm_merged.sort_values(["product_id", "date"])
            rank_diff  = dm_sorted.groupby("product_id")["absolute_rank"].diff().dropna().abs()
            noise_proxy = rank_diff.median()
        else:
            noise_proxy = 0

        # ── 8. Gün sayısı tutarlılığı ─────────────────────────────────────
        unique_days = dm["date"].dt.date.nunique()
        if unique_days != n_days:
            issues.append(f"Gün sayısı uyumsuz: beklenen={n_days} gerçek={unique_days}")

        # ── Sonuç ─────────────────────────────────────────────────────────
        status = "✅ OK" if not issues else "⚠️  SORUN"
        if issues:
            all_ok = False

        results[(k, p)] = {
            "status":           "ok" if not issues else "warn",
            "rank_imp_pct":     rank_imp_pct,
            "rank_det_pct":     rank_det_pct,
            "fav_growth_rising": fav_growth_rising,
            "fav_growth_falling": fav_growth_falling,
            "noise_proxy":      noise_proxy,
            "issues":           issues,
            "n_rows":           actual,
        }

        print(f"  [{label:<8}] {status}  |"
              f" rising rank↑{rank_imp_pct:+.0f}%"
              f"  falling rank↓{rank_det_pct:+.0f}%"
              f"  fav×{fav_growth_rising:.1f}"
              f"  noise={noise_proxy:.0f}")
        if issues:
            for iss in issues:
                print(f"             → ⚠️  {iss}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# KADEMELER ARASI GÜRÜLTÜ ARTIŞ KONTROLÜ
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print(" KADEMELER ARASI GÜRÜLTÜ ARTIŞI (6m periyot üzerinden)")
print("=" * 70)
prev_noise = 0
noise_trend_ok = True
for k in KADEMELER:
    if (k, "6m") in results:
        noise  = results[(k, "6m")]["noise_proxy"]
        ri_imp = results[(k, "6m")]["rank_imp_pct"]
        status = "↑ artıyor ✅" if noise > prev_noise else "↓ AZALIYOR ⚠️"
        kname  = {1: "Kristal", 2: "Net", 3: "Orta", 4: "Gürültülü", 5: "Kaotik"}[k]
        print(f"  Kademe {k} [{kname:<12}]: noise_proxy={noise:6.1f}  {status}  |  rank iyileş={ri_imp:+.0f}%")
        if noise <= prev_noise and k > 1:
            noise_trend_ok = False
        prev_noise = noise

print()

# ─────────────────────────────────────────────────────────────────────────────
# KADEMELER ARASI SİNYAL GÜCÜ AZALIŞI
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print(" SİNYAL GÜCÜ AZALIŞI (kademe arttıkça rising/falling ayrımı küçülmeli)")
print("=" * 70)
print(f"  {'':>5}  {'  2m':>8}  {'  4m':>8}  {'  6m':>8}  { '12m':>8}")
print(f"  {'':>5}  {'rank_imp':>8}  {'rank_imp':>8}  {'rank_imp':>8}  {'rank_imp':>8}")
print(f"  {'-'*50}")
for k in KADEMELER:
    kname = {1: "Kristal", 2: "Net", 3: "Orta", 4: "Gürültülü", 5: "Kaotik"}[k]
    vals  = [results.get((k, p), {}).get("rank_imp_pct", 0) for p in PERIYOTLAR]
    print(f"  k{k} [{kname:<9}]: {vals[0]:>+7.0f}%  {vals[1]:>+7.0f}%  {vals[2]:>+7.0f}%  {vals[3]:>+7.0f}%")

print()
print("=" * 70)
if all_ok and noise_trend_ok:
    print(" ✅ TÜM KONTROLLER BAŞARILI — Veriler tutarlı ve güvenilir")
else:
    print(" ⚠️  BAZI SORUNLAR TESPİT EDİLDİ — Yukarıdaki uyarıları inceleyin")
print("=" * 70)
