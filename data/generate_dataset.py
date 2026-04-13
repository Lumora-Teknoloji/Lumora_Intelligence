"""
generate_dataset.py
===================
Benchmark Dataset Generator — 5 Kademe × 4 Zaman Periyodu = 20 Dataset

Kademe 1 → Kristal berrak sinyal (kolay algılanır)
Kademe 5 → Kaotik gürültü (neredeyse algılanamaz)

Her kademe × zaman periyodu kombinasyonu farklı seed ile tamamen bağımsız.

Çıktı: data/datasets/k{1-5}/{2m|4m|6m|12m}/
         ├── products.csv
         └── daily_metrics.csv
"""

import os, sys
import numpy as np
import pandas as pd
from datetime import date, timedelta
import math, json, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KADEME KONFİGÜRASYONLARI
# ─────────────────────────────────────────────────────────────────────────────
KADEME_CONFIGS = {
    1: {
        "name": "Kristal Berraklık",
        "desc": "Sinyal çok net, gürültü minimal. Basit kural tabanlı sistem bile %95+ doğrulukla tahmin eder.",
        "noise_sigma":        0.08,   # günlük ±8% gürültü
        "momentum_alpha":     0.80,   # %80 yeni sinyal ağırlığı
        "rising_start_pct":   0.96,   # max_rank'ın %96'sında başla (kötü sıra)
        "rising_end_pct":     0.02,   # max_rank'ın %2'sinde bitir  (çok iyi)
        "falling_start_pct":  0.02,
        "falling_end_pct":    0.96,
        "stable_center_pct":  0.45,   # ortaya konumlan
        "stable_volatility":  0.03,   # ±3% rank dalgalanması
        "fav_growth_rising":  40.0,   # rising: baştan 40x büyüme
        "fav_growth_falling": 0.025,  # falling: başın %2.5'ine düş
        "fav_growth_stable":  1.0,    # stable: sabit
        "irregular_prob":     0.01,   # %1 rastlantısal spike
        "irregular_magnitude": 1.5,
        "weekend_boost":      1.12,
        "discount_effect":    1.8,
        "season_amplitude":   0.55,
    },
    2: {
        "name": "Net",
        "desc": "Net sinyal, az gürültü. İyi bir ML modeli kolayca öğrenir.",
        "noise_sigma":        0.18,
        "momentum_alpha":     0.65,
        "rising_start_pct":   0.92,
        "rising_end_pct":     0.05,
        "falling_start_pct":  0.05,
        "falling_end_pct":    0.92,
        "stable_center_pct":  0.45,
        "stable_volatility":  0.06,
        "fav_growth_rising":  15.0,
        "fav_growth_falling": 0.07,
        "fav_growth_stable":  1.0,
        "irregular_prob":     0.04,
        "irregular_magnitude": 2.0,
        "weekend_boost":      1.10,
        "discount_effect":    1.8,
        "season_amplitude":   0.50,
    },
    3: {
        "name": "Orta",
        "desc": "Gerçekçi gürültü seviyesi. Mevcut generate_csv_v2.py'ye yakın.",
        "noise_sigma":        0.28,
        "momentum_alpha":     0.50,
        "rising_start_pct":   0.88,
        "rising_end_pct":     0.09,
        "falling_start_pct":  0.09,
        "falling_end_pct":    0.88,
        "stable_center_pct":  0.45,
        "stable_volatility":  0.10,
        "fav_growth_rising":  7.0,
        "fav_growth_falling": 0.15,
        "fav_growth_stable":  1.0,
        "irregular_prob":     0.06,
        "irregular_magnitude": 2.5,
        "weekend_boost":      1.08,
        "discount_effect":    1.8,
        "season_amplitude":   0.45,
    },
    4: {
        "name": "Gürültülü",
        "desc": "Yüksek gürültü, zayıf sinyal. CatBoost'un feature importance testi için ideal.",
        "noise_sigma":        0.42,
        "momentum_alpha":     0.35,
        "rising_start_pct":   0.82,
        "rising_end_pct":     0.19,
        "falling_start_pct":  0.19,
        "falling_end_pct":    0.82,
        "stable_center_pct":  0.45,
        "stable_volatility":  0.16,
        "fav_growth_rising":  3.0,
        "fav_growth_falling": 0.35,
        "fav_growth_stable":  1.0,
        "irregular_prob":     0.10,
        "irregular_magnitude": 3.0,
        "weekend_boost":      1.06,
        "discount_effect":    1.8,
        "season_amplitude":   0.35,
    },
    5: {
        "name": "Kaotik",
        "desc": "Maksimum gürültü. Rising/falling/stable neredeyse aynı görünür. Sinyal gürültü içinde kaybolur.",
        "noise_sigma":        0.58,
        "momentum_alpha":     0.22,
        "rising_start_pct":   0.76,
        "rising_end_pct":     0.32,  # rising bile kötü sırada bitiyor
        "falling_start_pct":  0.32,
        "falling_end_pct":    0.76,
        "stable_center_pct":  0.50,
        "stable_volatility":  0.22,
        "fav_growth_rising":  1.8,   # stable ile neredeyse aynı
        "fav_growth_falling": 0.60,
        "fav_growth_stable":  1.0,
        "irregular_prob":     0.15,
        "irregular_magnitude": 4.0,
        "weekend_boost":      1.04,
        "discount_effect":    1.8,
        "season_amplitude":   0.25,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# ZAMAN PERİYOTLARI
# ─────────────────────────────────────────────────────────────────────────────
TIME_PERIODS = {
    "2m":  {"days": 60,  "start": date(2025, 10,  1)},
    "4m":  {"days": 120, "start": date(2025,  9,  1)},
    "6m":  {"days": 180, "start": date(2025,  7,  1)},
    "12m": {"days": 365, "start": date(2025,  1,  1)},
}

# ─────────────────────────────────────────────────────────────────────────────
# 120 KATEGORİ
# ─────────────────────────────────────────────────────────────────────────────
CATEGORIES = [
    # Üst giyim
    "crop top", "crop top baskılı", "crop top dantelli", "crop top fitilli",
    "kazak crop", "kazak oversize", "hırka oversize", "hırka kısa",
    "bluz kolsuz", "bluz uzun kol", "gömlek oversize", "gömlek crop",
    "sweatshirt oversize", "sweatshirt crop", "tişört oversize", "tişört ribana",
    "yelek örme", "yelek şişme", "ceket kadın", "blazer kadın",
    # Alt giyim
    "tayt", "tayt yüksek bel", "tayt kapri", "tayt kısa",
    "pantolon kadın", "pantolon yüksek bel", "pantolon deri görünümlü", "pantolon palazzo",
    "şort kadın", "şort denim", "jean kadın", "jean yüksek bel",
    "etek midi", "etek mini", "etek maxi", "etek volanlı",
    "eşofman altı", "eşofman takım", "pijama takım", "pijama kadın",
    # Elbise
    "elbise günlük", "elbise oversize", "elbise midi", "elbise mini",
    "elbise maxi", "elbise dantelli", "elbise ribana", "elbise keten",
    # Abiye & Özel
    "abiye mezuniyet", "abiye nikah", "abiye kısa", "abiye uzun",
    "kadın abiye", "gece elbisesi", "mezuniyet elbisesi", "nişan elbisesi",
    # Spor
    "spor tayt", "spor sutyeni", "spor şort", "spor takım",
    "yoga tayt", "yoga üst", "koşu tayt", "fitness üst",
    # Dış giyim
    "mont kadın", "kaban kadın", "trençkot", "yağmurluk",
    "deri ceket", "bomber ceket", "rüzgarlık", "polar ceket",
    # Aksesuar & Tamamlayıcı
    "çanta kadın", "çanta tote", "çanta omuz", "çanta sırt",
    "şapka bere", "şapka bucket", "kemer kadın", "atkı şal",
    "çorap kadın", "çorap dizüstü", "iç çamaşırı seti", "sütyen",
    # İlkbahar-Yaz
    "ilkbahar elbise", "yaz elbise", "bikini üst", "bikini alt",
    "mayo kadın", "plaj elbisesi", "keten şort", "keten gömlek",
    # Sonbahar-Kış
    "sonbahar kazak kombini", "kış kombini", "bordo kazak", "mürdüm kazak",
    "termal set", "kalın tayt", "yünlü şapka",
    # Trend karışık
    "katmanlı giyim set",
    "fransız tarzı elbise", "minimal elbise", "vintage jean kadın", "business casual",
    "bohem elbise", "romantik elbise", "sportif set",
    "seksi elbise", "zarif takım", "ev kıyafeti seti", "gecelik",
    # Genişletilmiş
    "crop top bağcıklı", "push up tayt", "yüzme takımı", "tunik",
    "polar şapka", "kulaklık bandı", "terlik kadın", "sandalet",
    "sneaker kadın", "topuklu ayakkabı",
][:120]  # tam 120 kategori

# ─────────────────────────────────────────────────────────────────────────────
# KATEGORİ PARAMETRELERİ
# Nedensel zincir için base_views ve dönüşüm oranları da burada tanımlanır
# ─────────────────────────────────────────────────────────────────────────────
def get_cat_params(cat, rng):
    # ── Fiyat & Mevsim segmenti ──────────────────────────────────────────
    if any(k in cat for k in ["abiye", "gece", "mezuniyet", "nişan"]):
        price              = float(rng.uniform(450, 1800))
        base_views         = float(rng.uniform(300, 800))    # günlük temel görüntülenme
        view_to_fav_rate   = 0.04   # daha seçici alışveriş → az fav
        fav_to_cart_rate   = 0.18   # favori → sepet daha yüksek (niyet yüksek)
        cart_to_buy_rate   = 0.20   # pahalı ürün, karar zor
        season_peak_month  = 5
    elif any(k in cat for k in ["mont", "kaban", "trençkot", "kış"]):
        price              = float(rng.uniform(200, 900))
        base_views         = float(rng.uniform(500, 1500))
        view_to_fav_rate   = 0.07
        fav_to_cart_rate   = 0.14
        cart_to_buy_rate   = 0.25
        season_peak_month  = 11
    elif any(k in cat for k in ["mayo", "bikini", "plaj", "yaz", "ilkbahar"]):
        price              = float(rng.uniform(80, 350))
        base_views         = float(rng.uniform(600, 2000))
        view_to_fav_rate   = 0.10   # mevsimsel aciliyet → hızlı fav
        fav_to_cart_rate   = 0.12
        cart_to_buy_rate   = 0.30
        season_peak_month  = 6
    elif any(k in cat for k in ["spor", "yoga", "koşu", "fitness", "tayt"]):
        price              = float(rng.uniform(80, 350))
        base_views         = float(rng.uniform(800, 2500))
        view_to_fav_rate   = 0.09
        fav_to_cart_rate   = 0.16   # sporseverler kararlı alıcılar
        cart_to_buy_rate   = 0.35
        season_peak_month  = 1
    elif any(k in cat for k in ["çanta", "kemer", "şapka", "atkı", "çorap"]):
        price              = float(rng.uniform(60, 400))
        base_views         = float(rng.uniform(400, 1200))
        view_to_fav_rate   = 0.08
        fav_to_cart_rate   = 0.13
        cart_to_buy_rate   = 0.28
        season_peak_month  = 3
    else:  # standart üst/alt giyim
        price              = float(rng.uniform(80, 450))
        base_views         = float(rng.uniform(600, 1800))
        view_to_fav_rate   = 0.08
        fav_to_cart_rate   = 0.12
        cart_to_buy_rate   = 0.25
        season_peak_month  = 9

    discount_rate = float(rng.choice([0.0, 0.0, 0.10, 0.15, 0.20, 0.30, 0.40],
                                      p=[0.30, 0.20, 0.15, 0.15, 0.10, 0.06, 0.04]))
    base_fav  = int(rng.integers(20, 800))
    max_sizes = int(rng.choice([4, 5, 6, 7, 8]))

    return {
        "price":             price,
        "discount_rate":     discount_rate,
        "base_views":        base_views,
        "view_to_fav_rate":  view_to_fav_rate,
        "fav_to_cart_rate":  fav_to_cart_rate,
        "cart_to_buy_rate":  cart_to_buy_rate,
        "base_fav":          base_fav,
        "max_sizes":         max_sizes,
        "season_peak_month": season_peak_month,
    }

# ─────────────────────────────────────────────────────────────────────────────
# TRAJECTORY: Rank & Favourite eğrileri
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid_curve(n, steepness=6.0):
    """0→1 sigmoid eğrisi (normalize). Yumuşak geçiş."""
    x = np.linspace(-steepness/2, steepness/2, n)
    return 1.0 / (1.0 + np.exp(-x))

def make_rank_trajectory(profile, n_days, cfg, rng, max_rank):
    """
    Ürün için n_days uzunluğunda rank serisi üret.
    Lower rank = better (rank 1 = en iyi)
    """
    noise = cfg["noise_sigma"]
    alpha = cfg["momentum_alpha"]

    if profile == "rising":
        start = cfg["rising_start_pct"] * max_rank
        end   = cfg["rising_end_pct"]   * max_rank
        # Sigmoid iniş: başta yavaş, ortada hızlı, sonda yavaş
        base_curve = start + (end - start) * sigmoid_curve(n_days)

    elif profile == "falling":
        start = cfg["falling_start_pct"] * max_rank
        end   = cfg["falling_end_pct"]   * max_rank
        base_curve = start + (end - start) * sigmoid_curve(n_days)

    else:  # stable
        center = cfg["stable_center_pct"] * max_rank
        vol    = cfg["stable_volatility"]
        # Yavaş sinüs dalgalanması
        phase  = float(rng.uniform(0, 2 * math.pi))
        base_curve = center * (1 + vol * 0.5 * np.sin(
            np.linspace(0, 2 * math.pi, n_days) + phase
        ))

    # Günlük gürültü ekle (rank mutlak değer → her zaman pozitif)
    daily_noise = rng.normal(0, noise, n_days)

    # Irregüler spike / crash (kademe arttıkça daha sık)
    for d in range(n_days):
        if rng.random() < cfg["irregular_prob"]:
            spike_dir = rng.choice([-1, 1])
            spike_mag = float(rng.uniform(1.0, cfg["irregular_magnitude"]))
            daily_noise[d] += spike_dir * spike_mag

    # Momentum smoothing
    noisy = np.zeros(n_days)
    prev  = base_curve[0]
    for d in range(n_days):
        new_signal = base_curve[d] * (1 + daily_noise[d])
        prev = alpha * new_signal + (1 - alpha) * prev
        noisy[d] = prev

    # Sınır: 1 ile max_rank arasında kal
    noisy = np.clip(noisy, 1, max_rank).astype(int)
    return noisy

def make_fav_trajectory(profile, n_days, cfg, rng, base_fav, alpha):
    """Favori sayısı eğrisi."""
    noise = cfg["noise_sigma"]

    if profile == "rising":
        growth = cfg["fav_growth_rising"]
        final  = base_fav * growth
    elif profile == "falling":
        growth = cfg["fav_growth_falling"]
        final  = base_fav * growth
    else:
        final  = base_fav * cfg["fav_growth_stable"]

    # Log-linear büyüme
    if profile == "rising":
        base_curve = np.exp(np.linspace(math.log(max(base_fav, 1)),
                                         math.log(max(final, base_fav + 1)),
                                         n_days))
    elif profile == "falling":
        base_curve = np.exp(np.linspace(math.log(max(base_fav, 1)),
                                         math.log(max(final, 1)),
                                         n_days))
    else:
        phase = float(rng.uniform(0, 2 * math.pi))
        base_curve = base_fav * (1 + 0.05 * np.sin(
            np.linspace(0, 4 * math.pi, n_days) + phase))

    # Momentum smoothing + gürültü
    noisy = np.zeros(n_days)
    prev  = float(base_curve[0])
    for d in range(n_days):
        new_signal = base_curve[d] * max(0.1, 1 + rng.normal(0, noise * 0.6))
        prev = alpha * new_signal + (1 - alpha) * prev
        noisy[d] = max(0.0, prev)

    return noisy

# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTS CSV
# ─────────────────────────────────────────────────────────────────────────────

def generate_products(n_rising, n_falling, n_stable, seed, kademe):
    rng = np.random.default_rng(seed)
    rows = []
    pid  = 1

    for cat in CATEGORIES:
        for profile in (
            ["rising"]  * n_rising  +
            ["falling"] * n_falling +
            ["stable"]  * n_stable
        ):
            params = get_cat_params(cat, rng)
            dp     = params["discount_rate"]
            disc_p = params["price"] * (1 - dp)

            fab_choices = ["Pamuk", "Polyester", "Viskon", "Keten",
                           "Modal", "Ribana", "Saten", "Kadife"]
            col_choices = ["Siyah", "Beyaz", "Gri", "Bej", "Kırmızı",
                           "Mavi", "Yeşil", "Pembe", "Sarı", "Mor", "Turuncu", "Bordo"]

            attrs = {
                "Renk":    rng.choice(col_choices),
                "Materyal": rng.choice(fab_choices),
                "Kalıp":  rng.choice(["Slim Fit", "Regular", "Oversize", "Crop"]),
                "Sezon":  rng.choice(["İlkbahar/Yaz", "Sonbahar/Kış", "Tüm Sezonlar"]),
            }

            rows.append({
                "product_id":       pid,
                "category":         cat,
                "name":             f"{attrs['Renk']} {cat.title()} #{pid}",
                "brand":            rng.choice(["Marka A","Marka B","Marka C","Marka D","Marka E"]),
                "price":            round(params["price"], 2),
                "discounted_price": round(disc_p, 2),
                "discount_rate":    round(dp, 2),
                "attributes":       json.dumps(attrs, ensure_ascii=False),
                "_trend_profile":   profile,
                "_kademe":          kademe,
            })
            pid += 1

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# DAILY METRICS CSV
# ─────────────────────────────────────────────────────────────────────────────

def generate_daily_metrics(products_df, n_days, start_date, cfg, seed):
    rng      = np.random.default_rng(seed + 9999)
    alpha    = cfg["momentum_alpha"]
    max_rank = 200 * 48  # same as v2: 9600

    # Mevsim faktörü (tüm günler için bir kez hesapla)
    dates = [start_date + timedelta(days=d) for d in range(n_days)]
    months = np.array([d.month for d in dates])
    weekday = np.array([d.weekday() for d in dates])
    is_weekend = (weekday >= 4).astype(float)  # Cuma-Pazar

    records = []

    for _, prod in products_df.iterrows():
        pid     = int(prod["product_id"])
        cat     = prod["category"]
        profile = prod["_trend_profile"]
        params  = get_cat_params(cat, rng)

        base_price       = params["price"]
        disc_rate        = params["discount_rate"]
        base_fav         = params["base_fav"]
        base_views       = params["base_views"]
        view_to_fav_rate = params["view_to_fav_rate"]
        fav_to_cart_rate = params["fav_to_cart_rate"]
        cart_to_buy_rate = params["cart_to_buy_rate"]

        # ── Mevsim faktörü ──────────────────────────────────────────────────
        peak_m = params["season_peak_month"]
        season = 1 + cfg["season_amplitude"] * np.sin(
            2 * math.pi * (months - peak_m) / 12
        )

        # ── 1. RANK TRAJEKTORİSİ (birincil) ────────────────────────────────
        ranks     = make_rank_trajectory(profile, n_days, cfg, rng, max_rank)
        rank_pct  = ranks / max_rank
        rank_reach = np.exp(-3.5 * rank_pct)   # 0→1, iyi rank=yüksek erişim

        # ── 2. FAV TRAJEKTORİSİ (birincil eğri) ────────────────────────────
        fav_base_curve = make_fav_trajectory(profile, n_days, cfg, rng, base_fav, alpha)

        # ── 3. VIEW_COUNT ───────────────────────────────────────────────────
        # Zincir: rank_reach → sayfada görünürlük → görüntülenme
        # İndirim, hafta sonu ve mevsim görüntülenmeyi artırır
        discount_view_boost = 1 + disc_rate * 0.5   # indirim listing'i öne çıkarır
        weekend_view_boost  = 1 + is_weekend * 0.30

        views_raw = (rank_reach * base_views * season
                     * discount_view_boost * weekend_view_boost
                     * (1 + rng.normal(0, cfg["noise_sigma"] * 0.5, n_days)))
        views_raw = views_raw.clip(0)
        # Bazı günler görüntülenme 0 (ürün listeden düştü, sistem hatası vb.)
        views_zero_mask = rng.random(n_days) < (rank_pct * 0.20)  # kötü rankta daha sık
        views = np.where(views_zero_mask, 0, views_raw).astype(int)

        # ── 4. FAVORITE_COUNT ───────────────────────────────────────────────
        # Zincir: view_count → yeni günlük fav (view_to_fav_rate kadar)
        # + fav eğrisi (genel popülarite trendi)
        # + sosyal kanıt (ileride rating birikince geri besleme)
        daily_new_favs_from_views = views * view_to_fav_rate

        # Fav eğrisi + görüntülenmeden gelen anlık sinyal blend'i
        # view sinyali eğriye en fazla ±20% katkı yapabilir
        view_signal = np.zeros(n_days)
        views_mean = views.mean() + 1
        for d in range(n_days):
            view_signal[d] = alpha * (views[d] / views_mean) + (
                (1 - alpha) * (view_signal[d-1] if d > 0 else 1.0))
        fav_view_mult = np.clip(view_signal, 0.5, 2.0)  # sınırlı etki

        favs = fav_base_curve * fav_view_mult
        favs = favs.clip(0)

        # ── 5. CART_COUNT ───────────────────────────────────────────────────
        # Zincir: fav_count × fav_to_cart_rate
        # + indirim etkisi (anlık sepet kararını hızlandırır)
        # + hafta sonu etkisi (alışveriş günleri)
        # + haftalık ortalama fav (kararlılık sinyali)
        fav_7d_avg = pd.Series(favs).rolling(7, min_periods=1).mean().values

        discount_cart_boost = 1 + disc_rate * 2.5   # indirim sepeti doğrudan etkiler
        weekend_cart_boost  = 1 + is_weekend * (cfg["weekend_boost"] - 1)
        price_elast         = disc_rate * 1.8

        carts_raw = (fav_7d_avg * fav_to_cart_rate
                     * discount_cart_boost
                     * weekend_cart_boost
                     * season
                     * (1 + rng.normal(0, cfg["noise_sigma"] * 0.6, n_days)))
        carts_raw = carts_raw.clip(0)
        # Çoğu gün sepet 0 (Trendyol'da %60-80 sıfır)
        cart_zero_prob = max(0.30, 1.0 - fav_to_cart_rate * 3)
        cart_zero_mask = rng.random(n_days) < cart_zero_prob
        carts = np.where(cart_zero_mask, 0, carts_raw).astype(int)

        # ── 6. PURCHASES (gizli değişken) ──────────────────────────────────
        # Zincir: cart → satın alma (daha düşük dönüşüm)
        purchases = (carts * cart_to_buy_rate
                     * (1 + rng.normal(0, 0.10, n_days))).clip(0)

        # ── 7. RATING_COUNT ─────────────────────────────────────────────────
        # Zincir: purchases[d-7] → yorumlar (7 gün gecikme)
        # Satın alan kişilerin ~%35-45'i yorum bırakıyor
        REVIEW_RATE = 0.40
        DELAY_DAYS  = 7
        delayed_purchases = np.zeros(n_days)
        delayed_purchases[DELAY_DAYS:] = purchases[:-DELAY_DAYS]
        daily_reviews = (delayed_purchases * REVIEW_RATE
                         * (1 + rng.normal(0, 0.12, n_days))).clip(0)
        rating_count = np.cumsum(daily_reviews).astype(int)

        # ── 8. AVG_RATING ───────────────────────────────────────────────────
        # Zincir: ilk yorumlar daha aşırı (1 ya da 5 yıldız)
        # Yorum sayısı arttıkça ortalama 4.2 civarında stabilleşir
        initial_rating = float(rng.uniform(3.8, 5.0))
        review_maturity = np.log1p(rating_count) / (np.log1p(rating_count.max()) + 1)
        target_rating   = 4.2  # uzun vadeli denge noktası
        rating = initial_rating + (target_rating - initial_rating) * review_maturity
        rating = np.clip(rating + rng.normal(0, 0.05, n_days), 1.0, 5.0)

        # ── 9. SOSYAL KANIT GERI BESLEMESİ ────────────────────────────────
        # Zincir: rating_count büyüdükçe → fav oranı artar ("9.432 yorum" güven verir)
        # Küçük etki: max %15 fav artışı
        review_growth  = np.diff(rating_count.astype(float), prepend=0).clip(0)
        review_ma7     = pd.Series(review_growth).rolling(7, min_periods=1).mean().values
        review_ma_norm = (review_ma7 / (review_ma7.max() + 1e-6)).clip(0, 1)
        social_proof   = review_ma_norm * 0.15  # max %15 destek

        # Favori sayısını geriye dönük sosyal kanıtla güncelle
        favs = (favs * (1 + social_proof)).clip(0)

        # ── 10. ENGAGEMENT SCORE (kompozit) ────────────────────────────────
        # Zincir: rank_reach + fav_norm + cart_norm
        fav_norm  = favs  / (favs.max()  + 1)
        cart_norm = carts / (carts.max() + 1)
        engagement = (
            0.45 * rank_reach +
            0.35 * fav_norm +
            0.20 * cart_norm
        )

        # ── SONUÇ ──────────────────────────────────────────────────────────
        for d, dt in enumerate(dates):
            records.append({
                "date":              dt.isoformat(),
                "product_id":        pid,
                "category":          cat,
                "absolute_rank":     int(ranks[d]),
                "view_count":        int(views[d]),
                "favorite_count":    int(favs[d]),
                "cart_count":        int(carts[d]),
                "price":             round(base_price, 2),
                "discounted_price":  round(base_price * (1 - disc_rate), 2),
                "discount_rate":     round(disc_rate, 2),
                "rating":            round(float(rating[d]), 3),
                "rating_count":      int(rating_count[d]),
                "engagement_score":  round(float(engagement[d]), 4),
                "rank_reach_mult":   round(float(rank_reach[d]), 4),
                "season_factor":     round(float(season[d]), 4),
                "price_elast_boost": round(price_elast, 4),
                "_trend_profile":    profile,
            })

    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────────────────────
# TEK DATASET OLUŞTUR
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(kademe: int, period_name: str, n_days: int,
                     start_date: date, output_dir: str):
    cfg  = KADEME_CONFIGS[kademe]
    seed = kademe * 1000 + {"2m": 1, "4m": 2, "6m": 3, "12m": 4}[period_name]

    n_per_profile = 10   # 10 rising + 10 falling + 10 stable = 30 ürün/kategori
                         # 120 kategori × 30 = 3600 ürün

    print(f"  Ürünler üretiliyor... ", end="", flush=True)
    products_df = generate_products(
        n_rising=n_per_profile, n_falling=n_per_profile, n_stable=n_per_profile,
        seed=seed, kademe=kademe
    )
    print(f"{len(products_df):,} ürün")

    print(f"  Günlük metrikler üretiliyor ({n_days} gün × {len(products_df):,} ürün)... ",
          end="", flush=True)
    metrics_df = generate_daily_metrics(
        products_df=products_df,
        n_days=n_days,
        start_date=start_date,
        cfg=cfg,
        seed=seed,
    )
    expected   = len(products_df) * n_days
    print(f"{len(metrics_df):,} satır (beklenen: {expected:,})")

    os.makedirs(output_dir, exist_ok=True)
    prod_path  = os.path.join(output_dir, "products.csv")
    metric_path= os.path.join(output_dir, "daily_metrics.csv")
    products_df.to_csv(prod_path,   index=False)
    metrics_df.to_csv(metric_path,  index=False)

    # Metadata
    meta = {
        "kademe": kademe,
        "kademe_name": cfg["name"],
        "kademe_desc": cfg["desc"],
        "period": period_name,
        "n_days": n_days,
        "start_date": start_date.isoformat(),
        "n_products": len(products_df),
        "n_rows": len(metrics_df),
        "noise_sigma": cfg["noise_sigma"],
        "momentum_alpha": cfg["momentum_alpha"],
        "fav_growth_rising": cfg["fav_growth_rising"],
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return len(metrics_df)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    BASE_DIR   = os.path.join(os.path.dirname(__file__), "datasets")
    total_rows = 0

    print("=" * 65)
    print(" LUMORA INTELLIGENCE — Benchmark Dataset Generator")
    print(" 5 Kademe × 4 Zaman Periyodu = 20 Dataset")
    print("=" * 65)
    print()
    print(" Kademe 1: Kristal Berraklık  (±8%  gürültü, sinyal çok net)")
    print(" Kademe 2: Net                (±18% gürültü)")
    print(" Kademe 3: Orta               (±28% gürültü, v2.py benzeri)")
    print(" Kademe 4: Gürültülü          (±42% gürültü)")
    print(" Kademe 5: Kaotik             (±58% gürültü, sinyal neredeyse yok)")
    print()

    dataset_num = 0
    for kademe in range(1, 6):
        for period_name, period_cfg in TIME_PERIODS.items():
            dataset_num += 1
            n_days     = period_cfg["days"]
            start_date = period_cfg["start"]
            out_dir    = os.path.join(BASE_DIR, f"k{kademe}", period_name)
            cfg_name   = KADEME_CONFIGS[kademe]["name"]

            print(f"[{dataset_num:2d}/20] Kademe {kademe} ({cfg_name}) | {period_name} ({n_days} gün)")
            print(f"       → {out_dir}")

            rows = generate_dataset(kademe, period_name, n_days, start_date, out_dir)
            total_rows += rows
            print(f"       ✅ Tamamlandı ({rows:,} satır)")
            print()

    print("=" * 65)
    print(f" ✅ TÜM DATASETLER OLUŞTURULDU")
    print(f"    Toplam satır : {total_rows:,}")
    print(f"    Çıktı dizini : {BASE_DIR}")
    print("=" * 65)
    print()
    print(" Yapı:")
    for k in range(1, 6):
        cfg_name = KADEME_CONFIGS[k]["name"]
        print(f"  datasets/k{k}/ [{cfg_name}]")
        for p in TIME_PERIODS:
            print(f"    {p}/  → products.csv | daily_metrics.csv | metadata.json")

if __name__ == "__main__":
    main()
