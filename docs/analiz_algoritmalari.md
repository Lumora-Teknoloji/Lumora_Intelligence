# 🧠 Lumora Intelligence — Analiz Algoritmaları

**Versiyon:** v2.1 (2026-03-06)  
**Sistem:** Ensemble Tahmin Motoru — 8 Algoritma, 6 Katman

Bu doküman, ürünlerin trend olup olmadığını, yükselen mi yoksa düşen mi olduğunu anlamak için kullandığımız algoritmaları, seçilme nedenlerini ve birbirleriyle nasıl çalıştıklarını açıklar.

Sistemimiz tek bir zayıf noktaya bağlı kalmamak için **Ensemble (Hibrit)** bir yapı kullanır. Her algoritma farklı bir sinyal üretir; final skor bunların ağırlıklı toplamıdır.

---

## Sistem Mimarisi — 6 Katman

```
Ham Veri (daily_metrics)
       │
       ▼
[Katman 1] Data Preprocessing    → Z-Score anomali temizleme
       │
       ▼
[Katman 2] Feature Engineering   → 40+ özellik türetme
       │
       ├──────────────────────────────────────────────────────────
       ▼                          ▼                    ▼
[Katman 3a] CatBoost        [Katman 3b] Kalman    [Katman 3c] Prophet*
Composite demand score      Velocity + State      Mevsimsel bileşen
       │                          │                    │
       └──────────────────────────┘────────────────────┘
                                  │
                                  ▼
                    [Katman 4] Ensemble Skor
                    rank_reach×0.30 + CatBoost×0.30
                    + Kalman×0.20 + fav_growth×0.15
                                  │
                                  ▼
                    [Katman 5] Trend Label
                    TREND / POTANSİYEL / STABİL / DÜŞEN
                                  │
                                  ▼
                    [Katman 6] Feedback Loop
                    Kalman güncelle → Ağırlıkları optimize et
```

*Prophet opsiyonel (--no-prophet bayrağıyla devre dışı bırakılabilir)*

---

## KATMAN 1 — Z-Score Anormallik Tespiti

### Nerede Kullanılıyor?
Eğitim verisini modele sokmadan önceki temizlik aşamasında.

### Neden Kullanıyoruz?
Bir ürün o gün büyük bir indirime girerse veya influencer paylaşırsa sepet sayısı normalin 100 katına çıkıp ertesi gün normale döner. Bu gerçek bir trend değil, geçici bir "anormallik"tir. Eğer model bu veriyle eğitilirse "böyle zıplayanlar trenddir" diye yanlış öğrenir.

Z-Score, istatistiksel olarak aşırı sapan değerleri tespit edip eğitim setinden çıkarır.

### Nasıl Çalışır?
```python
z_score = (değer - ortalama) / standart_sapma

# Eşik aşılırsa → anomali flaglenir
if abs(z_score) > threshold:
    kayıt "error" olarak işaretlenir
```

### Pratik Etki
- Şu an DB'de: ~24.214 kayıt içinden geçirilir
- Parse hataları, duplicate'lar, scraper gap'leri → filtrelenir
- Mock veri testinde: Z-score'a takılmayan temiz sinyal üretildi

---

## KATMAN 2 — Feature Engineering (Özellik Mühendisliği)

### Nerede Kullanılıyor?
Tüm algoritmaların girdisi; ham veriyi 40+ anlamlı özelliğe dönüştürür.

### Neden Gerekli?
Ham veri (`price`, `favorite_count`, `absolute_rank`) tek başına yetersizdir. Modelin "bu ürün 3 günde 340'tan 8'e çıktı" sinyalini görmesi için türetilmiş özellikler gerekir.

### Üretilen Feature Grupları

#### Zaman Serisi Sinyalleri
| Feature | Açıklama |
|---------|----------|
| `rolling_avg_cart_count_7d` | 7 günlük hareketli sepet ortalaması |
| `rolling_avg_favorite_count_14d` | 14 günlük favori ortalaması |
| `favorite_growth_14d` | Son 7 gün / önceki 7 gün oranı (büyüme hızı) |
| `favorite_growth_3d` | Son 3 gün / önceki 7 gün (erken uyarı sinyali) |
| `momentum_cart_count_7d` | Sepet momentum oranı |

#### Rank Sinyalleri (En Güçlü)
| Feature | Açıklama |
|---------|----------|
| `absolute_rank` | Kategorideki gerçek sıralama |
| `rolling_avg_rank_3d` | 3 günlük rank ortalaması |
| `rank_velocity_7d` | 7 gün rank değişim hızı (negatif = iyileşiyor) |
| `rank_momentum_ratio` | Son 7 / önceki 7 rank oranı (<0.8 = güçlü yükseliş) |
| `rank_reach_mult` | `exp(-3.5 × rank_pct)` — 0.03 ile 1.0 arası erişim katsayısı |

#### Mevsimsel Feature'lar
| Feature | Açıklama |
|---------|----------|
| `season_factor` | Kategori bazlı mevsim ağırlığı (sinüs dalgası) |
| `month`, `week_of_year` | Zaman damgası |
| `is_weekend`, `day_of_week` | Hafta sonu etkisi |

#### Fiyat & İndirim
| Feature | Açıklama |
|---------|----------|
| `discount_rate` | İndirim oranı |
| `price_elast_boost` | İndirim kaynaklı talep artışı (%1 indirim = +%1.8 cart) |
| `price_vs_category_avg` | Kategori ortalamasına göre fiyat pozisyonu |
| `discount_intensity` | `low` / `medium` / `high` kategorik |

#### Stok & Ürün Yaşı
| Feature | Açıklama |
|---------|----------|
| `size_depletion_rate` | Stok tükenme hızı (güçlü talep sinyali) |
| `product_age_days` | Ürünün kaç gündür izlendiği |
| `is_new_product` | 7 günden genç mi? |

---

## KATMAN 3a — CatBoost (Gradient Boosting)

### Nerede Kullanılıyor?
Ana ML tahmin motoru. Composite demand score üretir.

### Neden CatBoost?
E-ticaret verilerinin çoğu kategorik metinlerden oluşur (Marka, Renk, Kumaş, Kalıp...). CatBoost bu kategorik verileri hiçbir ön işleme yapmadan doğrudan kullanır. Boş gelene (`unknown`) "bu verinin boş olması da bir işarettir" der — sistemi çökertmez.

### Hedef Değişken (Composite Target)

Tek sütun tahmin etmiyoruz. Birden fazla sinyalin ağırlıklı bileşiği:

```python
composite_target = (
    rank_reach_mult     * 0.45 +  # rank erişim (rising: 0.7-0.8, falling: 0.02-0.04)
    rank_imp            * 0.25 +  # 30 günlük rank değişim hızı
    log(fav_growth)     * 0.15 +  # favori büyüme oranı
    log1p(rating_count) * 0.08 +  # rating birikim
    log1p(cart_count)   * 0.07    # sepet (seyrek sinyal)
) × price_penalty × season_factor
```

### Neden Son 30 Gün?
Composite target `tail(30)` üzerinden hesaplanır. Son 7 gün yetmez çünkü rising ürün zaten son 7 günde tepede — değişim görünmez. 30 günde rank hareketi netleşir.

### Eğitim Ayarları
```python
CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    early_stopping_rounds=50,
    l2_leaf_reg=3,
)
```
- Time-based train/test split (%80 / %20)
- Mock veri test sonucu: **R² = 0.925**

### Feature Importance (Son Test)
| Sıra | Feature | Önem |
|------|---------|------|
| 1 | `favorite_count` | %18.1 |
| 2 | `rolling_avg_cart_count_7d` | %16.5 |
| 3 | `rolling_avg_favorite_count_14d` | %13.7 |
| 4 | `rolling_avg_favorite_count_7d` | %12.3 |
| 5 | `rolling_avg_rank_3d` | %4.8 |
| 6 | `rating_count` | %4.1 |
| 7 | `momentum_cart_count_7d` | %4.0 |

---

## KATMAN 3b — Kalman Filtresi

### Nerede Kullanılıyor?
İki ayrı seviyede:
1. **Ürün bazlı Kalman:** Her ürün için `cart_count` serisi → `velocity` (trend hızı)
2. **Kategori bazlı Kalman:** Her kategori için `engagement_score` → genel kategori durumu

### Neden Kalman?
Aslında bir uzay/havacılık algoritmasıdır. Pazarlamada her günkü veri çok gürültülüdür (botlar girebilir, yanlış tıklamalar olabilir). Kalman bu **gürültülerin arkasındaki gerçek ivmeyi** tahmin eder.

```
State vector: [trend_level, trend_velocity]
Her gün yeni veri gelince:
  1. Önceki tahmini tahmin et (prediction step)
  2. Gerçek değeri gör (observation)
  3. Farkı hesapla: innovation = actual - predicted
  4. Kalman Gain ile güncelle: state += gain × innovation
```

### Feedback Loop Entegrasyonu
```python
# Kullanıcı gerçek satışı girdikten sonra
engine.feedback(category="crop", actual_sales=30, predicted_demand=80)
# → Kalman anında hata marjını öğrenir
# → Büyük hata: R artar (daha temkinli tahmin)
# → Küçük hata: R azalır (daha güvenli tahmin)
```

### Velocity Yorumu
- `velocity > 0.3` → Yükselen ürün (trend sinyali)
- `velocity < -0.3` → Düşen ürün
- Şu an **383-714 ürün** velocity > 0.3 olarak tespit edildi

---

## KATMAN 3c — Prophet (Mevsimsel Ayrıştırma)

### Nerede Kullanılıyor?
Opsiyonel katman. `--no-prophet` bayrağıyla devre dışı bırakılabilir.

### Neden Kullanıyoruz?
Bazı kategoriler mevsimseldir: mont Kasım'da doğal olarak sat(ış)lar, yazlıklar Temmuz'da. Prophet bu mevsimsel bileşeni ham veriden ayrıştırır.

### Sezon Modeli (generate_csv_v2.py'de simüle edilen)
```python
sezon_tipleri = {
    "kış": {"peak_month": 11, "amplitude": 0.6},  # Kasım peak
    "yaz": {"peak_month":  7, "amplitude": 0.5},  # Temmuz peak
    "düğün": {"peak_months": [4, 6, 11, 12]},      # Çift peak
    "spor": {"amplitude": 0.2},                     # Zayıf mevsimsellik
}
season_factor = 1 + amplitude × sin(2π × (month - peak_month) / 12)
```

**Not:** Prophet 60+ günlük veri gerektiriyor. Şu an DB'de 8 günlük veri var → `--no-prophet` ile çalışıyoruz.

---

## KATMAN 4 — Ensemble Skor (Final Trend Skoru)

### Ağırlık Dağılımı
```python
trend_score = (
    score_reach  * 0.30 +  # rank_reach_mult — en güçlü ayırıcı
    score_cb     * 0.30 +  # CatBoost composite tahmin
    score_v      * 0.20 +  # Kalman ürün velocity
    score_fav    * 0.15 +  # favorite_growth (3d+14d karışımı)
    score_growth * 0.05    # cart_growth_pct bonus
) × 100
```

Her bileşen kendi kategorisi içinde 0-1'e normalize edilir → aynı kategorideki ürünler birbirleriyle kıyaslanır.

### Neden rank_reach_mult %30?
Bu sinyalin gücü:
- Rising ürün (180. gün): `rank_reach_mult ≈ 0.70-0.80`
- Falling ürün (180. gün): `rank_reach_mult ≈ 0.02-0.05`

18-40x ayrım! Diğer sinyallere kıyasla çok net bir discriminator.

### Etiket Eşikleri (Hybrid Quantile)
```python
p85 = max(scores.quantile(0.85), 62.0)  # TREND eşiği
p55 = max(scores.quantile(0.55), 42.0)  # POTANSİYEL eşiği
p25 = max(scores.quantile(0.25), 22.0)  # STABİL eşiği
```
Eşikler her kategori için ayrı hesaplanır → küçük kategoriler de doğru etiket alır.

### Son Test Sonuçları
| Etiket | Ürün Sayısı | Gerçek Rising İçeriği |
|--------|------------|----------------------|
| TREND | 714 | **%100** rising |
| POTANSİYEL | 1,084 | %83.2 rising |
| STABİL | 1,010 | %78.3 stable |
| DÜŞEN | 1,992 | %64.5 falling |

---

## KATMAN 5 — Override Mekanizmaları

### Aktif DÜŞEN Override
Rank hızla kötüleşiyor VE favori azalıyorsa → DÜŞEN etiketi zorla:
```python
rank_worsening = abs_rank_change_7d > 300   # 300+ pozisyon kaybı
fav_declining  = favorite_growth_14d < 0.80  # %20+ favori kaybı
→ DÜŞEN
```

### Viral Spike Override
7 günde %200+ favori artışı VE rank iyileşiyorsa → TREND etiketi zorla:
```python
is_fav_spike = fav_spike_7d > 2.0   # %200+
rank_improving_strong = rank_momentum_ratio < 0.8
→ TREND
```

---

## KATMAN 6 — Feedback Loop & Adaptive Ağırlıklar

### Nasıl Çalışır?
```
Kullanıcı: "TREND dediğin ürünlerden X sattı"
    ↓
Sistem: Tahmin vs Gerçek farkını hesapla
    ↓
Kalman: Anlık güncelleme (retraining gerekmez)
    ↓
Ceza sistemi: Büyük hata → prediction_score × 0.35
    ↓
Ağırlık güncelleme: Her 10 feedback'te bir
```

### Ceza Sistemi
| Hata Oranı | Ceza |
|-----------|------|
| > %70 | trend_score × 0.35 |
| %50–70 | trend_score × 0.55 |
| %30–50 | trend_score × 0.75 |
| < %30 | Ceza yok |

### Ağırlık Güncelleme (Adaptive)
```python
if avg_error > 50:
    weights = {"catboost": 0.5, "kalman_product": 0.3, "kalman_category": 0.2}
elif avg_error > 20:
    weights = {"catboost": 0.55, "kalman_product": 0.27, "kalman_category": 0.18}
else:
    weights = {"catboost": 0.6, "kalman_product": 0.24, "kalman_category": 0.16}
```

---

## Ek Algoritmalar

### Change Point Detection (CPD)
Zaman serisinde ani rejim değişikliklerini tespit eder. Feature olarak CatBoost'a verilir (`days_since_changepoint`). Ayrı bir çıktı üretmez.

### K-Prototypes Clustering
Hem sayısal hem kategorik verilerle ürün segmentasyonu. `cluster_id` feature'ı CatBoost'a girer. Şu an devre dışı (somut kazanım sağlamadı).

### CLIP (Görsel Eşleştirme)
Ürün fotoğraflarını 512 boyutlu vektörlere dönüştürür. "Bu fotoğrafa benzeyen ürünler hangileri?" sorusunu cevaplar. **GPU gerektirir, henüz aktif değil.**

### Bayesian Optimization
Grid Search'ün yerini alacak parametre optimizasyon yöntemi. Şu an hybrid quantile eşikler manuel ayarlanıyor.

---

## Bileşen Kısıtları & Açık Sorunlar

| Kısıt | Detay | Çözüm Yolu |
|-------|-------|-----------|
| Veri derinliği | Ortalama 1.7 gün/ürün | 30+ gün birikince CatBoost eğitilecek |
| cart_count %91 sıfır | Weak signal | rank + favorite sinyallerine odaklanıldı |
| Prophet inaktif | 8 günlük veri yetersiz | 60+ günde aktifleşecek |
| K-Prototypes kapalı | Kazanım az | İleriki versiyonda değerlendirilecek |

---

## Özet: Her Algoritmanın Rolü

| # | Algoritma | Rol | Sinyal Tipi |
|---|-----------|-----|-------------|
| 1 | **Z-Score** | Anormallik filtresi | Güvenlik |
| 2 | **Feature Engineering** | Ham veriyi zenginleştir | Altyapı |
| 3 | **rank_reach_mult** | Sıralama erişim katsayısı | En güçlü sinyal |
| 4 | **CatBoost** | Composite demand skoru | Tahmin motoru |
| 5 | **Kalman Filter** | Velocity + Gürültü filtresi | Anlık ivme |
| 6 | **Prophet** | Mevsimsel bileşen | Uzun dönem |
| 7 | **Ensemble** | Tüm sinyalleri birleştir | Final skor |
| 8 | **Feedback Loop** | Sistem öğrenir, düzelir | Adaptasyon |

---

*Lumora Intelligence v2.1 — 2026-03-06*
