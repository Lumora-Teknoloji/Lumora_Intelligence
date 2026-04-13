# 🧠 Lumora Intelligence — Kapsamlı Tasarım & Hedefler

**Proje:** Lumora Intelligence  
**Dizin:** `LancChain_Intelligence/`  
**Başlangıç:** 2026-02-25  
**Son güncelleme:** 2026-03-06  
**Durum:** Geliştirme aşamasında

---

## 0. 🔎 Mevcut Durum Özeti

### Veritabanı (2026-03-06 itibarıyla)

| Tablo | Satır | Durum |
|-------|------:|-------|
| products | 20,421 | ✅ Aktif |
| daily_metrics | 24,214 | ✅ Aktif |
| scraping_queue | 2,376 | ✅ Aktif |
| scraping_logs | 41 | ✅ Aktif |
| sales_forecasts | 0 | ❌ Hiç kullanılmamış |
| product_reviews | 0 | ❌ Boş |
| seo_rankings | 0 | ❌ Boş |

### Kategoriler & Veri Derinliği

| Kategori | Farklı Ürün | Kayıt |
|----------|------------:|------:|
| crop | 6,820 | 15,157 |
| tayt | 3,832 | 5,477 |
| grup | 2,902 | 2,902 |
| kadın abiye | 169 | 169 |

| Metric | Değer |
|--------|-------|
| Veri aralığı | 26 Şubat → 6 Mart 2026 (8 gün) |
| Ortalama gün/ürün | 1.7 gün |
| Max gün/ürün | 7 gün |
| **Duplicate kayıt** | **153** (günde 2 kez scrape edilen ürünler) |

### Metrik Kalitesi

| Kolon | Kullanılabilirlik |
|-------|-------------------|
| price, discounted_price | ✅ Mükemmel |
| search_rank, absolute_rank | ✅ %98.2 dolu — Ana sinyal |
| favorite_count | 🟡 %69.3 dolu |
| rating_count, avg_rating | 🟡 Kullanılabilir |
| cart_count, view_count | 🔴 %87-91 sıfır |
| engagement_score | ❌ Hiç hesaplanmamış |

### Test Sonuçları (2026-03-06, mock veri)

Gerçekçi 6 aylık mock veri (1.296.000 satır, 7.200 ürün) üzerinde:

| Metrik | Değer |
|--------|-------|
| TREND precision | **%100** (rising ürünlerin tamamı doğru) |
| DUSEN precision | %64.5 |
| POTANSIYEL içindeki rising oranı | %83.2 |
| Max trend_score | 92.1 |
| R² (CatBoost) | 0.925 |

**Algoritma stack:** CatBoost + Kalman + Z-Score + rank_reach_mult feature

---

## 1. 🗄️ Tarihsel Veri — Günlük Kayıt

| Tarih | Kayıt | Kategoriler |
|-------|------:|-------------|
| 2026-02-26 | 169 | kadın abiye |
| 2026-02-28 | 1,849 | crop, tayt |
| 2026-03-01 | 4,393 | crop, tayt |
| 2026-03-02 | 6,392 | crop, grup, tayt |
| 2026-03-03 | 1,553 | crop, tayt |
| 2026-03-04 | 4,189 | crop |
| 2026-03-05 | 3,169 | crop, tayt |
| 2026-03-06 | 2,500+ | devam ediyor |

---

## 2. 🏷️ Sistem Tanımı

**Amaç:** Trendyol'daki pazar verilerini işleyerek "ne üretelim, ne kadar üretelim, ne zaman üretelim" sorusuna veri tabanlı cevap vermek  
**Bağımsızlık:** LangChain backend'den tamamen ayrı çalışır (:8001)  
**Öğrenme:** Gerçek satış verileri girildikçe kendini günceller

### Mimari

```
[Trendyol Scraper]
        │
        ▼
[lumora_db : daily_metrics, products]
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
[Lumora Intelligence :8001]          [LangChain Backend :8000]
        │                                      │
        ├── Otomatik (APScheduler)             ├── Kullanıcıya sunar
        │   ├── Her gece 02:00                 └── /intelligence/... çağırır
        │   │   → Rank momentum hesapla
        │   │   → Category signals güncelle
        │   │   → Alerts üret
        │   │   → products.trend_score güncelle
        │   └── Her Pazar 03:00
        │       → CatBoost yeniden eğit
        │
        └── API Endpoint'leri
            ├── POST /analyze {product_id}
            ├── GET  /predict {category, top_n}
            ├── POST /trigger {scope, priority}
            ├── POST /feedback {production_id, sold}
            ├── GET  /alerts
            └── GET  /health
```

---

## 3. ✅ Çekirdek Dosyalar

### Engine (AI algoritmaları)
- `algorithms/catboost_model.py` — Ana ML modeli (rank_reach_mult ile güçlendirildi)
- `algorithms/kalman.py` — Anlık adaptasyon
- `algorithms/zscore.py` — Anomali temizleme
- `algorithms/prophet_model.py` — Mevsimsellik
- `algorithms/changepoint.py` — Trend kırılma tespiti
- `algorithms/clustering.py` — Ürün segmentasyonu
- `algorithms/clip_model.py` — Görsel analiz (CLIP, henüz aktif değil)
- `engine/features.py` — Feature engineering pipeline
- `engine/predictor.py` — Ensemble tahmin motoru

### Veri
- `data/generate_csv_v2.py` — 6 aylık gerçekçi mock veri üreteci
- `data/output_v2/products.csv` — 7,200 ürün
- `data/output_v2/daily_metrics.csv` — 1,296,000 satır

### Test & Analiz
- `run_csv_test.py` — CSV → engine adapter + test koşucu
- `results/csv_test_results.csv` — Son tahmin sonuçları

---

## 4. 💡 Özellik Öncelik Listesi

### 4.1 ⭐ Rank Momentum (Veri hazır, hemen başlanabilir)

```python
rank_change_3d = 3_gün_önceki_rank - bugünkü_rank
momentum_score = tanh(rank_change_3d / 100)
# +0.91 → GÜÇLÜ YÜKSELİŞ, -0.97 → DÜŞÜŞ
```

**DB kolonu eklenecek:**
```sql
ALTER TABLE daily_metrics
  ADD COLUMN rank_change_1d INTEGER,
  ADD COLUMN rank_change_3d INTEGER,
  ADD COLUMN rank_velocity   FLOAT,
  ADD COLUMN momentum_score  FLOAT CHECK (momentum_score BETWEEN -1 AND 1),
  ADD COLUMN is_new_entrant  BOOLEAN DEFAULT FALSE;
```

### 4.2 ⭐⭐ Kategori Sıcaklık Haritası

```python
category_heat = (
    rising_count   / total_products * 0.5 +
    new_entrants   / total_products * 0.3 +
    avg_fav_change / 1000           * 0.2
) * 2 - 1   # -1 soğuyor, +1 ısınıyor
```

**Tablo:** `category_daily_signals` (date, search_term, heat, total_products, rising_count...)

### 4.3 ⭐⭐ Yeni Girişler Tespiti

```
Ürün X: Dün top 10.000'de → Bugün top 50'de girdi
→ ALERT: "Viral başlangıç tespit edildi"
```

### 4.4 ⭐⭐⭐ Stil Trendleri (JSONB attributes analizi)

Top 50 ürünün renk/kumaş/kalıp dağılımını haftalık analiz:
```
Bu hafta crop top50: Siyah %62 ↑+12% ← TREND
                     Pamuk  %71 ↑+8%  ← TREND
```

**Tablo:** `style_trends` (date, search_term, attribute_key, attribute_value, pct, is_trending)

### 4.5 ⭐⭐⭐ Uyarı Sistemi

| Tür | Tetikleme |
|-----|----------|
| `rank_spike` | 3 günde >500 rank iyileşme |
| `rank_drop` | 3 günde >300 rank kötüleşme |
| `viral_start` | İlk kez top50'ye giriş |
| `category_heat` | category_heat > 0.8 |
| `new_brand_entry` | Yeni marka top100'e girdi |

### 4.6 ⭐⭐⭐⭐ Kalman Filter (30 gün sonrası anlamlı)

Her kategori için ayrı state, her gün anlık güncelleme.  
Feedback gelince otomatik hata toleransı ayarlanır.

### 4.7 ⭐⭐⭐⭐⭐ CatBoost (30+ gün + feedback gerekli)

**Şu an mock veri üzerinde R²=0.925 başarı sağlandı.**  
Gerçek veriye bağlamak için: min. 30 günlük `daily_metrics` + `actual_sales` feedback'i gerekiyor.

**Key features:**
- `rank_reach_mult` (0.03–1.0) — En güçlü sinyal ✅ aktif
- `season_factor`, `price_elast_boost` ✅ aktif
- `favorite_growth_14d`, `rolling_avg_*` ✅ aktif
- `dominant_color`, `fabric_type` — products tablosundan gelecek

---

## 5. 🔄 Feedback Loop

```
Trendyol rank ↑ → "trend" diyor
AMA senin üretip satmadığını bilmiyor.
→ Sistem sonsuza kadar aynı hatayı yapabilir.
```

**Gerekli tablolar:**

```sql
-- Üretim kararları
CREATE TABLE production_decisions (
    id              SERIAL PRIMARY KEY,
    product_id      INTEGER REFERENCES products(id),
    search_term     VARCHAR(100),
    predicted_score FLOAT,
    decision        VARCHAR(20),  -- 'produce'/'skip'/'wait'
    quantity        INTEGER,
    decided_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Gerçek satış sonuçları
CREATE TABLE actual_sales (
    id                  SERIAL PRIMARY KEY,
    production_id       INTEGER REFERENCES production_decisions(id),
    sold_quantity       INTEGER,
    produced_quantity   INTEGER,
    sell_through_rate   FLOAT,  -- sold/produced
    feedback_date       DATE DEFAULT CURRENT_DATE
);
```

**Öğrenme eğrisi:**
```
0  feedback: Evrensel Trendyol sinyali → ~%55-60 isabetlilik
10 feedback: Kalman hata marjı öğrendi
30 feedback: CatBoost ilk retraining → ~%70-75
100 feedback: Fiyat bandı + müşteri profili → ~%80-85
1 yıl:       Mevsim hafızası → ~%85-90
```

---

## 6. 🗄️ Veritabanı Değişiklikleri

### Yeni tablolar (öncelik sırasına göre)

| Tablo | Öncelik |
|-------|---------|
| `production_decisions` | 🔴 İlk |
| `actual_sales` | 🔴 İlk |
| `category_daily_signals` | 🔴 İlk |
| `trend_alerts` | 🟡 İkinci |
| `style_trends` | 🟡 İkinci |
| `categories` (kayıt defteri) | 🟡 İkinci |
| `model_versions` | 🟢 Üçüncü |
| `intelligence_results` | 🟢 Üçüncü |

### Mevcut tablolara eklentiler

```sql
-- daily_metrics
ALTER TABLE daily_metrics
  ADD COLUMN rank_change_1d  INTEGER,
  ADD COLUMN rank_change_3d  INTEGER,
  ADD COLUMN rank_velocity   FLOAT,
  ADD COLUMN momentum_score  FLOAT;

-- products
ALTER TABLE products
  ADD COLUMN dominant_color  VARCHAR(50),
  ADD COLUMN fabric_type     VARCHAR(50),
  ADD COLUMN fit_type        VARCHAR(50),
  ADD COLUMN trend_score     FLOAT,
  ADD COLUMN trend_direction VARCHAR(20),
  ADD COLUMN last_scored_at  TIMESTAMPTZ;
```

### Kritik indeksler

```sql
CREATE INDEX idx_dm_product_date  ON daily_metrics(product_id, DATE(recorded_at));
CREATE INDEX idx_dm_search_date   ON daily_metrics(search_term, DATE(recorded_at));
CREATE INDEX idx_dm_momentum      ON daily_metrics(momentum_score DESC) WHERE momentum_score IS NOT NULL;
CREATE INDEX idx_products_trend   ON products(trend_score DESC) WHERE trend_score IS NOT NULL;
```

---

## 7. 📅 İmplementasyon Yol Haritası

### Faz 1 — Temel Sinyaller (VERİ HAZIR, hemen başlanabilir)
- [ ] `daily_metrics`'e rank_change kolonlarını ekle
- [ ] Rank momentum nightly batch script'i yaz
- [ ] `category_daily_signals` tablosunu oluştur
- [ ] `products` tablosuna JSONB'den `dominant_color`, `fabric_type` extract et
- [ ] 153 duplicate kaydı temizle (`ON CONFLICT` guard'ını düzelt)

### Faz 2 — Feedback Altyapısı (Sistem öğrenmeye başlar)
- [ ] `production_decisions` tablosunu oluştur
- [ ] `actual_sales` tablosunu oluştur
- [ ] Kalman'ı feedback'e bağla
- [ ] `trend_alerts` tablosunu oluştur

### Faz 3 — Engine → DB Entegrasyonu
- [ ] `db/reader.py`: daily_metrics + products okuma
- [ ] `db/writer.py`: intelligence_results yazma
- [ ] Feature pipeline gerçek veriye bağla
- [ ] CatBoost'u 30+ günlük gerçek veriyle eğit
- [ ] `style_trends` tablosu + JSONB analizi

### Faz 4 — CatBoost Entegrasyonu (30+ gün + feedback gerekli)
- [ ] CatBoost ilk eğitimini gerçek verilerle yap
- [ ] Haftalık otomatik yeniden eğitim (APScheduler)
- [ ] SHAP açıklamaları: "neden bu ürünü önerdi?"

### Faz 5 — API & Otomasyon
- [ ] FastAPI skeleton (`api/` klasörü)
- [ ] `/health`, `/predict`, `/analyze`, `/trigger`, `/feedback`, `/alerts`
- [ ] APScheduler: gece 02:00 toplu skor, Pazar 03:00 retraining
- [ ] LangChain → Lumora Intelligence HTTP entegrasyonu
- [ ] Dockerfile + sunucuya deploy

### Faz 6 — İleri Seviye
- [ ] Prophet mevsimsellik (60+ gün sonrası)
- [ ] K-Prototypes ürün segmentasyonu
- [ ] Web dashboard
- [ ] CLIP görsel analiz (GPU gerektirir)

---

## 8. 🔀 Dinamik Kategori Sistemi

### Temel Prensip: Sınırsız kategori, sıfır kod değişikliği

```python
# Yeni kategori gelince otomatik adapte olur
model = ModelRegistry.get_or_create(category)
```

### Kategori Yaşam Döngüsü

```
COLD (0-30 gün)    → Rank momentum + Favori sinyal
WARMING (30-90 gün) → Kalman aktif, CatBoost ilk eğitimi
HOT (90-365 gün)   → Tam ensemble, feedback öğrenmesi
MATURE (1 yıl+)    → Prophet mevsimsellik, %85-90 isabetlilik
```

### Kategori Tiplerine Göre Parametre

| Tip | Örnek | Z-Score eşiği |
|-----|-------|---------------|
| `mid_fashion` | crop, tayt | 3.0 |
| `premium` | kadın abiye | 2.0 (hassas) |
| `heterogeneous` | grup | 5.0 (geniş) |

**Kategori kayıt defteri:** `categories` tablosu (status, data_days, feedback_count, kalman_state JSONB...)

---

## 9. ⚠️ Kısıtlar & Riskler

| Kısıt | Detay | Çözüm |
|-------|-------|-------|
| Veri derinliği | Ort. 1.7 gün/ürün — CatBoost için min. 30 gün | Faz 1-2'yi 30 gün birikince başlat |
| Sıfır metrikler | cart/view %87-91 sıfır | Rank + favori sinyallerine odaklan |
| 153 duplicate | Aynı günde 2 kez scrape | `ON CONFLICT` guard düzelt |
| Engagement NULL | `engagement_score` hiç hesaplanmamış | Nightly batch'te hesapla |
| Ground truth eksik | Gerçek satış verisi yok | `production_decisions` + `actual_sales` |

---

## 10. 📋 Dosya Yapısı (Hedef)

```
LancChain_Intelligence/
├── main.py                     # FastAPI + Scheduler
├── config.py
├── requirements.txt
│
├── api/routes/
│   ├── analyze.py, predict.py, trigger.py
│   ├── feedback.py             ← KRİTİK
│   └── alerts.py, health.py
│
├── engine/
│   ├── features.py             ← MEVCUT ✅
│   ├── predictor.py            ← MEVCUT ✅
│   └── pipeline.py             (eklenecek)
│
├── algorithms/                 ← HEPSI MEVCUT ✅
│   ├── catboost_model.py, kalman.py, zscore.py
│   ├── prophet_model.py, changepoint.py
│   └── clustering.py, clip_model.py
│
├── db/
│   ├── connection.py           (eklenecek)
│   ├── reader.py               (eklenecek)
│   └── writer.py               (eklenecek)
│
├── scheduler/jobs.py, triggers.py
├── data/generate_csv_v2.py     ← MEVCUT ✅
└── models/                     ← Kategoriye göre dinamik
    ├── crop/catboost_v*.cbm
    └── tayt/catboost_v*.cbm
```

---

*Birleştirilmiş doküman: 2026-03-06 | targets.md + targets2.md*  
*Lumora Intelligence v1.1 — Tasarım & Hedefler*
