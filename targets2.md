# 📊 Lumora Intelligence — Kapsamlı Tasarım Raporu

**Tarih:** 2026-03-05
**Versiyon:** v1.0
**Durum:** Planlama aşaması — henüz uygulanmadı

---

## 1. 🗄️ Mevcut Veritabanı Durumu

### 1.1 Tablo Boyutları (2026-03-05 itibarıyla)

| Tablo | Satır | Boyut | Durum |
|-------|------:|-------|-------|
| products | 19,987 | 31 MB | ✅ Aktif |
| daily_metrics | 21,714 | 5 MB | ✅ Aktif |
| scraping_queue | 2,210 | 3 MB | ✅ Aktif |
| scraping_logs | 38 | 88 kB | ✅ Aktif |
| scraping_tasks | 1 | 48 kB | ⚠️ `is_active=False` |
| sales_forecasts | **0** | — | ❌ Hiç kullanılmamış |
| product_reviews | **0** | — | ❌ Boş |
| seo_rankings | **0** | — | ❌ Boş |

### 1.2 Veri Geçmişi (Tarih Bazlı)

| Tarih | Kayıt | Ürün | Kategoriler |
|-------|------:|-----:|-------------|
| 2026-02-27 | 169 | 169 | kadın abiye |
| 2026-02-28 | 519 | 519 | crop, tayt |
| 2026-03-01 | 5,723 | 5,719 | crop, tayt |
| 2026-03-02 | 6,256 | 6,256 | crop, grup, tayt |
| 2026-03-03 | 843 | 843 | crop, tayt |
| 2026-03-04 | 4,147 | 3,869 | crop |
| 2026-03-05 | 4,057 | 3,951 | crop, tayt |

**Not:** 2026-03-04'te tayt hiç çalışmadı — 51 saatlik boşluk.

### 1.3 Ürün Veri Derinliği

| Gün Sayısı | Ürün | Oran |
|-----------|-----:|-----:|
| 5 gün | 4 | %0.03 |
| 4 gün | 527 | %3.8 |
| 3 gün | 1,717 | %12.5 |
| 2 gün | 2,897 | %21.1 |
| **1 gün** | **8,588** | **%62.5** |

**CatBoost için min. 30 gün, Prophet için min. 60 gün gerekiyor.**

### 1.4 Metrik Kalitesi

| Kolon | Dolu (>0) | Sıfır | NULL | Kullanılabilirlik |
|-------|----------:|------:|-----:|-------------------|
| price | %100 | %0 | %0 | ✅ Mükemmel |
| discounted_price | %100 | %0 | %0 | ✅ Mükemmel |
| search_rank | %98.2 | — | %1.8 | ✅ Ana sinyal |
| absolute_rank | %98.2 | — | %1.8 | ✅ Ana sinyal |
| page_number | %98.2 | — | %1.8 | ✅ İyi |
| favorite_count | %69.3 | %30.7 | — | 🟡 Kullanılabilir |
| rating_count | %82.8 | %17.2 | — | 🟡 Kullanılabilir |
| avg_rating | %78.5 | %21.5 | — | 🟡 Kullanılabilir |
| discount_rate | %45.6 | %54.4 | — | 🟡 Kısmi |
| cart_count | **%9** | **%91** | — | 🔴 Güvenilmez |
| view_count | **%13** | **%87** | — | 🔴 Güvenilmez |
| engagement_score | — | — | **%100** | ❌ Hiç hesaplanmamış |
| popularity_score | — | — | **%100** | ❌ Hiç hesaplanmamış |
| sales_velocity | — | — | **%100** | ❌ Hiç hesaplanmamış |
| available_sizes | — | — | **%100** | ❌ Scraper çekmiyor |

### 1.5 Tespit Edilen Kritik Sorunlar

1. **289 duplicate kayıt** — 03-04'te crop için hem gece (00:00-09:18) hem akşam (22:36-23:59) tur çalışmış, aynı ürünler 2 kez kuyruğa girmiş. `ON CONFLICT` guard'ı atlatılıyor.
2. **15 zombie log** — `running` yazıyor ama 4-5 gündür crash edilmiş. `finished_at` güncellenmemiş.
3. **51 saatlik tayt boşluğu** — 03-03 01:19 → 03-05 04:09 arası tayt çalışmamış.
4. **6,468 ürün (%32) hiç izlenmemiş** — Linker bulmuş, worker işlememiş.
5. **scraping_tasks `is_active=False`** — Otomasyon yok, sistem manuel tetikleniyor.

---

## 2. 🏷️ Sistem Tanımı

**Ad:** Lumora Intelligence
**Amaç:** Trendyol'daki pazar verilerini işleyerek "ne üretelim, ne kadar üretelim, ne zaman üretelim" sorusuna veri tabanlı cevap vermek
**Bağımsızlık:** LangChain backend'den tamamen ayrı çalışır (:8001)
**Öğrenme:** Gerçek satış verileri girildikçe kendini günceller

---

## 3. 💡 Değer Üretecek Özellikler (Öncelik Sırası)

### 3.1 ⭐ Rank Momentum — En Değerli Sinyal (İLK YAPILACAK)

**Ne yapar:** Ürünün arama sıralamasının ne kadar hızlı değiştiğini hesaplar.

**Neden ilk sırada:**
- Veri zaten mevcut, %98.2 dolu
- Hiçbir ML modeline gerek yok
- Anlık değer üretir
- Tüm diğer algoritmaların en kritik girdisi

**Formül:**
```python
rank_change_1d = dünkü_absolute_rank - bugünkü_absolute_rank
rank_change_3d = 3_gün_önceki_rank - bugünkü_rank
rank_velocity  = rank_change_3d / 3   # günlük ortalama iyileşme

# Normalize skor (0-1)
# Positif = iyileşiyor, Negatif = kötüleşiyor
momentum_score = tanh(rank_velocity / 100)
```

**Örnek:**
```
Ürün A:
  3 gün önce: absolute_rank = 850
  Bugün:      absolute_rank = 120
  rank_change_3d = 730
  momentum_score = +0.91 → GÜÇLÜ YÜKSELİŞ SİNYALİ

Ürün B:
  3 gün önce: absolute_rank = 50
  Bugün:      absolute_rank = 380
  rank_change_3d = -330
  momentum_score = -0.97 → DÜŞÜŞ SİNYALİ, DURDUR
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

**Nasıl hesaplanır (her gece):**
```sql
UPDATE daily_metrics dm
SET
  rank_change_1d = prev_day.absolute_rank - dm.absolute_rank,
  rank_change_3d = prev_3d.absolute_rank  - dm.absolute_rank,
  momentum_score = TANH((prev_3d.absolute_rank - dm.absolute_rank) / 100.0),
  is_new_entrant = (prev_7d.product_id IS NULL AND dm.absolute_rank < 200)
FROM daily_metrics prev_day
LEFT JOIN daily_metrics prev_3d ON ...
LEFT JOIN daily_metrics prev_7d ON ...
WHERE DATE(dm.recorded_at) = CURRENT_DATE;
```

---

### 3.2 ⭐⭐ Kategori Sıcaklık Haritası

**Ne yapar:** Her kategorinin günlük "mood"unu sayısal olarak ifade eder.

**Formül:**
```python
category_heat = (
    rising_count   / total_products * 0.5 +   # yükselen ürün oranı
    new_entrants   / total_products * 0.3 +   # yeni giriş oranı
    avg_fav_change / 1000           * 0.2     # favori artışı
) * 2 - 1   # -1 ile +1 arasına normalize et
```

**Değer aralığı:**
- `+0.7` ile `+1.0` → Kategori ISIYOR → Bu kategoride üret
- `0.0` ile `+0.7` → Nötr → İzle
- `-1.0` ile `0.0` → Kategori SOĞUYOR → Dikkatli ol

**Tablo:**
```sql
CREATE TABLE category_daily_signals (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    search_term     VARCHAR(100) NOT NULL,

    -- Hacim metrikleri
    total_products  INTEGER,
    new_entrants    INTEGER,       -- İlk kez top 200'e giren ürün sayısı
    dropped_out     INTEGER,       -- Top 200'den çıkan ürün sayısı

    -- Sıralama hareketleri
    avg_rank_change     FLOAT,     -- Ortalama rank değişimi (+ = iyileşiyor)
    median_rank_change  FLOAT,
    rising_count        INTEGER,   -- rank_change_1d > 0 olan ürün sayısı
    falling_count       INTEGER,   -- rank_change_1d < 0 olan ürün sayısı
    stable_count        INTEGER,   -- |rank_change_1d| < 5 olan ürün sayısı

    -- Fiyat
    avg_price           NUMERIC,
    median_price        NUMERIC,
    avg_discount_rate   FLOAT,

    -- Favori (güvenilir metrik)
    avg_favorite_count  FLOAT,
    avg_fav_change      FLOAT,     -- Önceki güne göre ortalama favori değişimi

    -- Ana sinyal
    category_heat       FLOAT,     -- -1 soğuyor, +1 ısınıyor
    heat_trend          VARCHAR(20), -- 'heating', 'cooling', 'stable', 'volatile'

    UNIQUE(date, search_term)
);
```

---

### 3.3 ⭐⭐ Yeni Girişler Tespiti

**Ne yapar:** Bir ürün daha önce hiç görünmemişken bugün top 200'e giriyorsa flagler.

**Önemi:** Viral başlangıç veya yeni rakip tespiti için kritik.

**Senaryo 1 — Viral başlangıç:**
```
Dün:   Ürün X top 10.000'de (veya hiç yoktu)
Bugün: Ürün X top 50'ye girdi
→ ALERT: "Viral başlangıç tespit edildi"
```

**Senaryo 2 — Yeni rakip:**
```
Dün:   crop top100'de World Fashion Trends yoktu
Bugün: 5 yeni World Fashion Trends ürünü top100'e girdi
→ ALERT: "World Fashion Trends agresif giriş yapıyor"
```

---

### 3.4 ⭐⭐⭐ Stil Trendleri (JSONB attributes analizi)

**Ne yapar:** 20k ürünün JSONB attribute'larını analiz ederek hangi renk/kumaş/kalıp kombinasyonunun trend olduğunu bulur.

**Mevcut JSONB örneği:**
```json
{
  "Renk": "Siyah",
  "Materyal": "Pamuk",
  "Kalıp": "Slim Fit",
  "Yaka": "Bisiklet Yaka",
  "Kol": "Uzun Kol",
  "Sezon": "Kış"
}
```

**Analiz mantığı:**
```python
# Top 50 ürünün attribute dağılımı
top50 = daily_metrics[absolute_rank <= 50]
top50_colors = top50.attributes.str['Renk'].value_counts()

# 7 gün öncesiyle karşılaştır
prev_top50_colors = ...

# En çok yükselen renk/kumaş kombinasyonu = TREND
trend_signal = (current_pct - prev_pct) / prev_pct
```

**Beklenen çıktı:**
```
Bu hafta crop top50'de:
  Renk:   Siyah %62 ↑+12%  ← TREND
          Beyaz %18 ↓-5%
          Kırmızı %8 ↑+3%

  Kumaş:  Pamuk %71 ↑+8%   ← TREND
          Polyester %15 ↓-11%

Sonuç: "Siyah pamuk ürünler bu hafta crop'ta baskın"
```

**Tablo:**
```sql
CREATE TABLE style_trends (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    search_term     VARCHAR(100) NOT NULL,
    rank_band       VARCHAR(20) NOT NULL,  -- 'top50', 'top200', 'top500'

    attribute_key   VARCHAR(50),    -- 'Renk', 'Materyal', 'Kalıp'
    attribute_value VARCHAR(100),   -- 'Siyah', 'Pamuk', 'Slim Fit'

    product_count   INTEGER,        -- Bu haftaki top_band'daki ürün sayısı
    pct_of_band     FLOAT,          -- Top band içindeki oran (%)

    prev_pct        FLOAT,          -- Geçen hafta aynı değerin oranı
    pct_change      FLOAT,          -- Değişim yüzdesi
    is_trending     BOOLEAN,        -- +%15 eşiğini geçti mi?
    is_declining    BOOLEAN,        -- -%15 eşiğini geçti mi?

    UNIQUE(date, search_term, rank_band, attribute_key, attribute_value)
);
```

**`products` tablosuna eklenecek (JSONB'den extract):**
```sql
ALTER TABLE products
  ADD COLUMN dominant_color  VARCHAR(50),   -- JSONB'den: attributes->>'Renk'
  ADD COLUMN fabric_type     VARCHAR(50),   -- JSONB'den: attributes->>'Materyal'
  ADD COLUMN fit_type        VARCHAR(50),   -- JSONB'den: attributes->>'Kalıp'
  ADD COLUMN collar_type     VARCHAR(50),   -- JSONB'den: attributes->>'Yaka'
  ADD COLUMN trend_score     FLOAT,         -- Son hesaplanan trend skoru
  ADD COLUMN trend_direction VARCHAR(20),   -- 'rising', 'falling', 'stable', 'viral'
  ADD COLUMN last_scored_at  TIMESTAMPTZ;

-- Mevcut veriyi doldur
UPDATE products SET
  dominant_color = attributes->>'Renk',
  fabric_type    = attributes->>'Materyal',
  fit_type       = attributes->>'Kalıp',
  collar_type    = attributes->>'Yaka'
WHERE attributes IS NOT NULL;
```

---

### 3.5 ⭐⭐⭐ Uyarı Sistemi (trend_alerts)

**Ne yapar:** Önemli bir hareket tespit edildiğinde kayıt oluşturur.

**Alert türleri:**

| Tür | Tetikleme | Açıklama |
|-----|----------|----------|
| `rank_spike` | 3 günde >500 rank iyileşme | Ürün hızla yukaru fırlıyor |
| `rank_drop` | 3 günde >300 rank kötüleşme | Ürün hızla düşüyor |
| `viral_start` | İlk kez top50'ye giriş | Viral başlangıç |
| `new_brand_entry` | Yeni marka top100'e girdi | Yeni rakip |
| `price_drop_rank` | Fiyat düştü + rank iyileşti | Fiyat-rank ilişkisi |
| `category_heat` | category_heat > 0.8 | Kategori çok ısındı |
| `competitor_surge` | Bir marka top200'de ani artış | Rakip agresif giriş |

**Tablo:**
```sql
CREATE TABLE trend_alerts (
    id              SERIAL PRIMARY KEY,
    product_id      INTEGER REFERENCES products(id),
    search_term     VARCHAR(100),
    alert_type      VARCHAR(50) NOT NULL,
    severity        VARCHAR(20) NOT NULL,  -- 'low', 'medium', 'high', 'critical'
    detected_at     TIMESTAMPTZ DEFAULT NOW(),

    -- Bağlam
    details         JSONB,
    -- Örnek: {"from_rank": 850, "to_rank": 12, "change_pct": 98.6,
    --         "days": 3, "price": 249.0, "brand": "ETHIQUET"}

    -- Durum
    is_read         BOOLEAN DEFAULT FALSE,
    is_acted_upon   BOOLEAN DEFAULT FALSE,  -- "Bu alerta göre ürettim" mi?
    action_taken    TEXT,                   -- "100 adet ürettim"
    expires_at      TIMESTAMPTZ             -- Ne zaman geçerliliğini yitirir
);
```

---

### 3.6 ⭐⭐⭐⭐ Kalman Filter (30 gün sonra anlamlı)

**Ne yapar:** Her kategori için anlık bir "trend seviyesi" ve "değişim hızı" tahmin eder. Yeni veri geldikçe otomatik güncellenir.

**Nasıl çalışır:**
```
State vector: [trend_level, trend_velocity]
  trend_level    = şu anki trend gücü tahmini
  trend_velocity = trendin hız trendi (ivme)

Her gün yeni veri gelince:
  1. Önceki tahmini tahmin et (prediction step)
  2. Gerçek değeri gör (observation)
  3. Farkı hesapla (innovation = actual - predicted)
  4. Kalman Gain ile ağırlıklı ortalama al (update step)
  5. Yeni state = eski tahmin + (Gain × innovation)
```

**Kalman'ın asıl gücü — feedback loop:**
```
Sistem "100 adet satarsın" dedi, gerçekte 30 sattı
Hata = 70

Kalman gain ayarlıyor:
  70 birim hata var → bu kategoride
  "gerçek satışa daha fazla güven,
   kendi tahminime daha az güven"
  → Bir sonraki tahmin daha temkinli
```

---

### 3.7 ⭐⭐⭐⭐⭐ CatBoost (30+ gün + feedback gerekli)

**Ne yapar:** Geçmişteki tüm ürünlerin özelliklerini ve gerçek satış sonuçlarını öğrenerek yeni ürünler için talep tahmini yapar.

**Özellikler (features):**
```
Ürün özellikleri (products.attributes JSONB):
  - dominant_color     (siyah, beyaz, kırmızı...)
  - fabric_type        (pamuk, polyester, viskon...)
  - fit_type           (slim, bol, regular...)
  - collar_type        (bisiklet, v, polo...)
  - price_segment      (düşük/orta/yüksek - median'a göre)
  - brand_tier         (büyük marka mı küçük mü?)

Rank sinyalleri (daily_metrics):
  - momentum_score     (rank momentum)
  - rank_change_3d     (3 günlük rank değişimi)
  - is_new_entrant     (yeni giren mi?)
  - category_heat      (kategori sıcaklığı)
  - absolute_rank      (kategorideki genel konum)

Sosyal sinyaller:
  - favorite_count     (favori sayısı)
  - avg_rating         (ortalama puan)
  - rating_count       (puan sayısı)

Mevsimsel (Prophet'tan):
  - seasonal_phase     (rising/peak/falling/trough)
  - days_to_season_peak

Cluster bilgisi (K-Prototypes'tan):
  - cluster_label      ('star', 'rising', 'stable', 'declining')
```

**Hedef değişken (target):**
```
Reel satış: sell_through_rate = sold / produced
  0.0 = hiç satmadı
  1.0 = tamamı satıldı
```

**Minimum veri ihtiyacı:**
- 30+ günlük daily_metrics geçmişi
- 30+ adet `actual_sales` feedback kaydı

**Periyodik yeniden eğitim:**
- Her Pazar gece 03:00'da otomatik
- 100+ yeni feedback noktası birikimleri tetikleyebilir
- Her eğitim `model_versions` tablosuna kaydedilir

---

### 3.8 ⭐⭐⭐⭐⭐ CLIP Görsel Analiz (En Sona Bırak)

**Ne yapar:** Ürün fotoğraflarını 512 boyutlu vektörlere dönüştürür. "Şu an en çok satan ürünlere visually benzeyen ürünler hangileri?" sorusunu cevaplar.

**GPU gerektirir, kurulumu karmaşık, veri hazır olmadan anlamsız.**

---

## 4. 🔄 Feedback Loop — Öğrenen Sistem

### 4.1 Neden Gerekli?

Sistem şu an sadece Trendyol'u görüyor:
```
Trendyol rank ↑  →  "trend" diyor
```

Ama senin üretip satıp satmadığını bilmiyor. `sales_forecasts` tablosu tamamen boş.

**→ Algoritma sonsuza kadar aynı hatayı yapabilir.**

### 4.2 Temel Gerçek

```
Trendyol'da trend  ≠  Lumora'da trend

Trendyol: 10 milyon+ müşteri → farklı yaş, bölge, gelir
Lumora:   Belirli bir kitle → senin müşteri profilin

Örnek:
  Trendyol'da polyester beyaz tayt → rank 8 (çok satıyor)
  Lumora müşterisi → pamuk siyah tayt tercih ediyor

Sistem 50+ feedback sonrası bunu öğrenir ve
polyester beyaz taytı önermez, senin kitlen için önermez.
```

### 4.3 Gerekli Tablolar

**Tablo 1: `production_decisions`**
```sql
CREATE TABLE production_decisions (
    id              SERIAL PRIMARY KEY,
    decided_at      TIMESTAMPTZ DEFAULT NOW(),
    product_id      INTEGER REFERENCES products(id),
    search_term     VARCHAR(100),

    -- Sistem ne dedi?
    predicted_score     FLOAT,          -- 0.87 (trend skoru)
    predicted_demand_7d INTEGER,        -- "7 günde 80 adet satarsın" tahmini
    confidence          FLOAT,          -- tahmin güven skoru
    signals_used        JSONB,          -- hangi sinyaller bu kararı verdi
    -- Örnek: {"momentum": 0.91, "category_heat": 0.73, "rank": 12,
    --         "fabric": "pamuk", "color": "siyah", "cluster": "star"}
    model_version       VARCHAR(30),    -- hangi model versiyonu (v1.2.0)

    -- Sen ne yaptın?
    decision        VARCHAR(20) NOT NULL,  -- 'produce' / 'skip' / 'wait' / 'order'
    quantity        INTEGER,               -- kaç adet ürettin/sipariş verdin
    unit_cost       NUMERIC,               -- birim maliyet
    total_cost      NUMERIC,               -- toplam maliyet
    planned_price   NUMERIC,               -- satış fiyatı planı

    notes           TEXT                   -- "pamuk bitti, viskon yaptım" gibi
);
```

**Tablo 2: `actual_sales`**
```sql
CREATE TABLE actual_sales (
    id              SERIAL PRIMARY KEY,
    production_id   INTEGER REFERENCES production_decisions(id),
    product_id      INTEGER REFERENCES products(id),
    search_term     VARCHAR(100),
    period_start    DATE,               -- satış dönemi başı
    period_end      DATE,               -- satış dönemi sonu
    days_in_market  INTEGER,            -- kaç gün vitrine koyulduydu

    -- Gerçek sonuç
    produced_quantity  INTEGER,         -- kaç adet üretildi
    sold_quantity      INTEGER,         -- kaç adet satıldı
    unsold_quantity    INTEGER,         -- kaç adet kaldı
    sell_through_rate  FLOAT,           -- sold / produced (0-1)
    avg_sell_price     NUMERIC,         -- ortalama satış fiyatı
    total_revenue      NUMERIC,         -- toplam gelir
    profit_margin      FLOAT,           -- net kar marjı

    -- Satış hızı
    first_sale_days    INTEGER,         -- ilk satış kaçıncı günde oldu?
    days_to_50pct      INTEGER,         -- %50'si kaç günde sattı?
    days_to_sellout    INTEGER,         -- tamamen bitti mi? kaç günde?

    -- Sistem değerlendirmesi
    was_good_decision   BOOLEAN,        -- genel olarak iyi karar mıydı?
    prediction_accuracy FLOAT,          -- |predicted-actual|/predicted
    feedback_date       DATE DEFAULT CURRENT_DATE,
    notes               TEXT
);
```

**Tablo 3: `model_versions`**
```sql
CREATE TABLE model_versions (
    id          SERIAL PRIMARY KEY,
    model_name  VARCHAR(50),        -- 'catboost_crop', 'kalman_tayt'
    version     VARCHAR(30),        -- 'v1.0.0', 'v1.1.0'
    trained_at  TIMESTAMPTZ,
    trained_on  INTEGER,            -- kaç kayıt üzerinde eğitildi?
    feedback_count INTEGER,         -- kaç feedback noktası vardı?
    metrics     JSONB,              -- {"mae": 12.3, "rmse": 18.7, "r2": 0.85}
    file_path   VARCHAR(255),       -- model dosyasının yolu
    is_active   BOOLEAN DEFAULT FALSE,
    notes       TEXT
);
```

### 4.4 Algoritmaların Feedback'ten Öğrenmesi

**Kalman Filter (Anlık, retraining gerekmez):**
```python
# Sen feedback veriyorsun
feedback = {
    "product_id": 2053,
    "category": "crop",
    "predicted": 80,   # sistem bunu tahmin etti
    "actual": 30       # gerçekte bu sattı
}

# Kalman anında güncelleniyor
innovation = feedback["actual"] - filter.state["level"]
kalman_gain = filter.P / (filter.P + filter.R)
filter.state["level"] += kalman_gain * innovation
filter.P = (1 - kalman_gain) * filter.P

# Büyük hata → R artıyor → gelecekte gerçek veriye daha çok güven
if abs(innovation) > 50:
    filter.R *= 1.2  # daha temkinli ol
```

**CatBoost (Haftalık retraining):**
```python
# Yeni eğitim verisi: feedback'ler
new_rows = pd.read_sql("""
    SELECT
        pd.product_id,
        p.dominant_color, p.fabric_type, p.fit_type,
        dm.momentum_score, dm.category_heat, dm.absolute_rank,
        dm.favorite_count, dm.avg_rating,
        as_.sell_through_rate AS target  -- Öğrenilecek şey bu!
    FROM production_decisions pd
    JOIN actual_sales as_ ON as_.production_id = pd.id
    JOIN products p ON p.id = pd.product_id
    JOIN daily_metrics dm ON dm.product_id = pd.product_id
    WHERE as_.feedback_date > NOW() - INTERVAL '30 days'
""", conn)

# Mevcut modele yeni verileri ekle ve yeniden eğit
model.fit(new_rows[features], new_rows["target"], ...)
```

### 4.5 Öğrenme Eğrisi

```
Başlangıç (0 feedback):
  Evrensel Trendyol sinyali → %55-60 isabetlilik tahmini
  "Rank yükseliyor = satacak" (herkesin bildiği şey)

10-20 feedback sonrası:
  Kalman hata marjını öğrendi
  "Crop'ta hata marjım ±30, tayta'ta ±15"
  → Daha gerçekçi güven aralıkları

30-50 feedback sonrası:
  CatBoost ilk kez yeniden eğitiliyor
  "Pamuk > polyester, siyah > beyaz senin müşterinde"
  → %70-75 isabetlilik

100+ feedback sonrası:
  "Ramazan'dan önce abiye öner, yazın tayt öner"
  Fiyat bandı öğrenildi: "500-800₺ arası en iyi dönüşüm"
  → %80-85 isabetlilik

1 Yıl sonrası:
  Tam döngüsel öğrenme. Mevsim + Müşteri profili + Fiyat bandı
  → %85-90 isabetlilik
```

### 4.6 Kullanıcı Arayüzü Senaryosu

```
=== LUMORA INTELLIGENCE ===

Bu haftanın önerileri (crop kategorisi):

  #1  Siyah Pamuk Crop Bluz         Skor: 0.94  ↑↑ Güçlü yükseliş
      Marka: ETHIQUET | Fiyat: 256₺ | Rank: 8 → (3 gün önce 340)
      Sinyaller: rank+332/3gün, pamuk trend, top kluster
      → [ÜRETMEYİ PLANLA] [Atla] [Bekle]

  #2  Beyaz Viskon Tayt              Skor: 0.81  ↑ Yukseliş
      Marka: Koton | Fiyat: 349₺ | Rank: 23 → (3 gün önce 89)
      ⚠️  Uyarı: Geçen ay benzer ürün satmadı (sell_through: %30)
      → [ÜRETMEYİ PLANLA] [Atla] [Bekle]

Geçen haftanın kararları — Gerçek sonuçları girin:
  Siyah Crop #A12 → 40 ürettim → Kaç sattı? [___] / Kaçı kaldı? [___]
  Polyester Tayt → Atladım (sistem öğrensin mi?) [Evet] [Hayır]
```

---

## 5. 🏗️ Sistem Mimarisi

### 5.1 Genel Akış

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
        │   │   → Style trends güncelle
        │   │   → products.trend_score/direction güncelle
        │   └── Her Pazar 03:00
        │       → CatBoost yeniden eğit
        │       → model_versions'a kaydet
        │
        ├── Manuel (API)
        │   ├── POST /trigger {scope, target, priority}
        │   ├── POST /analyze {product_id}
        │   ├── GET  /predict {category, top_n}
        │   ├── GET  /alerts
        │   ├── GET  /style-trends {category}
        │   ├── POST /feedback {production_id, sold, unsold}
        │   └── GET  /health
        │
        └── Queue Worker (arka plan)
            ├── Toplu hesaplamalar (3000+ ürün)
            ├── Model eğitimi
            └── CLIP embedding (ileride)
```

### 5.2 Akıllı Kuyruk Kararı

```python
DIRECT_TASKS = {
    "single_product_score",   # Tek ürün → <500ms
    "quick_predict",          # Önceden hesaplanmış okuma → <50ms
    "kalman_update",          # Anlık Kalman → <100ms
    "health_check",           # Sistem durumu → <10ms
}

QUEUE_TASKS = {
    "full_category_score",    # 3000+ ürün → 2-5 dk
    "catboost_train",         # Model eğitimi → 10-30 dk
    "style_trends_compute",   # JSONB analizi → 1-2 dk
    "clip_embed_all",         # Tüm görseller → saatler
    "bulk_rank_compute",      # Tüm rank_change → 1-2 dk
}
```

### 5.3 Dosya Yapısı (İleride)

```
lumora_intelligence/
├── main.py                     # FastAPI app başlatıcı + Scheduler
├── config.py                   # Merkezi ayarlar
├── requirements.txt
├── .env
│
├── api/
│   ├── routes/
│   │   ├── analyze.py          # POST /analyze
│   │   ├── predict.py          # GET  /predict
│   │   ├── trigger.py          # POST /trigger
│   │   ├── feedback.py         # POST /feedback ← ÖNEMLİ
│   │   ├── alerts.py           # GET  /alerts
│   │   └── health.py           # GET  /health
│   └── models/
│       ├── requests.py
│       └── responses.py
│
├── engine/
│   ├── pipeline.py             # 6 katmanı çalıştıran orchestrator
│   ├── features.py             # Feature engineering
│   ├── predictor.py            # Ensemble tahmin (CatBoost × 0.6 + Kalman × 0.4)
│   └── decision.py             # Üretim önerisine dönüştür
│
├── algorithms/
│   ├── zscore.py               ← MEVCUT
│   ├── catboost_model.py       ← MEVCUT
│   ├── kalman.py               ← MEVCUT
│   ├── prophet_model.py        ← MEVCUT
│   ├── changepoint.py          ← MEVCUT
│   ├── clustering.py           ← MEVCUT
│   ├── clip_model.py           ← MEVCUT (AÇMA henüz)
│   └── optimizer.py            ← MEVCUT
│
├── db/
│   ├── connection.py           # DB bağlantısı (SSH tunnel veya direkt)
│   ├── reader.py               # daily_metrics + products okuma
│   └── writer.py               # intelligence_results + alerts yazma
│
├── scheduler/
│   ├── jobs.py                 # Zamanlanmış görev tanımları
│   └── triggers.py             # Event-driven tetikleyiciler
│
├── queue/
│   ├── manager.py              # Kuyruk yöneticisi (asyncio.Queue)
│   ├── worker.py               # Arka plan işçisi
│   └── decision.py             # "Direkt mi? Kuyruk mu?"
│
└── models/                     # Eğitilmiş model dosyaları
    ├── catboost_crop_v1.cbm
    └── catboost_tayt_v1.cbm
```

---

## 6. 🗄️ Veritabanı Değişiklikleri Özeti

### 6.1 Yeni Tablolar

| Tablo | Amaç | Öncelik |
|-------|------|---------|
| `rank_snapshots` (veya daily_metrics kolonu) | Rank değişimi ve momentum | 🔴 İlk |
| `category_daily_signals` | Kategori nabzı | 🔴 İlk |
| `trend_alerts` | Uyarı sistemi | 🟡 İkinci |
| `style_trends` | Stil analizi | 🟡 İkinci |
| `production_decisions` | Üretim kararları (feedback) | 🔴 İlk |
| `actual_sales` | Gerçek satış sonuçları (feedback) | 🔴 İlk |
| `model_versions` | Model yönetimi | 🟢 Üçüncü |
| `intelligence_results` | Hesaplanan skorlar | 🟢 Üçüncü |

### 6.2 Mevcut Tablolara Eklentiler

```sql
-- daily_metrics
ALTER TABLE daily_metrics
  ADD COLUMN rank_change_1d  INTEGER,
  ADD COLUMN rank_change_3d  INTEGER,
  ADD COLUMN rank_velocity   FLOAT,
  ADD COLUMN momentum_score  FLOAT,
  ADD COLUMN is_new_entrant  BOOLEAN DEFAULT FALSE;

-- products
ALTER TABLE products
  ADD COLUMN dominant_color  VARCHAR(50),
  ADD COLUMN fabric_type     VARCHAR(50),
  ADD COLUMN fit_type        VARCHAR(50),
  ADD COLUMN collar_type     VARCHAR(50),
  ADD COLUMN trend_score     FLOAT,
  ADD COLUMN trend_direction VARCHAR(20),
  ADD COLUMN last_scored_at  TIMESTAMPTZ;
```

### 6.3 Önerilen İndeksler

```sql
-- Hız için kritik indeksler
CREATE INDEX idx_dm_product_date
  ON daily_metrics(product_id, DATE(recorded_at));

CREATE INDEX idx_dm_search_date
  ON daily_metrics(search_term, DATE(recorded_at));

CREATE INDEX idx_dm_momentum
  ON daily_metrics(momentum_score DESC)
  WHERE momentum_score IS NOT NULL;

CREATE INDEX idx_products_trend
  ON products(trend_score DESC)
  WHERE trend_score IS NOT NULL;

CREATE INDEX idx_alerts_unread
  ON trend_alerts(detected_at DESC)
  WHERE is_read = FALSE;
```

---

## 7. 📅 İmplementasyon Yol Haritası

### Faz 1 — Temel Sinyaller (Veri hazır, hemen başlanabilir)

- [ ] `daily_metrics`'e rank_change kolonlarını ekle
- [ ] Rank momentum hesaplama script'i yaz (nightly batch)
- [ ] `category_daily_signals` tablosunu oluştur
- [ ] Kategori heat skoru hesaplama script'i yaz
- [ ] `products` tablosuna JSONB'den `dominant_color`, `fabric_type` extract et
- [ ] `trend_alerts` tablosunu oluştur
- [ ] Rank spike alert mantığını yaz (rank_change_3d > 300 → alert)

**Beklenti:** Bu çalışınca bile çok değerli dashboard veri var.

### Faz 2 — Feedback Altyapısı (Sistem öğrenmeye başlar)

- [ ] `production_decisions` tablosunu oluştur
- [ ] `actual_sales` tablosunu oluştur
- [ ] Feedback giriş arayüzü (CLI veya basit web formu)
- [ ] Kalman'ı feedback'e bağla (her feedback → anlık güncelleme)
- [ ] `model_versions` tablosunu oluştur

**Beklenti:** İlk 10 feedback sonrası Kalman daha isabetli çalışır.

### Faz 3 — Stil Trendleri

- [ ] `style_trends` tablosunu oluştur
- [ ] JSONB attribute analizi script'i (haftalık batch)
- [ ] "Bu hafta hangi renk/kumaş trend?" raporu üret

### Faz 4 — CatBoost Entegrasyonu (30+ gün + 30+ feedback sonrası)

- [ ] `engine/features.py`'yi gerçek DB verisiyle bağla
- [ ] CatBoost ilk eğitimini gerçek verilerle yap
- [ ] Haftalık otomatik yeniden eğitim (APScheduler)
- [ ] SHAP açıklamaları: "neden bu ürünü önerdi?"

### Faz 5 — API & Otomasyon

- [ ] FastAPI skeleton
- [ ] `/analyze`, `/predict`, `/feedback`, `/alerts` endpoint'leri
- [ ] APScheduler gece 02:00 otomatik çalıştırma
- [ ] LangChain entegrasyonu (HTTP → Lumora Intelligence)
- [ ] Sunucuya deploy + systemd service

### Faz 6 — İleri Seviye (İsteğe bağlı)

- [ ] Prophet mevsimsellik analizi (60+ gün sonrası)
- [ ] K-Prototypes ürün segmentasyonu
- [ ] CLIP görsel analiz (GPU gerektirir)
- [ ] Bayesian parametre optimizasyonu
- [ ] Web dashboard (React/Next.js?)

---

## 8. ⚠️ Kısıtlar & Riskler

| Risk | Açıklama | Çözüm |
|------|----------|-------|
| Veri yetersizliği | 7 günlük veri — CatBoost için yetersiz | Faz 1-2'yi 30 gün veri birikince başlat |
| Sıfır metrik | cart/view %87-91 sıfır | Rank + favori sinyallerine odaklan |
| Scraper boşlukları | Tayt 51 saat çalışmadı | Scraper kararlılığını artır |
| Zombie loglar | 15 işlem "running" gösteriyor | Scraper'a watchdog ekle |
| Ground truth eksikliği | Satış verisi yok | production_decisions + actual_sales tabloları |
| Sistem kararı belirsiz | "Bu yapılacak mı?" henüz netleşmedi | Plan hazır, karar bekliyor |

---

## 9. 🔀 Dinamik Kategori Sistemi — Sınırsız Kategori Desteği

### 9.1 Temel Prensip: Hiçbir Yere Kategori İsmi Yazılmaz

```python
# YANLIŞ — sabit kodlu sistem ❌
if category == "crop":
    model = catboost_crop
elif category == "tayt":
    model = catboost_tayt
# Yeni kategori gelince KOD DEĞİŞTİRMEK gerekiyor

# DOĞRU — dinamik sistem ✅
model = ModelRegistry.get_or_create(category)
# Kategori yoksa → otomatik oluşturur, sıfırdan başlatır
```

**Bu tasarımla:** crop, tayt, grup bugün çalışıyor.
Yarın "elbise" gelirse → kod değişikliği yok, otomatik adapte olur.
6 ay sonra "çanta", "ayakkabı", "aksesuar" → hepsi otomatik.

---

### 9.2 Kategori Yaşam Döngüsü

```
Scraper yeni bir kategori tarar (örn. "elbise")
          │
          ▼
  Sistem gece kontrolünde algılar
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  COLD (0-30 gün, <500 kayıt)                        │
│  ─────────────────────────────────────────────────  │
│  Aktif: Rank momentum, Favori sinyali               │
│  Pasif: CatBoost (yok), Prophet (yok)               │
│  Kalman: Başlatıldı (global default değerlerle)     │
│  Z-Score eşiği: Geçici (global ortalamadan)         │
│  Güven: Düşük — "Veri birikiliyor"                  │
└─────────────────────────────────────────────────────┘
          │ 30 gün + 500 kayıt geçince otomatik geçiş
          ▼
┌─────────────────────────────────────────────────────┐
│  WARMING (30-90 gün)                                │
│  ─────────────────────────────────────────────────  │
│  Aktif: Kalman kendi datasına göre ayarlandı        │
│  Aktif: CatBoost ilk eğitimini yaptı (temel model) │
│  Z-Score: Bu kategorinin kendi istatistiklerinden  │
│  Güven: Orta — "Model öğreniyor"                   │
└─────────────────────────────────────────────────────┘
          │ 90 gün + 30 feedback geçince
          ▼
┌─────────────────────────────────────────────────────┐
│  HOT (90-365 gün)                                   │
│  ─────────────────────────────────────────────────  │
│  Tam ensemble: CatBoost × 0.6 + Kalman × 0.4      │
│  Gerçek satış feedback'iyle öğreniyor               │
│  Haftalık CatBoost yeniden eğitimi                 │
│  Güven: Yüksek — "Senin müşterini öğrendi"         │
└─────────────────────────────────────────────────────┘
          │ 365 gün + 200 feedback
          ▼
┌─────────────────────────────────────────────────────┐
│  MATURE (1 yıl+)                                    │
│  ─────────────────────────────────────────────────  │
│  Prophet mevsimsellik aktif                         │
│  "Geçen yıl Ramazan'da ne oldu?" hafızası          │
│  CPD ile rejim değişikliği tespiti                 │
│  Güven: Çok yüksek — "%85-90 isabetlilik"          │
└─────────────────────────────────────────────────────┘
```

---

### 9.3 Her Modelin Davranışı: Ayrı mı Paylaşımlı mı?

| Bileşen | Yapı | Açıklama |
|---------|------|----------|
| **CatBoost** | 🔴 Her kategori ayrı model | Fiyat/kumaş dinamikleri çok farklı |
| **Kalman Filter** | 🔴 Her kategori ayrı state | Crop ısınırken tayt soğuyabilir |
| **Prophet** | 🔴 Her kategori ayrı model | Mevsimsellik tamamen farklı |
| **Z-Score eşikleri** | 🔴 Her kategori ayrı hesaplanır | Grupta 100k sepet normal, cropte anormal |
| **CLIP modeli** | 🟢 Tek paylaşımlı model | Ama embedding'ler kategori içinde sorgulanır |
| **Feature pipeline** | 🟢 Aynı şablon, farklı parametre | Yapı aynı, ağırlıklar değişiyor |
| **K-Prototypes** | 🟢 Paylaşılabilir | Cluster tanımları evrensel |

---

### 9.4 Kategori Kayıt Defteri (Ana Tablo — `categories`)

```sql
CREATE TABLE categories (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) UNIQUE NOT NULL,  -- search_term (crop, tayt, elbise...)
    display_name    VARCHAR(200),                  -- "Kadın Elbise" (görünen ad)

    -- Yaşam döngüsü
    first_seen          DATE,
    status              VARCHAR(20) DEFAULT 'cold', -- cold/warming/hot/mature
    data_days           INTEGER DEFAULT 0,          -- kaç günlük veri var
    record_count        INTEGER DEFAULT 0,          -- toplam kayıt sayısı
    feedback_count      INTEGER DEFAULT 0,          -- kaç gerçek satış feedback'i alındı

    -- Model durumu
    has_kalman          BOOLEAN DEFAULT FALSE,
    has_catboost        BOOLEAN DEFAULT FALSE,
    has_prophet         BOOLEAN DEFAULT FALSE,
    catboost_version    VARCHAR(30),                -- 'v1.2.0' veya NULL
    kalman_state        JSONB,
    -- Örnek: {"level": 0.65, "velocity": 0.02, "P": 1.0, "R": 0.5}

    -- Otomatik hesaplanan kategori karakteristikleri
    avg_price               NUMERIC,
    price_std               NUMERIC,
    avg_rank_range          INTEGER,       -- Kategoride tipik ürün sayısı
    typical_rank_volatility FLOAT,         -- Rank ne kadar hızlı değişiyor
    zscore_price_threshold  FLOAT DEFAULT 3.0,
    zscore_rank_threshold   FLOAT DEFAULT 2.5,

    -- Kategorinin otomatik sınıflandırması
    category_type   VARCHAR(50),
    -- 'mid_fashion' | 'premium' | 'budget_fashion' | 'heterogeneous' | 'accessories'
    is_homogeneous  BOOLEAN,    -- tekstil gibi homojen mi, grup gibi karışık mı?
    price_segment   VARCHAR(20), -- 'budget' | 'mid' | 'premium' | 'luxury'

    -- Kontrol
    is_active           BOOLEAN DEFAULT TRUE,
    last_analyzed       TIMESTAMPTZ,
    auto_provisioned    BOOLEAN DEFAULT TRUE,  -- sistem mi oluşturdu, sen mi?
    notes               TEXT
);
```

---

### 9.5 Otomatik Provizyon Motoru

```python
class CategoryProvisioner:
    """
    Yeni kategori scraper'dan geldiğinde otomatik devreye girer.
    Hiçbir müdahale gerekmez.
    """

    def provision(self, category_name: str):
        # 1. İstatistiklerini hesapla
        stats = self.compute_stats(category_name)
        # avg_price, price_std, rank_volatility vb.

        # 2. Tipini otomatik belirle
        category_type = self._classify(stats)

        # 3. Bu kategoriye özel Z-Score eşiği hesapla
        zscore = self._compute_thresholds(stats)

        # 4. Kalman sıfırdan başlat (belirsiz → P yüksek)
        kalman_state = {
            "level": stats["avg_momentum_score"],
            "velocity": 0.0,
            "P": 1.0   # yüksek belirsizlik, cold start
        }

        # 5. DB'ye yaz
        self.db.insert_category({
            "name": category_name,
            "status": "cold",
            "kalman_state": kalman_state,
            "category_type": category_type,
            **stats, **zscore
        })

    def _classify(self, stats) -> str:
        price_cv = stats["price_std"] / stats["avg_price"]

        if price_cv > 2.0:       return "heterogeneous"   # grup gibi
        elif stats["avg_price"] > 3000: return "premium"
        elif stats["avg_price"] < 300:  return "budget_fashion"
        else:                    return "mid_fashion"
```

---

### 9.6 Kategori Tipine Göre Sistem Davranışı

| Tip | Örnek | Z-Score | Kalman | CatBoost Feature'ları |
|-----|-------|---------|--------|-----------------------|
| `mid_fashion` | crop, tayt | 3.0 | Orta | fabric, color, fit |
| `premium` | kadın abiye, "abiye" | 2.0 (hassas) | Düşük (temkinli) | brand, fabric, occasion |
| `budget_fashion` | alt segment | 3.5 (geniş) | Yüksek (reaktif) | price_rank, discount |
| `heterogeneous` | grup | 5.0 (çok geniş) | Düşük | brand_tier, subcategory |
| `accessories` | çanta, ayakkabı | 2.5 | Orta | material, brand, season |

---

### 9.7 Category-Agnostic Pipeline

```python
class LumoraEngine:
    """
    'crop' için de çalışır, 'elbise' için de,
    henüz görmediğimiz bir kategori için de.
    """

    def analyze(self, category: str, product_ids: list = None):
        # 1. Registry'den al (yoksa otomatik oluştur)
        cat = self.registry.get_or_create(category)

        # 2. Status'e göre pipeline seç
        pipeline = self.build_pipeline(cat.status)

        # 3. Çalıştır
        return pipeline.run(category=category, ...)

    def build_pipeline(self, status: str) -> Pipeline:
        # Her zaman çalışır
        steps = [ZScoreCleaner, RankMomentum]

        if status in ["warming", "hot", "mature"]:
            steps.append(KalmanFilter)

        if status in ["hot", "mature"]:
            steps.append(CatBoostPredictor)
            steps.append(EnsembleLayer)

        if status == "mature":
            steps.append(ProphetDecomposer)

        return Pipeline(steps)
```

---

### 9.8 Model Dosya Yapısı (Dinamik)

```
models/
├── crop/
│   ├── catboost_v1.0.cbm
│   ├── catboost_v1.1.cbm
│   └── catboost_v1.2.cbm  ← is_active
├── tayt/
│   └── catboost_v1.0.cbm
├── grup/
│   └── catboost_v0.8.cbm  ← az veri, düşük versiyon
├── kadin_abiye/
│   └── catboost_v0.5.cbm  ← warm start (croptan başladı)
│
├── elbise/                 ← Yarın gelirse otomatik oluşur
│   └── (cold mode, model yok henüz)
│
└── canta/                  ← 6 ay sonra gelirse
    └── (cold mode)
```

---

### 9.9 Status Geçişi — Otomatik SQL

```sql
-- Her gece çalışır, statusleri günceller
UPDATE categories SET
    data_days    = (
        SELECT COUNT(DISTINCT DATE(recorded_at))
        FROM daily_metrics
        WHERE search_term = categories.name
    ),
    record_count = (
        SELECT COUNT(*)
        FROM daily_metrics
        WHERE search_term = categories.name
    ),
    feedback_count = (
        SELECT COUNT(*)
        FROM actual_sales as_
        JOIN production_decisions pd ON pd.id = as_.production_id
        WHERE pd.search_term = categories.name
    ),
    status = CASE
        WHEN data_days >= 365 AND feedback_count >= 200 THEN 'mature'
        WHEN data_days >= 90  AND feedback_count >= 30  THEN 'hot'
        WHEN data_days >= 30  AND record_count   >= 500 THEN 'warming'
        ELSE 'cold'
    END,
    last_analyzed = NOW();
```

---

### 9.10 Warm Start — Az Veri Olan Kategori için

```python
# Kadın abiye sadece 169 kayıt → CatBoost için yetersiz
# Çözüm: En benzer kategoriden başla

def get_initial_model(category: str, stats: dict):
    if stats["record_count"] >= 500:
        # Yeterli veri, sıfırdan eğit
        return CatBoost().fit(category_data)
    else:
        # En benzer kategorinin modelinden başla
        similar = find_similar_category(stats)
        # "kadin_abiye" → en benzer "crop" (ikisi de tekstil, mid-premium)
        base_model = load_model(similar)
        # Üstüne bu kategorinin az verisiyle fine-tune et
        return base_model.fit(category_data, init_model=base_model)
```

---

### 9.11 Sonuç: Scale Edilebilirlik

```
Bugün:     crop, tayt, grup, kadın abiye (4 kategori)
           → Sistem otomatik yönetiyor

6 ay:      + elbise, pantolon, çanta, ayakkabı, aksesuar (9 kategori)
           → SIFIR KOD DEĞİŞİKLİĞİ
           → "elbise cold mode, sadece rank sinyali"
           → "pantolon warming moda geçti, catboost eğitiliyor"

1 yıl:     crop mature → Ramazan hafızası var
           elbise hot  → feedback'ten öğreniyor
           çanta cold  → veri birikmeye devam

Sınır:     Teorik olarak yok. DB kapasitesi kadar kategori.
```

---

*Rapor: 2026-03-05 | Lumora Intelligence v1.1 Tasarım Dokümanı*
*Güncelleme: Dinamik kategori sistemi eklendi*
