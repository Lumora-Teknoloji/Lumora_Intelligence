# Lumora Intelligence — Trendyol Trend Tahmin Motoru

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/CatBoost-Ensemble-brightgreen" />
  <img src="https://img.shields.io/badge/Kalman-Online%20Learning-orange" />
  <img src="https://img.shields.io/badge/Trendyol-Native-red" />
  <img src="https://img.shields.io/badge/Simulation-44%20kolonlu-purple" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## 🎯 Ne Yapar?

Lumora Intelligence, Trendyol platformundaki ürünlerin trend durumunu tahmin eden, gerçek satış geri bildirimiyle öğrenen bir yapay zeka motorudur.

```
Bu hafta hangi 5 ürünü üretmeliyim?  →  TREND listesi
Bu ürün 2 sattı, bu 45 sattı        →  feedback_top_n()
Bir hafta sonra                      →  Sahte trendler listeden düştü
```

---

## 🏗️ Mimari (6 Katman)

```
┌─────────────────────────────────────────────────────┐
│  Katman 1  │  Z-Score Dedektörü (veri temizleme)    │
│  Katman 2  │  Feature Engineering (40+ özellik)     │
│  Katman 3  │  CatBoost (composite demand skoru)     │
│  Katman 4  │  Kalman Filtresi (online güncelleme)   │
│  Katman 5  │  Ensemble Layer (ağırlıklı birleştirme)│
│  Katman 6  │  Feedback Loop (sahte trend öğrenimi)  │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Hızlı Başlangıç

```bash
git clone https://github.com/Lumora-Teknoloji/Lumora_Intelligence.git
cd Lumora_Intelligence
pip install -r requirements.txt

# 5 senaryo testi çalıştır
python run_scenarios.py

# Tek kategori trend analizi
python show_trend_abiye.py
```

---

## 🔑 Temel API

```python
from engine.predictor import PredictionEngine

# 1. Motor eğit
engine = PredictionEngine()
engine.train(df_metrics)

# 2. Bu haftanın trend listesini al
predictions = engine.predict()
top5 = predictions.head(5)
print(top5[["product_id", "trend_label", "trend_score", "confidence"]])

# 3. Gerçek satış ile feedback ver (sahte trendleri öğret)
engine.feedback_top_n({
    pid_A: 45,   # iyi sattı → sistem güvenir
    pid_B: 2,    # az sattı → SAHTE TREND → ceza ×0.35
    pid_C: 30,
})

# 4. Toplu feedback (etiket bazlı)
engine.feedback_batch({
    "TREND":      450,   # Bu hafta TREND ürünler toplam 450 sattı
    "POTANSIYEL": 80,
})
```

---

## 📊 Test Sonuçları (v2)

| Senaryo | Yükseliş Tespiti | Skor Gap |
|---------|-----------------|----------|
| Karma Kategoriler (Baseline) | ✅ %60.0 | +0.4 |
| Sezonsal Pik (Kış Başlangıcı) | ⚠️ %52.3 | +4.8 |
| Soğuk Başlangıç (Yeni Kategori) | ✅ %71.0 | — |
| Fiyat/Enflasyon Baskısı | ⚠️ %56.2 | +1.5 |
| Rakip Çatışması + Viral Ürün | ✅ %65.4 | — |

> ✅ Baseline %60 eşiğini ilk kez v2'de geçti — growth-rate priority scoring ve stok sinyalleri ile.
> ⚠️ Sonuçlar simüle veri üzerindendir. Gerçek Trendyol verisi ile %70+ doğruluk beklenmektedir.

---

## 📁 Proje Yapısı

```
Lumora_Intelligence/
├── engine/
│   ├── predictor.py        # Ana motor (train + predict + feedback)
│   └── features.py         # Feature mühendisliği (40+ özellik)
├── algorithms/
│   ├── catboost_model.py   # CatBoost ensemble
│   ├── kalman_filter.py    # Online öğrenme
│   └── zscore.py           # Anomali tespiti
├── data/
│   ├── sample_data.py      # Gerçekçi simülasyon verisi (v3, 44 kolon)
│   ├── scenarios.py        # 5 test senaryosu
│   └── yearly_simulation.py
├── run_scenarios.py        # 5 senaryo karşılaştırma testi
├── run_yearly_persistent.py# 1 yıllık kalıcı öğrenme testi
├── show_trend_abiye.py     # Kategori bazlı trend analizi demo
├── demo_gercek_kullanim.py # Top-5 + feedback demo
└── requirements.txt
```

---

## 📦 Simülasyon Verisi (v3 — 44 Kolon)

`data/sample_data.py` gerçek Trendyol veritabanı yapısını yüksek doğrulukla simüle eder:

| Özellik | Detay |
|---------|-------|
| **Kategoriler** | crop, tayt, kadın abiye, mont, kazak, elbise, spor giyim, grup |
| **JSONB Attributes** | Her kategori için 10-14 spesifik key (bel yüksekliği, etek boyu, teknoloji...) |
| **Beden/Stok** | `available_sizes`, `total_stock`, `stock_depth` beden bazlı stok takibi |
| **Özel Gün Takvimi** | Sevgililer, Anneler Günü, Black Friday (11.11), Yılbaşı boost |
| **Sezon Faktörü** | Off-season ürün %2-10'a düşer (mont yazın, crop kışın) |
| **Pareto Dağılımı** | Gerçek uzun kuyruk favori dağılımı (%20 ürün %80 favori) |
| **Gerçek Doluluk** | DB'den alınan oranlar: cart %3-27, view %5-58, fav %55-93 |

---

## 🔧 Trend Etiketleri

| Etiket | Anlam |
|--------|-------|
| `TREND` | Yükselen, üretilmeli |
| `POTANSIYEL` | Takip edilmeli |
| `STABIL` | Sabit talep |
| `DUSEN` | Düşüşte, stok azaltılmalı |

---

## ⚙️ Scoring Mantığı (v2)

Growth-rate öncelikli — mutlak popülerlik değil, **değişim hızı** esas alınır:

```
trend_score = fav_growth  × 0.45   # erken büyüme sinyali (küçük ama hızlı ürün)
            + velocity    × 0.25   # Kalman anlık değişim
            + CatBoost    × 0.20   # tarihsel talep (azaltıldı: Pareto'yu bastırır)
            + cart_growth × 0.10   # sepet büyümesi
```

---

## 🗺️ Yol Haritası

- [x] CatBoost + Kalman ensemble motoru
- [x] Gerçek zamanlı feedback loop (`feedback_top_n`, `feedback_batch`)
- [x] Growth-rate öncelikli scoring (relative > absolute popularity)
- [x] 44 kolonlu gerçekçi simülasyon (JSONB + beden stok + özel günler)
- [x] 7 kategori, 5 senaryo test suite
- [ ] Instagram/TikTok sosyal medya sinyali
- [ ] OpenWeatherMap hava durumu entegrasyonu
- [ ] Trendyol Seller API otomatik entegrasyonu
- [ ] Prophet sezonsal analiz (180+ gün veri ile)
- [ ] Hepsiburada cross-platform konfirmasyon

---

## 📄 Lisans

MIT License — © 2025 Lumora Teknoloji
