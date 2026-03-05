# Lumora Intelligence — Trendyol Trend Tahmin Motoru

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/CatBoost-Ensemble-brightgreen" />
  <img src="https://img.shields.io/badge/Kalman-Online%20Learning-orange" />
  <img src="https://img.shields.io/badge/Trendyol-Native-red" />
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
│  Katman 6  │  Feedback Loop (sahte trend öğrenimi) │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Hızlı Başlangıç

```bash
git clone https://github.com/Lumora-Teknoloji/Lumora_Intelligence.git
cd Lumora_Intelligence
pip install -r requirements.txt
python run_test.py
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
    pid_C: 30,   # iyi sattı
})

# 4. Toplu feedback (etiket bazlı)
engine.feedback_batch({
    "TREND":      450,   # Bu hafta TREND ürünler toplam 450 sattı
    "POTANSIYEL": 80,
})
```

---

## 📊 Test Sonuçları

| Senaryo | Yükseliş Tespiti | Notlar |
|---------|-----------------|--------|
| Karma Kategoriler | ⚠️ %51.7 | Baseline |
| Sezonsal Pik | ⚠️ %59.6 | Kış→İlkbahar geçişi |
| Soğuk Başlangıç | ✅ %77.4 | Yeni kategoriye adaptasyon |
| Fiyat Baskısı | ⚠️ %44.4 | Enflasyon etkisi |
| Rakip + Viral | ✅ %70.8 | Viral ürün tespiti |

> ⚠️ Sonuçlar simüle veri üzerindendir. Gerçek Trendyol verisi ile kalibre edildikten sonra %65-75+ doğruluk beklenmektedir.

---

## 📁 Proje Yapısı

```
Lumora_Intelligence/
├── engine/
│   ├── predictor.py        # Ana motor (train + predict + feedback)
│   ├── features.py         # Feature mühendisliği (40+ özellik)
│   └── ...
├── algorithms/
│   ├── catboost_model.py   # CatBoost ensemble
│   ├── kalman_filter.py    # Online öğrenme
│   ├── zscore_detector.py  # Anomali tespiti
│   └── ...
├── data/
│   ├── sample_data.py      # Gerçekçi simülasyon verisi
│   └── yearly_simulation.py
├── run_test.py             # Temel test
├── run_scenarios.py        # 5 senaryo karşılaştırma
├── run_yearly_persistent.py # 1 yıllık kalıcı öğrenme testi
├── demo_gercek_kullanim.py # Top-5 + feedback demo
├── requirements.txt
└── README.md
```

---

## 🔧 Özellikler

### Trend Etiketleri
| Etiket | Anlam |
|--------|-------|
| `TREND` | Yükselen, üretilmeli |
| `POTANSIYEL` | Takip edilmeli |
| `STABIL` | Sabit talep |
| `DUSEN` | Düşüşte, stok azaltılmalı |

### Feedback Sistemi
- `feedback_top_n(dict)` — ürün bazlı satış girişi
- `feedback_by_label(label, total)` — etiket bazlı toplu giriş
- `feedback_batch(dict)` — birden fazla etiket aynı anda
- Sahte trend tespiti: `sold < 5 adet VEYA tahminin < %10`

### Öğrenilen Sinyaller
- `favorite_growth_3d`: 3 günlük erken büyüme (viral uyarısı)
- `favorite_growth_14d`: 2 haftalık standart büyüme
- `price_change_pct`: fiyat değişimi (enflasyon etkisi)
- `is_fav_spike`: %200+ ani artış (viral tespit)
- `rank_velocity_7d`: rank değişim hızı

---

## 📦 Gereksinimler

```
catboost>=1.2
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
pykalman>=0.9
```

---

## 🗺️ Yol Haritası

- [x] CatBoost + Kalman ensemble motoru
- [x] Gerçek zamanlı feedback loop
- [x] Growth-rate öncelikli scoring
- [ ] Instagram/TikTok sosyal medya sinyali
- [ ] OpenWeatherMap hava durumu entegrasyonu
- [ ] Trendyol Seller API otomatik entegrasyonu
- [ ] Prophet sezonsal analiz (180+ gün veri ile)
- [ ] Hepsiburada cross-platform konfirmasyon

---

## 📄 Lisans

MIT License — © 2025 Lumora Teknoloji
