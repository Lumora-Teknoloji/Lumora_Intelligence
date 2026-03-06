# Lumora Intelligence — Trendyol Trend Tahmin Motoru

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/CatBoost-Ensemble-brightgreen" />
  <img src="https://img.shields.io/badge/Kalman-Online%20Learning-orange" />
  <img src="https://img.shields.io/badge/Trendyol-Native-red" />
  <img src="https://img.shields.io/badge/Benchmark-K1--K4%20Validated-purple" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## 🎯 Ne Yapar?

Lumora Intelligence, Trendyol platformundaki ürünlerin trend durumunu tahmin eden, gerçek satış geri bildirimiyle öğrenen yapay zeka motorudur.

```
Bu hafta hangi ürünleri üretmeliyim?  →  TREND listesi
Bu ürün 2 sattı, bu 45 sattı          →  feedback_top_n()
Bir hafta sonra                        →  Sahte trendler listeden düştü
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

# CSV dataset üret (5 kademe × 4 periyot = 20 dataset)
python data/generate_dataset.py

# Tüm 20 dataseti test et
python batch_test.py

# Belirli kademeyi test et
python batch_test.py --kademeler 1 2 3
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

# 3. Gerçek satış ile feedback ver
engine.feedback_top_n({
    pid_A: 45,   # iyi sattı → sistem güvenir
    pid_B: 2,    # az sattı → SAHTE TREND → ceza ×0.35
})
```

---

## 📊 Benchmark Test Sonuçları (K1-K4)

> Tam detay: `results/benchmark_test_raporu.docx`

Algoritma, **5 gürültü seviyesi × 4 zaman periyodu = 20 farklı dataset** üzerinde test edilmiştir.
Her test: 120 kategori × 3 trend profil × 1 ürün = **360 ürün**.

### Nedensel Veri Zinciri

Dataset'lerdeki metrikler gerçekçi nedensel zincire göre üretilmektedir:

```
rank ↑ → görünürlük ↑ → view_count ↑ → favorite_count ↑
→ cart_count ↑ → satın alma → (7 gün gecikme) → rating_count artışı
→ sosyal kanıt → fav_count'a geri besleme
```

### Ortalama Performans

| Kademe | Gürültü | TREND Precision | Rising Recall | Not |
|--------|---------|----------------|--------------|-----|
| **K1 Kristal** | ±%8 | **%99.8** | **%99.8** | Teorik üst sınır |
| **K2 Net** | ±%18 | **%99.2** | **%96.7** | Güvenilir |
| **K3 Orta ★** | ±%28 | **%97.8** | **%88.6** | **Gerçek dünya** |
| **K4 Gürültülü** | ±%42 | **%87.0** | **%58.8** | Kırılma noktası |

> ★ K3 (±%28 gürültü) Trendyol üretim verilerine en yakın senaryodur.

### Temel Bulgular

- **TREND Precision** K3'te bile %97.8 — yanlış alarm oranı çok düşük ✅
- **Rising Recall** K3'te %88.6 — her 9 yükselen üründen 1'i kaçırılıyor ⚠️
- **Falling Recall** tüm kademelerde %40-58 — yapısal iyileştirme gerekli ❌
- **Algoritma kırılma noktası:** K4 (±%42) — TREND precision %87'ye iner

---

## 📁 Proje Yapısı

```
Lumora_Intelligence/
├── engine/
│   ├── predictor.py            # Ana motor (train + predict + feedback)
│   └── features.py             # Feature mühendisliği (40+ özellik, vectorized)
├── algorithms/
│   ├── catboost_model.py       # CatBoost ensemble
│   ├── kalman.py               # Online öğrenme
│   ├── zscore.py               # Anomali tespiti
│   ├── clustering.py           # Ürün kümeleme
│   └── changepoint.py          # Trend kırılma algısı
├── data/
│   ├── generate_dataset.py     # 20 benchmark dataset üretimi
│   └── datasets/               # k1/-k5/ × {2m,4m,6m,12m}/
├── results/
│   ├── benchmark_test_raporu.docx   # K1-K4 detay raporu
│   └── batch_test_results.csv
├── batch_test.py               # 20 dataset toplu test
├── validate_datasets.py        # Veri bütünlüğü doğrulama
├── check_correlations.py       # Nedensel zincir korelasyon analizi
├── run_csv_test.py             # Tek CSV test runner
├── config.py
├── main.py
└── requirements.txt
```

---

## 🔧 Trend Etiketleri

| Etiket | Anlam |
|--------|-------|
| `TREND` | Yükselen, üretilmeli |
| `POTANSIYEL` | Takip edilmeli |
| `STABIL` | Sabit talep |
| `DUSEN` | Düşüşte, stok azaltılmalı |

---

## ⚙️ Scoring Mantığı

Growth-rate öncelikli — mutlak popülerlik değil, **değişim hızı** esas alınır:

```
trend_score = fav_growth  × 0.45   # erken büyüme sinyali
            + velocity    × 0.25   # Kalman anlık değişim
            + CatBoost    × 0.20   # tarihsel talep
            + cart_growth × 0.10   # sepet büyümesi
```

---

## 🗺️ Yol Haritası

- [x] CatBoost + Kalman ensemble motoru
- [x] Gerçek zamanlı feedback loop
- [x] Growth-rate öncelikli scoring
- [x] 20 benchmark dataset (5 kademe × 4 periyot)
- [x] Nedensel veri zinciri (rank→view→fav→cart→rating)
- [x] K1-K4 benchmark test ve docx raporu
- [ ] Falling recall iyileştirmesi (düşen trend erken uyarı)
- [ ] K5 (Kaotik) kademesi testi
- [ ] Trendyol Seller API otomatik entegrasyonu
- [ ] Prophet sezonsal analiz (180+ gün veri)

---

## 📄 Lisans

MIT License — © 2025 Lumora Teknoloji
