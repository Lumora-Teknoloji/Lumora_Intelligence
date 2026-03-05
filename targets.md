# 🧠 Lumora Intelligence — Hedefler & Planlar

Proje adı: **Lumora Intelligence**
Dizin: `Analiz/` (ileride `lumora_intelligence/` olarak rename edilebilir)
Durum: Geliştirme aşamasında, henüz karara bağlanmamış

---

## 🎯 Ne Yapacak?

Trendyol scraper'ının topladığı günlük ürün verilerini yapay zeka ile analiz eden,
LangChain backend'den **bağımsız** çalışan bir AI sistemi.

- **Gerçek zamanlı** trend skoru hesaplama
- **Otomatik** zamanlanmış toplu hesaplama (gece)
- **Manuel** tetikleme (API veya doğrudan çalıştırma)
- **Akıllı kuyruk**: Hafif iş → direkt, ağır iş → arka planda

---

## 📐 Muhtemel Mimari

```
[Scraper] → [lumora_db] ← [Lumora Intelligence]
                               ├── FastAPI API (:8001)
                               ├── Scheduler (gece 02:00)
                               ├── Queue (ağır işler)
                               └── AI Engine (6 katman)
                                       ├── Z-Score (temizlik)
                                       ├── Feature Engineering
                                       ├── CatBoost (tahmin)
                                       ├── Kalman Filter (anlık)
                                       ├── Prophet (mevsimsellik)
                                       └── CLIP (görsel)

[LangChain Backend] → HTTP → [Lumora Intelligence] → sonuç
```

---

## ✅ Çekirdek Dosyalar (Korunacak)

### Engine (AI algoritmaları)
- `algorithms/catboost_model.py` — Ana ML modeli
- `algorithms/kalman.py` — Anlık adaptasyon
- `algorithms/zscore.py` — Anomali temizleme
- `algorithms/prophet_model.py` — Mevsimsellik
- `algorithms/changepoint.py` — Trend kırılma tespiti
- `algorithms/clustering.py` — Ürün segmentasyonu
- `algorithms/clip_model.py` — Görsel analiz (CLIP)
- `algorithms/optimizer.py` — Bayesian parametre optimizasyonu
- `engine/features.py` — Feature engineering pipeline
- `engine/predictor.py` — Ensemble tahmin motoru

### Uygulama
- `main.py` — Giriş noktası
- `config.py` — Merkezi ayarlar
- `requirements.txt` — Bağımlılıklar
- `data/sample_data.py` — Mock veri üreteci

### Dokümantasyon
- `targets.md` — Bu dosya (hedefler)
- `README.md` — Proje açıklaması
- `analiz_algoritmalari.md` — Algoritma notları

---

## 📋 Yapılacaklar (Karar verince)

### Aşama 1 — Altyapı
- [ ] FastAPI skeleton ekle (`api/` klasörü)
- [ ] `/health`, `/predict`, `/analyze`, `/trigger` endpoint'leri
- [ ] SSH Tunnel yerine direkt DB bağlantısı (sunucuya deploy edilince)
- [ ] `intelligence_results` tablosunu DB'de oluştur
- [ ] `ai_jobs` tablosunu DB'de oluştur (kuyruk yönetimi)

### Aşama 2 — Engine
- [ ] `db/reader.py`: daily_metrics + products okuma
- [ ] `db/writer.py`: intelligence_results yazma
- [ ] Feature pipeline gerçek veriye bağla (şu an mock data çalışıyor)
- [ ] CatBoost'u gerçek 20k ürün verisiyle eğit
- [ ] Kalman'ı her kategori için ayrı state ile başlat

### Aşama 3 — Otomasyon
- [ ] APScheduler entegrasyonu
- [ ] Gece 02:00 toplu skor hesaplama görevi
- [ ] `engagement_score` DB kolonunu doldur
- [ ] Yeni scraping sonrası Kalman otomatik güncelle

### Aşama 4 — Entegrasyon
- [ ] LangChain Backend → Lumora Intelligence HTTP çağrısı
- [ ] Sonuçları LangChain kullanıcıya iletsin
- [ ] Dockerfile + sunucuda deploy

---

## ⚠️ Mevcut Kısıtlar

| Kısıt | Detay |
|-------|-------|
| Veri derinliği | Sadece 7 günlük veri — CatBoost için min. 30 gün önerilir |
| Sıfır metrikler | cart_count %91, view_count %87 sıfır — engagement zayıf |
| Engagement NULL | `engagement_score` DB kolonu hiç hesaplanmamış |
| Karar verilmemiş | Bu sistem tam olarak kurulacak mı? Henüz belirsiz |

---

## 📅 Notlar

- Geliştirme başlangıcı: 2026-02-25
- En son analiz: 2026-03-05 (7 günlük veri, 21,714 kayıt)
- Test sonuçları: Mock veri + gerçek crop verisiyle R²=0.925 başarı
- Gerçek entegrasyon: Veri 30+ güne ulaşınca anlamlı hale gelecek
