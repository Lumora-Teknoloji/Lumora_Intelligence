# 🧠 E-Ticaret Trend Analiz Motoru: Algoritmalar ve Kullanım Nedenleri

Bu doküman, sistemimizde ürünlerin trend olup olmadığını, patlama yapıp yapmayacağını veya düşüşe geçip geçmediğini anlamak için kullandığımız algoritmaları ve bunların seçilme nedenlerini açıklar.

Sistemimiz tek bir zayıf noktaya bağlı kalmamak için **Hibrit (Ensemble)** bir yapı kullanır. Farklı algoritmalar birbirinin eksiğini kapatır.

---

## 1. CatBoost (Categorical Boosting) Karar Ağaçları
**Nerede Kullanılıyor?** Modelin kalbinde, ana tahmin (prediction) mekanizmasında.

**Neden Kullanıyoruz?**
- E-ticaret verilerinin çoğu sayılardan değil **kategorik (metinsel)** özelliklerden oluşur (Örn: Marka: *Setre*, Renk: *Krem*, Kumaş: *Saten*, Boy: *Kısa*).
- Geleneksel algoritmalar bu metinleri anlamakta zorlanırken, Yandex tarafından geliştirilen **CatBoost**, kategorik verilerle dışarıdan hiçbir dönüştürme yapmadan doğrudan ve kusursuz çalışır.
- Boş gelen verilerde (`NaN` veya `unknown`) ağlamaz, sistemi çökertmez; aksine "bu verinin boş olması da bir işarettir" diyerek bunu öğrenmenin bir parçası yapar.
- Birbirine bağlı özellikleri (örneğin Yaz mevsimi + Kısa elbise) harika bir şekilde yakalar.

---

## 2. Kalman Filtresi (Kalman Filter)
**Nerede Kullanılıyor?** Günlük zaman serilerindeki (sepete eklenme, görüntülenme gibi) gürültüleri temizleyip gerçek ivmeyi (velocity) bulmada.

**Neden Kullanıyoruz?**
- Aslında bir uzay/havacılık algoritması olan Kalman Filtresi, anlık sapmaları (ürünün o gün tesadüfen 1 sepete atılması veya sistem hatasıyla 50 sepete atılması) filtreler.
- Pazarlamada her günkü veri çok gürültülüdür (Botlar girebilir, yanlış tıklamalar olabilir). Kalman Filtresi bu **gürültülerin (noise) arkasındaki gerçek trendi ve hızı (velocity)** tahmin eder.
- Dünkü duruma ve bugünkü yeni veriye bakarak "gerçek talep şu anda X" dememizi sağlar. Kısacası anlık zıplamalara aldanmamızı önler.

---

## 3. Z-Score Anomaly Detection (Anormallik Tespiti)
**Nerede Kullanılıyor?** Eğitim verisini modele sokmadan hemen önceki temizlik aşamasında.

**Neden Kullanıyoruz?**
- Bazen bir ürün saatlik büyük bir indirime girer veya bir influencer paylaşır. O gün sepete eklenme sayısı normalin 100 katına çıkar ve ertesi gün sönüp eski haline döner.
- Bu gerçek bir trend değildir, sadece bir "anormallik"tir (outlier).
- Eğer modeli bu veriyle eğitirsek, makine "böyle zıplayanlar trenddir" diye yanlış öğrenir. **Z-Score**, istatistiksel olarak aşırı sapan (standart sapmanın dışına taşan) bu sahte patlamaları bulur ve o günleri eğitim setinden kesip atar. "Z-Score: 111 hatalı kayıt filtrelendi" uyarısının sebebi budur.

---

## 4. Zaman Serisi Büyüme Oranı (Empirical Growth Rate)
**Nerede Kullanılıyor?** Trend hesaplama final skorunda "Gerçeklik Kontrolü" olarak.

**Neden Kullanıyoruz?**
- Yapay zeka ve karmaşık matematik bazen çok bariz olanı gözden kaçırabilir veya fazla karmaşık düşünebilir.
- Bu formül çok basittir: `(Son Dönem Ortalaması - İlk Dönem Ortalaması) / İlk Dönem Ortalaması`.
- Sırf yapay zeka bir ürüne "bu kesin trend olacak" dedi diye ona hemen 100 puan vermeyiz. Geçmiş haftaya kıyasla **gerçekten sepete eklenme sayısı ne kadar artmış?** sorusunun somut, inkar edilemez matematiksel cevabıdır. Yapay zekanın uçuk tahminlerini ayakları yere basacak şekilde dengeler (Ensemble weight: `score_growth`).

---

## 5. Grid Search Optimizasyonu
**Nerede Kullanılıyor?** `finetune.py` ve büyük test senaryosunda parametrelerin ayarlanmasında.

**Neden Kullanıyoruz?**
- Elimizde 3 ana güç var: CatBoost'un tahmini (Demand), Kalman'ın hızı (Velocity) ve Gerçek Büyüme (Growth). Ama hangisine ne kadar güveneceğiz? %50 Growth, %30 Velocity mi yapmalıyız, yoksa başka bir şey mi?
- Grid Search, bizim adımıza binlerce farklı "ağırlık (weight)", "maksimum sınır (clip)" ve "etiket eşiği" kombinasyonunu tek tek dener.
- Bizim gözle veya tahminle bulamayacağımız, sisteme %100 doğruluk sağlayan altın oranları (Örn: `g=0.50 v=0.30 d=0.20`) matematiksel olarak kanıtlayarak bulur.

---

### Özet Mekanizma:
1. **Z-Score** verideki sahte anlık patlamaları temizler (Güvenlik).
2. **CatBoost** ürünün DNA'sına (boy, renk, marka, yaş) bakarak temel potansiyelini öngörür (Tahmin).
3. **Kalman Filtresi** günlük gürültüleri ekarte edip ürünün anlık ivmesini hesaplar (Hız radarı).
4. **Growth Rate** geçmişe bakıp "gerçekte somut olarak ne kadar büyüdük" sorusunun sağlamasını yapar (Kanıt).
5. Son olarak **Grid Search**'ün bulduğu katsayılarla bu 3 skor toplanarak kusursuz `TREND_SCORE` ve `TREND_LABEL` oluşturulur.
