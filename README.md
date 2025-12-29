# MedAI: Derin Öğrenme ve Açıklanabilir Yapay Zeka (XAI) ile Zatürre Teşhis Sistemi

Bu proje, pediatrik göğüs röntgeni (X-Ray) görüntülerinden otomatik olarak Zatürre (Pneumonia) teşhisi koyabilen, %99.4 duyarlılık (Recall) başarısına sahip bir derin öğrenme sistemidir. Sistem, sadece bir tahmin üretmekle kalmayıp, Grad-CAM teknolojisi ile teşhislerin tıbbi gerekçelerini görselleştirerek açıklanabilirlik sunar.

## Proje Bağlantıları
* Canlı Uygulama (Hugging Face Spaces): [MedAI-Pneumonia-Diagnosis](https://huggingface.co/spaces/sevdegulsahin/MedAI-Pneumonia-Diagnosis)
* Veri Kaynağı: [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## 1. Mimari Yapı ve Katman Detayları
Sistem kapsamında geliştirilen MedAI_DeepCNN mimarisi, tıbbi görüntülerdeki mikro doku değişimlerini ve patolojik sızıntıları yakalamak üzere özelleştirilmiş 4 temel bloktan oluşmaktadır:

* **Konvolüsyonel Bloklar:** Model, 32'den başlayarak 256'ya kadar artan filtre sayılarına sahip 4 adet Conv2d katmanı kullanır. Bu kademeli yapı, düşük seviyeli kenar bilgilerinden yüksek seviyeli klinik lezyon yapılarına kadar geniş bir özellik çıkarımı yapılmasını sağlar.
* **Batch Normalization Katmanları:** Her konvolüsyon sonrası uygulanan bu katmanlar, içsel değişken kaymasını (internal covariate shift) minimize ederek eğitimin kararlılığını artırır ve aşırı öğrenmeyi (overfitting) engellemeye yardımcı olur.
* **Non-Lineerite ve Pooling:** Aktivasyon fonksiyonu olarak ReLU kullanılarak karmaşık desenlerin çözümlenmesi sağlanırken, MaxPool2d katmanları ile mekansal boyut azaltılarak hesaplama maliyeti optimize edilmiştir.
* **Optimizasyon Stratejisi:** Eğitim sürecinde AdamW optimizasyon algoritması ve CrossEntropyLoss hata fonksiyonu tercih edilerek en iyi ağırlıklar (best weights) güncellenmiştir.

---

## 2. Eğitim Dinamikleri ve Karşılaştırmalı Analiz
Modelin öğrenme performansı, referans olarak kullanılan ResNet18 mimarisi ile 10 epoch boyunca eşzamanlı olarak takip edilmiştir:

* **Accuracy (Doğruluk) Analizi:** MedAI_DeepCNN, eğitim sürecinde %97.96 doğruluk seviyesine ulaşmıştır. Başlangıç aşamasındaki dalgalanmalara rağmen model, 6. epoch'tan itibaren yüksek bir öğrenme kararlılığı yakalamıştır.
* **Loss (Kayıp) Analizi:** Kayıp grafiğinde görülen 0.23'ten 0.08 seviyelerine düzenli azalış, modelin genelleme yeteneğinin yüksek olduğunu ve veriyi ezberlemek yerine öğrendiğini kanıtlamaktadır.
* **Klinik Performans Metrikleri:** Test seti üzerinde yapılan değerlendirmelerde, Pneumonia sınıfında %99.4 Recall (Duyarlılık) oranına ulaşılmıştır. Bu, hayati önem taşıyan "hastayı kaçırmama" kriterinin başarıyla karşılandığını göstermektedir.

---

## 3. Modüler Proje Organizasyonu
Proje, profesyonel yazılım standartlarına uygun olarak modüler bir dosya yapısında hiyerarşik olarak düzenlenmiştir:

* **model.py:** MedAI_DeepCNN sınıf yapısını ve Grad-CAM ısı haritası üretim fonksiyonlarını içeren çekirdek modüldür.
* **app.py:** Gradio arayüzünü yöneten, kullanıcı etkileşimlerini işleyen ve model tahminlerini görselleştiren ana uygulama dosyasıdır.
* **train.py:** Veri setinin yüklenmesi, veri artırma (augmentation) tekniklerinin uygulanması ve 10 epoch'luk eğitim döngüsünün yürütülmesinden sorumlu dosyadır.
* **eval.py:** Eğitim sonrası test verileri üzerinden Karmaşıklık Matrisi (Confusion Matrix) ve detaylı performans raporlarını (Precision, Recall, F1-Score) üreten analiz modülüdür.
* **best_MedAI_DeepCNN.pth:** Eğitim sonucunda doğrulama setinde en yüksek başarıyı gösteren, %99.4 duyarlılığa sahip modelin ağırlık dosyasıdır.
* **derin_ogrenme_rapor.pdf:** Projenin metodolojisini, literatür taramasını ve sonuçlarını içeren kapsamlı teknik rapordur.

---

## 4. Kurulum ve Çalıştırma

# Repoyu klonlayın
git clone [https://github.com/sevdegulsahin/MedAI-Pneumonia-Detection.git](https://github.com/sevdegulsahin/MedAI-Pneumonia-Detection.git)
cd MedAI-Pneumonia-Detection

# Gerekli bağımlılıkları kurun
pip install -r requirements.txt

# Uygulamayı başlatın
python app.py

## 🔍 5. Açıklanabilirlik ve Klinik Karar Destek

Sistem, yalnızca bir tahmin sonucu üretmek yerine, kararlarının arkasındaki nedenleri görselleştirerek sunar.

* **Grad-CAM Teknolojisi:** `Gradient-weighted Class Activation Mapping` algoritması kullanılarak, modelin röntgen üzerinde hangi piksellere odaklandığı tespit edilir.
* **Görsel Kanıt (Heatmap):** Teşhise neden olan patolojik bölgeler (infiltrasyonlar, konsolidasyonlar vb.) bir ısı haritası ile işaretlenir.

> [!IMPORTANT]
> **Klinik Fayda:** Bu şeffaflık, hekimin yapay zeka kararını klinik olarak doğrulamasını sağlar, teşhis sürecindeki belirsizlikleri azaltır ve sisteme duyulan güveni artırarak karar destek mekanizmasını güçlendirir.

---

## 🚀 6. Gelecekte Yapılabilecek Çalışmalar (Future Work)

Modelin klinik kullanım potansiyelini artırmak amacıyla aşağıdaki geliştirmeler yol haritasına eklenmiştir:

- [ ] **Veri Dengesi (Data Imbalance) Yönetimi:** "Normal" sınıfındaki %51'lik duyarlılık oranını yukarı çekmek için **SMOTE** veya gelişmiş **Sınıf Ağırlıklandırma (Class Weighting)** tekniklerinin entegre edilmesi.
- [ ] **Hibrit Mimari Yaklaşımları:** CNN mimarisinin yerel özellik çıkarma yeteneğini, **Vision Transformer (ViT)** blokları ile birleştirerek küresel bağlamın daha iyi yakalanması.
- [ ] **Çoklu Sınıflandırma:** Sistemin kapsamının genişletilerek; Tüberküloz, KOAH ve Akciğer Kanseri gibi hastalıkları da teşhis edebilir hale getirilmesi.
- [ ] **Mobil Entegrasyon:** **MobileNet** veya **TensorFlow Lite** kullanılarak saha çalışanları için optimize edilmiş mobil uygulama desteği.

---

## 🎓 Proje Hakkında
Bu proje, **Bilgisayar Mühendisliği Derin Öğrenme** bitirme çalışması kapsamında geliştirilmiştir.
