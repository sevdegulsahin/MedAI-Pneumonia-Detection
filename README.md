# MedAI: Derin Öğrenme ve Açıklanabilir Yapay Zeka (XAI) ile Zatürre Teşhis Sistemi

Bu proje, pediatrik göğüs röntgeni (X-Ray) görüntülerinden otomatik olarak Zatürre (Pneumonia) teşhisi koyabilen, %99.4 duyarlılık (Recall) başarısına sahip bir derin öğrenme sistemidir. Sistem, tıbbi kararların şeffaflığını sağlamak için **Açıklanabilir Yapay Zeka (XAI)** tekniklerini kullanır ve özgün mimarisini endüstri standardı olan **ResNet18** ile kıyaslar.

## Proje Bağlantıları
* Canlı Uygulama (Hugging Face Spaces): [MedAI-Pneumonia-Diagnosis](https://huggingface.co/spaces/sevdegulsahin/MedAI-Pneumonia-Diagnosis)
* Veri Kaynağı: [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## 1. Mimari Karşılaştırma: MedAI_DeepCNN vs. ResNet18
Proje kapsamında iki farklı yaklaşım test edilmiştir:

* **MedAI_DeepCNN (Özgün Mimari):** Tıbbi görüntülerin gri tonlamalı doku özelliklerini yakalamak için özel olarak tasarlanmış 4 katmanlı konvolüsyonel bir yapıdır. AdamW optimizasyonu ile %97.96 eğitim doğruluğuna ulaşmıştır.
* **ResNet18 (Transfer Learning):** Derin ağlardaki "kaybolan gradyan" sorununu aşmak için "Residual (Artık) Bağlantılar" kullanan, ImageNet üzerinde önceden eğitilmiş bir mimaridir. MedAI_DeepCNN'e göre daha stabil bir eğitim eğrisi çizmiş, ancak özgün modelimiz sınıflandırma metriklerinde rekabetçi sonuçlar vermiştir.

---

## 2. Açıklanabilir Yapay Zeka (XAI) Nedir?
Geleneksel derin öğrenme modelleri genellikle "kara kutu" (black box) olarak çalışır; yani bir tahmini neden yaptığını açıklayamazlar. **Açıklanabilir Yapay Zeka (XAI)**, modelin karar verme mekanizmasını insanlar için anlaşılır hale getiren yöntemler bütünüdür.

### **Grad-CAM Teknolojisi:**
Bu projede XAI yöntemi olarak **Grad-CAM (Gradient-weighted Class Activation Mapping)** kullanılmıştır:
* **Isı Haritası (Heatmap):** Modelin röntgen üzerindeki hangi piksellere veya dokulara bakarak "Zatürre" dediğini görselleştirir.
* **Klinik Doğrulama:** Hekimlerin, yapay zekanın işaretlediği bölge ile tıbbi bilgideki patolojik bulguları (buzlu cam görüntüsü, infiltrasyon vb.) karşılaştırmasına olanak tanır.
* **Güven:** Tahminlerin rastgele değil, akciğerlerdeki gerçek lezyon bölgelerine dayanıp dayanmadığını kanıtlar.

---

## 3. Eğitim ve Performans Analizi
Model, 10 epoch boyunca eğitilmiş ve her aşamada performansı titizlikle takip edilmiştir:

* **Accuracy (Doğruluk):** Her iki model de eğitim sonunda yüksek başarım göstermiştir.
* **Loss (Kayıp):** Kayıp grafiğindeki düzenli düşüş, modellerin veriden genelleme yapabildiğini doğrulamaktadır.
* **Kritik Metrik (Recall):** Klinik ortamda en önemli değer olan Pneumonia Recall oranı %99.4 olarak ölçülmüştür.

---

## 4. Modüler Proje Organizasyonu
Proje, hiyerarşik bir dosya yapısı ile organize edilmiştir:

* **`model.py`**: Hem MedAI_DeepCNN hem de ResNet18 tanımlamaları ile Grad-CAM algoritmalarını içerir.
* **`app.py`**: Gradio arayüzü ile modelin canlı test edilmesini sağlar.
* **`train.py` & `eval.py`**: Eğitim ve performans metriklerinin (Confusion Matrix) yönetimini yapar.
* **`derin_ogrenme_rapor.pdf`**: Projenin detaylı teknik raporu.

---

## 5. Gelecekte Yapılabilecek Çalışmalar (Future Work)
* **Veri Dengesi:** Normal sınıfındaki Recall oranını artırmak için veri artırma teknikleri çeşitlendirilebilir.
* **Hibrit Modeller:** CNN katmanları ile Vision Transformer (ViT) blokları birleştirilerek global öznitelik çıkarımı geliştirilebilir.
* **Mobil Uygulama:** Model, saha çalışanları için MobileNet gibi hafif mimarilerle mobil cihazlara entegre edilebilir.

---
Bu çalışma, Bilgisayar Mühendisliği Derin Öğrenme bitirme projesi kapsamında geliştirilmiştir.
