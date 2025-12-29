# MedAI: Derin Öğrenme ve Açıklanabilir Yapay Zeka (XAI) ile Zatürre Teşhis Sistemi

MedAI, pediatrik göğüs röntgeni (Chest X-Ray) görüntülerinden otomatik olarak **Zatürre (Pneumonia)** teşhisi koyabilen, **%99.4 Recall (Duyarlılık)** başarısına sahip bir derin öğrenme tabanlı tıbbi karar destek sistemidir.

Bu proje, tıbbi kararların şeffaflığını artırmak amacıyla **Açıklanabilir Yapay Zeka (Explainable AI - XAI)** tekniklerini kullanmakta ve özgün olarak geliştirilen **MedAI_DeepCNN** mimarisini endüstri standardı olan **ResNet18** modeli ile karşılaştırmalı olarak analiz etmektedir.

---

## Proje Bağlantıları

- Canlı Uygulama (Hugging Face Spaces): [MedAI-Pneumonia-Diagnosis](https://huggingface.co/spaces/sevdegulsahin/MedAI-Pneumonia-Diagnosis)
- Veri Seti: [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## 1. Mimari Karşılaştırma: MedAI_DeepCNN vs. ResNet18

Proje kapsamında iki farklı derin öğrenme yaklaşımı değerlendirilmiştir:

### MedAI_DeepCNN (Özgün Mimari)
- Gri tonlamalı tıbbi görüntülerin doku özelliklerini yakalamak üzere özel olarak tasarlanmış, **4 katmanlı konvolüsyonel sinir ağı** mimarisidir.
- **AdamW** optimizasyon algoritması kullanılmıştır.
- Maksimum **%97.96 eğitim doğruluğu** elde edilmiştir.

### ResNet18 (Transfer Learning)
- Derin ağlarda görülen **kaybolan gradyan** problemini çözmek için **Residual (Artık) Bağlantılar** kullanır.
- **ImageNet** veri seti üzerinde önceden eğitilmiştir.
- Eğitim süreci daha stabil ilerlemiş, özgün mimari ile rekabetçi sınıflandırma sonuçları üretmiştir.

---

## Mimari Karşılaştırma Tablosu

Modellerin performans metrikleri ve dosya çıktıları aşağıda detaylandırılmıştır:

| Özellik / Metrik | MedAI_DeepCNN (Özgün) | ResNet18 (Baseline) |
| :--- | :--- | :--- |
| **Eğitim Doğruluğu (Max)** | %97.96 | ~%98.8 |
| **Eğitim Kaybı (Min)** | ~0.08 | ~0.03 |
| **Ağırlık Dosyası** | best_MedAI_DeepCNN.pth | best_ResNet18.pth |
| **Hata Matrisi** | confusion_matrix_MedAI_CNN.png | confusion_matrix_ResNet18.png |
| **Metrik Raporu** | rapor_MedAI_CNN.txt | rapor_ResNet18.txt |

---

## 2. Açıklanabilir Yapay Zeka (XAI) Nedir?

Geleneksel derin öğrenme modelleri genellikle **kara kutu (black-box)** olarak çalışır; yani verdikleri kararların nedenlerini açıklayamazlar. **Açıklanabilir Yapay Zeka (XAI)**, modelin karar verme mekanizmasını insanlar için anlaşılır hale getiren yöntemler bütünüdür.

### Grad-CAM Teknolojisi

Bu projede XAI yöntemi olarak **Grad-CAM (Gradient-weighted Class Activation Mapping)** kullanılmıştır:

- **Isı Haritası (Heatmap):** Modelin röntgen görüntüsü üzerinde hangi bölgelere odaklanarak "Pneumonia" kararı verdiğini görselleştirir.
- **Klinik Doğrulama:** Hekimlerin, yapay zekanın işaretlediği alanları tıbbi patolojik bulgular (infiltrasyon, buzlu cam görünümü vb.) ile karşılaştırmasına imkân tanır.
- **Güvenilirlik:** Model tahminlerinin rastgele değil, gerçek akciğer lezyonlarına dayandığını gösterir.

---

## 3. Eğitim ve Performans Analizi

Modeller, **10 epoch** boyunca eğitilmiş ve her aşamada performans metrikleri detaylı şekilde analiz edilmiştir.

### Eğitim Dinamikleri

- **Accuracy (Doğruluk):** ResNet18 modeli eğitim boyunca %97–99 aralığında oldukça stabil bir seyir izlemiştir. MedAI_DeepCNN modeli ise ilk epochlarda dalgalanmalar yaşamış, 6. epochtan sonra %97 seviyesinde kararlılığa ulaşmıştır.
- **Loss (Kayıp):** Her iki modelde de kayıp değeri düzenli olarak azalmıştır. ResNet18 yaklaşık **0.03**, MedAI_DeepCNN ise yaklaşık **0.08** seviyesinde eğitimi tamamlamıştır.
- **Kritik Klinik Metrik (Recall):** Zatürre sınıfı için **%99.4 Recall** elde edilmiştir.


---

## 4. Modüler Proje Organizasyonu

Proje, sürdürülebilir ve modüler bir dosya yapısı ile tasarlanmıştır:

- `model.py`: MedAI_DeepCNN ve ResNet18 mimarileri ile Grad-CAM algoritmalarını içerir.
- `app.py`: Gradio tabanlı arayüz ile modelin canlı test edilmesini sağlar.
- `train.py` & `eval.py`: Eğitim süreci ve performans değerlendirmelerini (Confusion Matrix, raporlar) yönetir.
- `derin_ogrenme_rapor.pdf`: Projenin detaylı teknik raporu.
- `requirements.txt`: Gerekli Python kütüphaneleri listesi.

---

## 5. Gelecekte Yapılabilecek Çalışmalar (Future Work)

- **Veri Dengesi:** Normal sınıfı için Recall oranını artırmak amacıyla veri artırma teknikleri çeşitlendirilebilir.
- **Hibrit Modeller:** CNN katmanları ile Vision Transformer (ViT) blokları birleştirilerek global öznitelik çıkarımı güçlendirilebilir.
- **Mobil Uygulama:** MobileNet gibi hafif mimariler kullanılarak model mobil cihazlara entegre edilebilir.

---

Bu çalışma, **Bilgisayar Mühendisliği – Derin Öğrenme Bitirme Projesi** kapsamında geliştirilmiştir.
