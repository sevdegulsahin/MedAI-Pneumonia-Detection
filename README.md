# MedAI: Derin Öğrenme ve Açıklanabilir Yapay Zeka (XAI) ile Zatürre Teşhis Sistemi

MedAI, pediatrik göğüs röntgeni (Chest X-Ray) görüntülerinden otomatik olarak **Zatürre (Pneumonia)** teşhisi koyabilen, **%99.4 Recall (Duyarlılık)** başarısına sahip derin öğrenme tabanlı bir tıbbi karar destek sistemidir.

Bu çalışma, tıbbi teşhis süreçlerinde şeffaflık sağlamak amacıyla **Açıklanabilir Yapay Zeka (XAI)** tekniklerini entegre eder ve özgün olarak geliştirilen **MedAI_DeepCNN** mimarisini endüstri standardı **ResNet18** modeli ile karşılaştırmalı olarak analiz eder.

---

## Proje Bağlantıları

- **Canlı Uygulama (Hugging Face Spaces):** [MedAI-Pneumonia-Diagnosis](https://huggingface.co/spaces/sevdegulsahin/MedAI-Pneumonia-Diagnosis)
- **Veri Seti:** [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## 1. Mimari Karşılaştırma: MedAI_DeepCNN vs. ResNet18

Proje kapsamında iki farklı derin öğrenme yaklaşımı klinik performans açısından değerlendirilmiştir:

### MedAI_DeepCNN (Özgün Mimari)
- Tıbbi görüntülerin spesifik doku özelliklerini yakalamak üzere optimize edilmiş, **4 katmanlı konvolüsyonel sinir ağı (CNN)** mimarisidir.
- **AdamW** optimizasyon algoritması kullanılarak eğitilmiştir.
- Eğitim sürecinde maksimum **%97.96 doğruluk** oranına ulaşmıştır.

### ResNet18 (Transfer Learning)
- Derin ağlardaki gradyan kaybı problemini aşmak için **Residual (Artık) Bağlantılar** kullanan önceden eğitilmiş bir mimaridir.
- Eğitim sürecinde daha stabil bir öğrenme eğrisi sergilemiş, özgün mimari ile rekabetçi sonuçlar üretmiştir.

---

## 2. Teknik Karşılaştırma ve Performans Analizi

Modellerin 10 epoch süren eğitim süreci sonunda elde edilen nihai metrikleri ve dosya çıktıları aşağıda özetlenmiştir:

| Özellik / Metrik | MedAI_DeepCNN (Özgün) | ResNet18 (Baseline) |
| :--- | :--- | :--- |
| **Maksimum Eğitim Doğruluğu** | %97.96 | ~%98.8 |
| **Minimum Eğitim Kaybı (Loss)** | ~0.08 | ~0.03 |
| **Zatürre Sınıfı Duyarlılığı (Recall)** | **%99.4** | **%100.0** |
| **Model Ağırlık Dosyası** | `best_MedAI_DeepCNN.pth` | `best_ResNet18.pth` |
| **Performans Raporu** | `rapor_MedAI_CNN.txt` | `rapor_ResNet18.txt` |

### Eğitim Dinamikleri Analizi
* **Öğrenme İstikrarı:** ResNet18 modeli eğitim boyunca %97-99 aralığında stabil bir seyir izlerken; MedAI_DeepCNN modeli başlangıçta dalgalanmalar yaşamış, 6. epoch itibarıyla %97 bandında kararlılığa ulaşmıştır.
* **Hata Minimizasyonu:** Her iki modelde de kayıp (loss) değerleri düzenli olarak azalmış; ResNet18 0.03, MedAI_DeepCNN ise 0.08 seviyesinde eğitimi tamamlamıştır.

---

## 3. Açıklanabilir Yapay Zeka (XAI) ve Grad-CAM

Modelin teşhis kararlarını tıbbi olarak doğrulanabilir kılmak için **Grad-CAM** algoritması entegre edilmiştir:

* **Görsel Kanıt:** Sistem, röntgen görüntüsü üzerinde patolojik bulguların (infiltrasyon, buzlu cam görünümü vb.) yoğunlaştığı alanları ısı haritası ile işaretler.
* **Klinik Güven:** Model tahminlerinin rastgele piksellere değil, radyolojik olarak anlamlı lezyon bölgelerine dayandığı bu yöntemle kanıtlanır.

---

## 4. Modüler Proje Organizasyonu

Proje, sürdürülebilir yazılım geliştirme prensiplerine uygun olarak tasarlanmıştır:

* **`model.py`**: MedAI_DeepCNN ve ResNet18 mimarileri ile Grad-CAM algoritmalarının çekirdek tanımlarını içerir.
* **`app.py`**: Gradio kütüphanesi ile geliştirilmiş, modelin canlı test edilmesine olanak sağlayan kullanıcı arayüzü.
* **`train.py` & `eval.py`**: Eğitim döngüsünü ve performans değerlendirme raporlarını yöneten modüller.
* **`derin_ogrenme_rapor.pdf`**: Projenin tüm metodolojik ve teknik detaylarını içeren akademik rapor.

---

## 5. Gelecekte Yapılabilecek Çalışmalar

* **Veri Dengesi:** Normal sınıfı başarımını artırmak için ileri düzey veri artırma (Data Augmentation) teknikleri.
* **Hibrit Mimariler:** CNN mimarisine Vision Transformer (ViT) bloklarının entegre edilerek küresel bağlamın yakalanması.
* **Mobil Uygulama:** Saha çalışanları için MobileNet veya benzeri hafif mimarilerle mobil cihaz entegrasyonu.

---
Bu proje, **Bilgisayar Mühendisliği – Derin Öğrenme Bitirme Çalışması** kapsamında geliştirilmiştir.
