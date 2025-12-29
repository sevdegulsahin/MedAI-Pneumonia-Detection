import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

# Model dosyandan sınıfları çekiyoruz
from model import MedAI_DeepCNN, get_resnet_model

# Cihaz ayarı (MacBook için mps, yoksa cpu)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_test_loader():
    # Klasör yapısını kontrol et
    possibilities = ['chest_xray/test', 'chest_xray/chest_xray/test', 'test']
    test_dir = None
    for p in possibilities:
        if os.path.exists(p):
            test_dir = p
            break
    
    if test_dir is None:
        raise FileNotFoundError("❌ Test klasörü bulunamadı! Lütfen yolu kontrol edin.")
        
    print(f"✅ Test verisi bulundu: {test_dir}")
    
    t = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_data = datasets.ImageFolder(root=test_dir, transform=t)
    return DataLoader(test_data, batch_size=32, shuffle=False), test_data.classes

def generate_reports(model, model_name, loader, class_names):
    print(f"\n" + "="*40)
    print(f"📊 {model_name} ANALİZİ BAŞLIYOR...")
    print("="*40)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())

    # 1. Karmaşıklık Matrisi (Confusion Matrix) Çizimi
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    print(f"💾 Grafik kaydedildi: confusion_matrix_{model_name}.png")
    plt.show()

    # 2. Kritik Metriklerin (Recall, F1-Score) Terminale Basılması
    print(f"\n🔍 {model_name} İÇİN DETAYLI SINIFLANDIRMA RAPORU:")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    
    # Raporu dosyaya da yazdıralım (Terminalde kaybolursa buradan bakarsın)
    with open(f"rapor_{model_name}.txt", "w") as f:
        f.write(report)
    print(f"📝 Metin raporu kaydedildi: rapor_{model_name}.txt")

if __name__ == "__main__":
    try:
        test_loader, classes = get_test_loader()

        # --- SENİN MODELİN (MedAI_DeepCNN) ---
        model_path_me = "best_MedAI_DeepCNN.pth"
        if os.path.exists(model_path_me):
            cnn = MedAI_DeepCNN().to(DEVICE)
            cnn.load_state_dict(torch.load(model_path_me, map_location=DEVICE))
            generate_reports(cnn, "MedAI_CNN", test_loader, classes)
        else:
            print(f"⚠️ {model_path_me} bulunamadığı için atlanıyor.")

        # --- REFERANS MODEL (ResNet18) ---
        model_path_res = "best_ResNet18.pth"
        if os.path.exists(model_path_res):
            resnet = get_resnet_model().to(DEVICE)
            resnet.load_state_dict(torch.load(model_path_res, map_location=DEVICE))
            generate_reports(resnet, "ResNet18", test_loader, classes)
        else:
            print(f"⚠️ {model_path_res} bulunamadığı için atlanıyor.")

    except Exception as e:
        print(f"❌ Bir hata oluştu: {e}")