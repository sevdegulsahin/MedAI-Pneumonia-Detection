import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import os
import sys # Sistem çıktıları için
from model import MedAI_DeepCNN, get_resnet_model

# Cihaz Ayarı
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Eğitim {DEVICE} üzerinde yapılacak.", flush=True)

EPOCHS = 10
BATCH_SIZE = 32

def get_loaders():
    print("Veriler taranıyor, lütfen bekleyin...", flush=True)
    train_dir = 'chest_xray/train' if os.path.exists('chest_xray/train') else 'chest_xray/chest_xray/train'
    
    if not os.path.exists(train_dir):
        print(f"HATA: {train_dir} yolu bulunamadı! Lütfen veri setini kontrol edin.", flush=True)
        sys.exit()

    t = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_data = datasets.ImageFolder(root=train_dir, transform=t)
    print(f"Toplam {len(full_data)} görsel yüklendi. Bölünüyor...", flush=True)

    train_idx, val_idx = train_test_split(
        np.arange(len(full_data.targets)), 
        test_size=0.15, 
        stratify=full_data.targets,
        random_state=42
    )
    
    tl = DataLoader(Subset(full_data, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(Subset(full_data, val_idx), batch_size=BATCH_SIZE)
    print("Dataloader hazır!", flush=True)
    return tl, vl

def calculate_metrics(loader, model):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img, lbl in loader:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            out = model(img)
            _, p = torch.max(out, 1)
            all_preds.extend(p.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    return acc, precision, recall, f1

def train_loop(model, name, t_loader, v_loader):
    print(f"\n>>> {name} eğitimi başlıyor...", flush=True)
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    history = {"acc": [], "loss": []}
    best_f1 = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        # Batch takibi için sayaç
        for batch_idx, (img, lbl) in enumerate(t_loader):
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(img), lbl)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # Her 20 batch'te bir ekrana küçük bir işaret bas ki donmadığını anlayalım
            if batch_idx % 20 == 0:
                print(".", end="", flush=True)

        acc, prec, rec, f1 = calculate_metrics(v_loader, model)
        avg_loss = epoch_loss / len(t_loader)
        history["acc"].append(acc * 100)
        history["loss"].append(avg_loss)

        print(f"\nEpoch [{epoch+1}/{EPOCHS}] -> Loss: {avg_loss:.4f} | Acc: %{acc*100:.2f} | F1: {f1:.4f}", flush=True)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"best_{name}.pth")
            print(f"--- En iyi {name} kaydedildi (F1: {f1:.4f}) ---", flush=True)

    return history

if __name__ == "__main__":
    tl, vl = get_loaders()
    h1 = train_loop(MedAI_DeepCNN(), "MedAI_DeepCNN", tl, vl)
    h2 = train_loop(get_resnet_model(), "ResNet18", tl, vl)

    print("\nEğitim bitti. Grafikler oluşturuluyor...", flush=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(h1["acc"], label='MedAI DeepCNN'); ax1.plot(h2["acc"], label='ResNet18')
    ax1.set_title('Accuracy'); ax1.legend()
    ax2.plot(h1["loss"], label='MedAI DeepCNN'); ax2.plot(h2["loss"], label='ResNet18')
    ax2.set_title('Loss'); ax2.legend()
    
    plt.savefig('analiz.png')
    print("Grafik 'analiz.png' olarak kaydedildi.", flush=True)
    # plt.show() # Terminali kitlememesi için bunu kapatabilirsin, analiz.png'den bakarsın.