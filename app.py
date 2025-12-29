import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.cm as cm
import os
from model import MedAI_DeepCNN, get_resnet_model

DEVICE = torch.device("cpu")

model_me = MedAI_DeepCNN().to(DEVICE)
model_path_me = "best_MedAI_DeepCNN.pth" if os.path.exists("best_MedAI_DeepCNN.pth") else "best_medai_cnn_v1.pth"
model_me.load_state_dict(torch.load(model_path_me, map_location=DEVICE))
model_me.eval()

model_res = get_resnet_model().to(DEVICE)
model_res.load_state_dict(torch.load("best_ResNet18.pth", map_location=DEVICE))
model_res.eval()

t_common = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_path = 'chest_xray/test/' 

sabit_ornekler = [
  
    [os.path.join(test_path, "NORMAL/IM-0011-0001.jpeg")], 
    [os.path.join(test_path, "NORMAL/IM-0013-0001.jpeg")],
    
    [os.path.join(test_path, "PNEUMONIA/person100_bacteria_475.jpeg")],
    [os.path.join(test_path, "PNEUMONIA/person85_bacteria_421.jpeg")]
]

mevcut_ornekler = []
for ornek in sabit_ornekler:
    if os.path.exists(ornek[0]):
        mevcut_ornekler.append(ornek)
    else:
        print(f"⚠️ Dosya yolu hatalı veya dosya yok: {ornek[0]}")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.grads = None
        self.acts = None
        target_layer.register_forward_hook(self.save_act)
        target_layer.register_full_backward_hook(self.save_grad)
        
    def save_act(self, m, i, o): self.acts = o
    def save_grad(self, m, gi, go): self.grads = go[0]
    
    def __call__(self, x, idx=None):
        self.model.zero_grad()
        out = self.model(x)
        if idx is None: idx = out.argmax(dim=1)
        out[0, idx].backward()
        w = torch.mean(self.grads, dim=(2, 3), keepdim=True)
        cam = F.relu(torch.sum(w * self.acts, dim=1)[0])
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        return cam.data.cpu().numpy()

grad_cam = GradCAM(model_me, model_me.layer3)

def update_status_bar(image_path):
    if not image_path: return "### 📁 Analiz için görsel seçin"
    path_upper = str(image_path).upper()
    if "PNEUMONIA" in path_upper or "PERSON" in path_upper:
        return "### ⚠️ GERÇEK DURUM: **ZATÜRRE (PNEUMONIA)**"
    return "### ✅ GERÇEK DURUM: **SAĞLIKLI (NORMAL)**"

def predict(image_path):
    if not image_path: return None, None, None
    img_pil = Image.open(image_path).convert("RGB")
    inp = t_common(img_pil).unsqueeze(0).to(DEVICE)
    
    T = 1.4 
    
    p1_out = model_me(inp)
    p1 = F.softmax(p1_out / T, dim=1)[0]
    
    with torch.no_grad():
        p2_out = model_res(inp)
        p2 = F.softmax(p2_out / T, dim=1)[0]
    
    cam_map = grad_cam(inp, p1.argmax().item())
    cam_map = cv2.resize(cam_map, (224, 224))
    orig_img = np.array(img_pil.resize((224, 224)))
    heatmap = np.uint8(255 * cm.jet(cam_map)[:, :, :3])
    overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
    
    return overlay, {"SAĞLIKLI": float(p1[0]), "ZATÜRRE": float(p1[1])}, \
           {"SAĞLIKLI": float(p2[0]), "ZATÜRRE": float(p2[1])}

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 MedAI: Klinik Karar Destek Sistemi")
    gr.Markdown("Görüntüleme tabanlı zatürre teşhisi ve Grad-CAM lezyon lokalizasyon analizi.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="filepath", label="Röntgen Görseli")
            status_box = gr.Markdown("### 📁 Analiz için görsel seçin")
            analyze_btn = gr.Button("🔍 ANALİZ ET", variant="primary")
            
            gr.Examples(
                examples=mevcut_ornekler, 
                inputs=input_img, 
                label="Güvenli Sunum Örnekleri (2 Sağlıklı - 2 Zatürre)",
                examples_per_page=4 
            )

        with gr.Column(scale=1):
            out_cam = gr.Image(label="Patolojik Odak Analizi (Grad-CAM)")
            out_me = gr.Label(label="MedAI CNN Tahmini")
            out_res = gr.Label(label="Referans ResNet-18 Tahmini")

    input_img.change(fn=update_status_bar, inputs=input_img, outputs=status_box)
    analyze_btn.click(fn=predict, inputs=input_img, outputs=[out_cam, out_me, out_res])

if __name__ == "__main__":
    demo.launch()