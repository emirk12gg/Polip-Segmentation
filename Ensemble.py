import os, cv2, json, numpy as np, torch, matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from models.PraNet.PraNet import PraNet
from models.TransFuseL.TransFuseL import TransFuseL
from models.UACANET.UACANET import UACANet
from models.DeepLabV3.DeepLabv3 import DeepLabV3

DATA_ROOT   = "/content/content/POLIP_competition_dataset/POLIP_competition_dataset"
IMAGES_DIR  = DATA_ROOT
OUTPUT_DIR  = ""  
IMG_SIZE    = 384

THR_BIN   = 0.50
THR_LOGIT = 0.40
VOTE_K    = 3

AREA_SMALL_MAX = int(0.005 * IMG_SIZE * IMG_SIZE)
AREA_MED_MAX   = int(0.025 * IMG_SIZE * IMG_SIZE)
MIN_AREA       = int(0.0008 * IMG_SIZE * IMG_SIZE)

W_SMALL = (0.30, 0.30, 0.40, 0.00)
W_MED   = (0.20, 0.20, 0.45, 0.15)
W_LARGE = (0.10, 0.10, 0.40, 0.40)

USE_OPEN_MORPH   = True
MORPH_KERNEL_SZ  = 3
MORPH_ITER       = 1

SHOW_PREVIEW = True     
PREVIEW_MAX  = 20       
ALPHA        = 0.45     


PreNetPath = "/content/drive/MyDrive/Polyp Models/PraNet-60 90iou.pth"
TransFuseLPath = "/content/drive/MyDrive/Polyp Models/transfuse_l_merged_best.pth"

device ="cuda"

import torch
import torch.nn as nn

model1 = UACANet().to("cuda")

model1.load_state_dict(torch.load("/content/drive/MyDrive/Polyp Models/latest1.2.1.pth"))

models = [model1]
for m in models:
    m.eval().cuda()

model3 = TransFuseL(out_channels=1, pretrained=False).to(device)

ckpt = torch.load(TransFuseLPath, map_location="cpu")
sd = ckpt.get('state_dict', ckpt)
sd = {k.replace('module.', ''): v for k, v in sd.items()}
model3.load_state_dict(sd, strict=True) 

model2 = PraNet().to(device)
model2.load_state_dict(torch.load(PreNetPath, map_location="cpu"))

model4 = DeepLabV3()
model4.eval().cuda()




#KTO_1_NR!c1YJ(aL634


def _pick_device(*models):
    for m in models:
        p = next(m.parameters(), None)
        if p is not None:
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = _pick_device(model3, model2, model1, model4)
for _m in (model1, model2, model3, model4):
    _m.eval().to(device)


IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]
_tf_single = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMNET_MEAN, std=IMNET_STD),
    ToTensorV2()
])

def _pick_weights(area):
    if area <= AREA_SMALL_MAX: return W_SMALL, "SMALL"
    if area <= AREA_MED_MAX:   return W_MED,   "MEDIUM"
    return W_LARGE, "LARGE"

@torch.no_grad()
def _pranet_logits(m, x):
    out = m(x);  return out[0] if isinstance(out, (tuple, list)) else out

@torch.no_grad()
def _uacanet_logits(m, x):
    out = m({'image': x})
    if isinstance(out, (tuple, list)): out = out[0]
    if isinstance(out, dict): out = out.get('pred', out.get('logits', out.get('out', out)))
    return out

@torch.no_grad()
def _deeplab_logits(m, x): return m(x)

@torch.no_grad()
def _forward_all_single(x):
    p1 = torch.sigmoid(_uacanet_logits(model1, x)) 
    p2 = torch.sigmoid(_pranet_logits(model2, x)) 
    p3 = torch.sigmoid(model3(x))                    
    p4 = torch.sigmoid(_deeplab_logits(model4, x))   
    return p1, p2, p3, p4

def mask_to_polygons(bin_mask):
    H, W = bin_mask.shape
    m255 = (bin_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    min_area = max(3, int(0.0002 * H * W))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area: continue
        eps = 0.001 * np.hypot(H, W)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if approx.shape[0] < 3: continue
        pts = approx.reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, W-1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H-1)
        polys.append(pts.astype(int).reshape(-1).tolist())
    return polys

def overlay_mask(img_rgb, mask_bin, color=(0,255,0), alpha=ALPHA):
    base = img_rgb.copy()
    overlay = base.copy()
    overlay[mask_bin.astype(bool)] = color
    return cv2.addWeighted(overlay, alpha, base, 1-alpha, 0)


@torch.no_grad()
def predict_mask_and_polygons_for_image(image_path):
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert img_bgr is not None, f"Görsel okunamadı: {image_path}"
    H0, W0 = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    aug = _tf_single(image=img_rgb)
    x = aug["image"].unsqueeze(0).float().to(device)  

    p1, p2, p3, p4 = _forward_all_single(x)
    b1 = (p1 > THR_BIN).float()
    b2 = (p2 > THR_BIN).float()
    b3 = (p3 > THR_BIN).float()
    b4 = (p4 > THR_BIN).float()
    votes = (b1 + b2 + b3 + b4)

    union_bin = (votes >= 2).squeeze(1).cpu().numpy().astype(np.uint8)[0]
    if USE_OPEN_MORPH:
        k = np.ones((MORPH_KERNEL_SZ, MORPH_KERNEL_SZ), np.uint8)
        union_bin = cv2.morphologyEx(union_bin, cv2.MORPH_OPEN, k, iterations=MORPH_ITER)

    num, labels = cv2.connectedComponents(union_bin, connectivity=8)
    h, w = labels.shape
    dyn_ens = torch.zeros((1, 1, h, w), device=device)

    for lab in range(1, num):
        mask_np = (labels == lab)
        area = int(mask_np.sum())
        if area < MIN_AREA: continue
        wts, _ = _pick_weights(area)
        msk = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)
        dyn_ens[0,0] += wts[0]*p1[0,0]*msk + wts[1]*p2[0,0]*msk + wts[2]*p3[0,0]*msk + wts[3]*p4[0,0]*msk

    pred_small = ((dyn_ens > THR_LOGIT) & (votes >= VOTE_K)).float().cpu().numpy()[0,0]
    pred_mask  = cv2.resize(pred_small.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST)

    polygons = mask_to_polygons(pred_mask)
    return img_rgb, pred_mask, polygons 

def predict_dataset_with_preview_to_json(
    images_dir,
    output_dir,
    takim_id,
    takim_adi,
    faz="Demo",
    aciklama="",
    versiyon="v1.0",
    image_exts=(".png",".jpg",".jpeg",".bmp",".tif",".tiff"),
    show_preview=SHOW_PREVIEW,
    preview_max=PREVIEW_MAX
):
    os.makedirs(output_dir, exist_ok=True)
    safe_team = "".join(ch if ch.isalnum() or ch in (" ","-","_") else "_" for ch in takim_adi).strip().replace(" ","_")
    out_name = f"{takim_id}_{safe_team}_Polip_{faz}.json"
    out_path = os.path.join(output_dir, out_name)

    files = sorted([
        f for f in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, f)) and os.path.splitext(f)[1].lower() in image_exts
    ])

    results = {
        "kunye": {
            "takim_adi": takim_adi,
            "takim_id": takim_id,
            "aciklama": aciklama,
            "versiyon": versiyon
        },
        "tahminler": []
    }

    shown = 0
    for fname in tqdm(files, desc="Maskesiz tahmin + önizleme"):
        fpath = os.path.join(images_dir, fname)
        try:
            rgb, pred_mask, polys = predict_mask_and_polygons_for_image(fpath)
            results["tahminler"].append({"filename": fname, "segmentations": polys})

            if show_preview and (preview_max is None or shown < preview_max):
                vis = overlay_mask(rgb, pred_mask>0, color=(0,255,0), alpha=ALPHA)
                plt.figure(figsize=(10,4))
                plt.subplot(1,2,1); plt.imshow(rgb); plt.title(f"{fname} | ORİJİNAL"); plt.axis("off")
                plt.subplot(1,2,2); plt.imshow(vis); plt.title(f"{fname} | PRED"); plt.axis("off")
                plt.tight_layout(); plt.show()
                shown += 1

        except AssertionError as e:
            print(f"[UYARI] {fname}: {e}")
            results["tahminler"].append({"filename": fname, "segmentations": []})
        except Exception as e:
            print(f"[HATA] {fname}: {e}")
            results["tahminler"].append({"filename": fname, "segmentations": []})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nJSON hazır: {out_path}")
    return out_path

predict_dataset_with_preview_to_json(
    images_dir=IMAGES_DIR,
    output_dir=OUTPUT_DIR,
    takim_id="737670",
    takim_adi="ZAĞANOS YZ",
    faz="Demo",
    aciklama="Dinamik ensemble (maskesiz) — JSON'dan önce önizleme",
    versiyon="v1.0",
    show_preview=True,
    preview_max=1   
)
