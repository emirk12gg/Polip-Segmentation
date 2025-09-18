IMG_SIZE  = 512
BATCH     = 32
EPOCHS    = 40
LR        = 1e-3

# Colab + Py3.12 teardown bug'ını önlemek için 0 yapıyoruz
NUM_WORK  = 8

ENCODER = "timm-mobilenetv3_large_100"
ENCODER_WEIGHTS = "imagenet"


import torch
import segmentation_models_pytorch as smp


def load_smp_checkpoint(model, ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "ema", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if any(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    try:
        missing, unexpected = model.load_state_dict(ckpt, strict=True)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(ckpt, strict=False)

    model.to(device).eval()
    return missing if 'missing' in locals() else [], unexpected if 'unexpected' in locals() else []

# örnek kullanım
device = "cuda" if torch.cuda.is_available() else "cpu"



def DeepLabV3(pt_path = "/content/best_mbv3_binary (1).pt"):
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=None,   # eğitimden gelen ağırlıkları yükleyeceğin için None tutmak daha temiz
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)

    missing, unexpected = load_smp_checkpoint(model, pt_path , device)
    print("missing:", missing, "  unexpected:", unexpected)
