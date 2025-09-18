import torch
import torch.nn as nn
import torch.nn.functional as F
import timm



class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, g=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class SE(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch//r, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch//r, ch, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

class SpatialAttn(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, 1, bias=True)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        a = self.sig(self.conv(x))
        return x * a

class BiFusionBlock(nn.Module):
    def __init__(self, c_cnn, c_tr, c_out):
        super().__init__()
        self.cnn_reduce = ConvBNReLU(c_cnn, c_out, k=1, s=1, p=0)
        self.tr_reduce  = ConvBNReLU(c_tr,  c_out, k=1, s=1, p=0)
        self.se_cnn = SE(c_out)
        self.se_tr  = SE(c_out)
        self.spa_cnn = SpatialAttn(c_out)
        self.spa_tr  = SpatialAttn(c_out)
        self.fuse = ConvBNReLU(c_out*2, c_out, k=3, s=1, p=1)
    def forward(self, f_cnn, f_tr):
        f_c = self.cnn_reduce(f_cnn)
        f_t = self.tr_reduce(f_tr)
        f_c = self.spa_cnn(self.se_cnn(f_c))
        f_t = self.spa_tr(self.se_tr(f_t))
        x = torch.cat([f_c, f_t], dim=1)
        x = self.fuse(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch, 3,1,1)
        self.conv2 = ConvBNReLU(out_ch, out_ch, 3,1,1)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x); x = self.conv2(x)
        return x

class TransFuseL(nn.Module):
    def __init__(self, out_channels=1, pretrained=True):
        super().__init__()
        # CNN kolu: ResNet-50 (features_only)
        self.cnn = timm.create_model(
            'resnet50', features_only=True, pretrained=pretrained,
            out_indices=(1,2,3,4)
        )
        # Transformer kolu: Swin-Large 384
        self.tr  = timm.create_model(
            'swin_large_patch4_window12_384', features_only=True, pretrained=pretrained,
            out_indices=(0,1,2,3)
        )

        self.c_ch = self.cnn.feature_info.channels()   # beklenen: [256, 512, 1024, 2048]
        self.t_ch = self.tr.feature_info.channels()    # beklenen: [192, 384, 768, 1536]

        # ---- BiFusion/Decoder/Head aynen seninkiler ----
        self.bf4 = BiFusionBlock(self.c_ch[3], self.t_ch[3], 512)
        self.bf3 = BiFusionBlock(self.c_ch[2], self.t_ch[2], 256)
        self.bf2 = BiFusionBlock(self.c_ch[1], self.t_ch[1], 128)
        self.bf1 = BiFusionBlock(self.c_ch[0], self.t_ch[0], 64)

        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        self.head = nn.Sequential(
            ConvBNReLU(64, 32, 3,1,1),
            nn.Conv2d(32, out_channels, 1)
        )

    def forward(self, x):
        c_feats = self.cnn(x)
        t_feats = self.tr(x)

        # --- NHWC → NCHW düzeltmesi (gerekirse) ---
        fixed_t_feats = []
        for i, tf in enumerate(t_feats):
            exp_c = self.t_ch[i]  # beklenen kanal sayısı (örn. 192/384/768/1536)
            # Eğer [B,H,W,C] geldiyse ve C == exp_c ise NCHW'ye çevir
            if tf.dim() == 4 and tf.size(1) != exp_c and tf.size(-1) == exp_c:
                tf = tf.permute(0, 3, 1, 2).contiguous()
            fixed_t_feats.append(tf)
        t_feats = fixed_t_feats
        # -------------------------------------------

        c1, c2, c3, c4 = c_feats
        t1, t2, t3, t4 = t_feats

        f4 = self.bf4(c4, t4)
        f3 = self.bf3(c3, t3)
        f2 = self.bf2(c2, t2)
        f1 = self.bf1(c1, t1)
        x = self.dec3(f4, f3)
        x = self.dec2(x,  f2)
        x = self.dec1(x,  f1)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        logit = self.head(x)
        logit = F.interpolate(logit, scale_factor=2, mode='bilinear', align_corners=False)
        return logit