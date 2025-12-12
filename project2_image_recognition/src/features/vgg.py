# src/features/vgg.py (교체)
import torch, torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F

_MODELS = {
    "vgg13": torchvision.models.vgg13_bn,
    "vgg19": torchvision.models.vgg19_bn,
}

class VGGExtractor:
    def __init__(self, name="vgg19", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net = _MODELS[name](weights="DEFAULT").to(self.device).eval()
        self.features = torch.nn.Sequential(
            net.features,            # conv blocks
            net.avgpool              # 7×7 AdaptiveAvgPool
        )                            # 결과: (B,512,7,7)
        self.preprocess = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std =[0.229,0.224,0.225]),
        ])

    @torch.no_grad()
    def transform(self, paths, batch_size=32):
        out = []
        for i in range(0, len(paths), batch_size):
            imgs = torch.stack([self.preprocess(Image.open(p).convert("RGB"))
                                for p in paths[i:i+batch_size]]).to(self.device)
            feat = self.features(imgs)          # (B,512,7,7)
            feat = F.adaptive_avg_pool2d(feat, 1).view(feat.size(0), -1)  # (B,512)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)        # L2-norm
            out.append(feat.cpu())
        return torch.cat(out).numpy()           # shape: (N,512)
