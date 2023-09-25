import torch
from torch import nn
import segmentation_models_pytorch as smp


class PoseEstimator(nn.Module):
    def __init__(self, backbone='resnet50', seg_ckpt=None, dim_feat=2048):
        super(PoseEstimator, self).__init__()
        self.seg_model = smp.DeepLabV3Plus(backbone, encoder_weights='imagenet', classes=3)
        if seg_ckpt is not None:
            self.seg_model.load_state_dict(torch.load(seg_ckpt)['model'])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.mlp = nn.Linear(dim_feat, 24)
        self.mlp = nn.Sequential(
            nn.Linear(dim_feat, 256),
            nn.ReLU(),
            nn.Linear(256, 24)
        )

        self.feat = []
        self.seg_model.encoder.register_forward_hook(self.__forward_hook)

    def __forward_hook(self, module, inp, out):
        self.feat.clear()
        self.feat.append(out)

    def forward(self, x):
        m = self.seg_model(x)

        x = self.avgpool(self.feat[-1][-1])
        x = torch.flatten(x, 1)
        x = self.mlp(x)

        return x, m
    
    @torch.no_grad()
    def predict_RT(self, x):
        x = self.seg_model.encoder(x)
        x = self.avgpool(x[-1])
        x = torch.flatten(x, 1)
        x = self.mlp(x)

        return x
    