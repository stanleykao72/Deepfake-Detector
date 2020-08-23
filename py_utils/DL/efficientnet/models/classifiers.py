import os
import math
import torch
import numpy as np
from functools import partial
from torch import nn
from torch.nn import DataParallel
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d, AdaptiveMaxPool2d
from timm.models.efficientnet import tf_efficientnet_b2_ns, tf_efficientnet_b3_ns, \
    tf_efficientnet_b4_ns, tf_efficientnet_b5_ns, tf_efficientnet_b6_ns, tf_efficientnet_b7_ns


encoder_params = {
    "tf_efficientnet_b2_ns": {
        "features": 1408,
        "base_net": partial(tf_efficientnet_b2_ns, pretrained=False, drop_path_rate=0.2)
    },
    "tf_efficientnet_b3_ns": {
        "features": 1536,
        "base_net": partial(tf_efficientnet_b3_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b4_ns": {
        "features": 1792,
        "base_net": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.5)
    },
    "tf_efficientnet_b5_ns": {
        "features": 2048,
        "base_net": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b6_ns": {
        "features": 2304,
        "base_net": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b7_ns": {
        "features": 2560,
        "base_net": partial(tf_efficientnet_b7_ns, pretrained=True, drop_path_rate=0.2)
    }
}


class SpatialPyramidPool2D(nn.Module):
    def __init__(self, out_side):
        super(SpatialPyramidPool2D, self).__init__()
        self.out_side = out_side

    def forward(self, x):
        batch_size, c, h, w = x.size()
        out = None
        for n in self.out_side:
            max_pool = AdaptiveMaxPool2d(output_size=(n, n))
            y = max_pool(x)
            if out is None:
                out = y.view(y.size()[0], -1)
            else:
                out = torch.cat((out, y.view(y.size()[0], -1)), 1)
        return out



class DeepFakeSppClassifier(nn.Module):
    def __init__(self, encoder, pool_size=(1, 2, 6)):
        super().__init__()
        self.encoder = encoder_params[encoder]["base_net"]()
        #print('encoder')
        self.spp = SpatialPyramidPool2D(out_side=pool_size)
        #print('spp')
        num_features = encoder_params[encoder]["features"] * (pool_size[0] ** 2 + pool_size[1] ** 2 + pool_size[2] ** 2)
        self.spp_fc = nn.Linear(num_features, 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.spp(x)
        x = self.spp_fc(x)
        return x


class DeepFakeClassifier(nn.Module):
    def __init__(self, encoder, dropout_rate=0.0):
        super().__init__()
        self.encoder = encoder_params[encoder]["base_net"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def build_model(encoder, weights, no_spp=False):
    if no_spp:
        print("Building DeepFakeClassifier...")
        model = DeepFakeClassifier(encoder)
    else:
        print("Building DeepFakeSppClassifier...")
        model = DeepFakeSppClassifier(encoder)
    #model = model.cuda()
    
    epoch = 0
    bce_best = 100
    if os.path.isfile(weights):
        print(f"Loading checkpoint '{weights}'")
        checkpoint = torch.load(weights, map_location='cpu')
        state_dict = checkpoint['state_dict']
        state_dict = {k[7:]: w for k, w in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        epoch = checkpoint['epoch']
        bce_best = checkpoint.get('bce_best', 0)
        print(f"Loaded checkpoint '{weights}' with epoch: {epoch}, bce_best: {bce_best})")
    else:
        print(f"[ERROR] no checkpoint found at '{weights}'")

    # model = DataParallel(model).cuda()
    # summary(model,(3,224,224))
    return model, epoch, bce_best
