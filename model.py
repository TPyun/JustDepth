import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib import Grapher, act_layer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, trunc_normal_
from timm.models import register_model

import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 896, 1600),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1',
        'classifier': 'head', 
        **kwargs
    }

default_cfgs = {
    'justdepth': _cfg(
        crop_pct=0.9, input_size=(3, 896, 1600),
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
    ),
}


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            # nn.GroupNorm(8, hidden_features),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            # nn.GroupNorm(8, out_features),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

############################
# 1) BasicBlock (for 2D)
############################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act   = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)
        return out


############################
# 2) BasicBlock1D (for 1D)
############################
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.act   = nn.GELU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)
        return out


############################
# 3) RadarEncoder (1D)
############################
class RadarEncoder(nn.Module):
    """
    1D Conv 기반 레이더 인코더:
      conv1 -> 16 channels
      layer1 -> 32 channels
      layer2 -> 64 channels
      layer3 -> 128 channels
      layer4 -> 256 channels
    """
    def __init__(self, block=BasicBlock1D, layers=[2, 2, 2, 2], in_channels=1):
        super(RadarEncoder, self).__init__()
        self._in_channels = 16  # conv1 출력 채널

        # conv1
        self.conv1 = nn.Conv1d(in_channels, self._in_channels,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm1d(self._in_channels)
        self.act   = nn.GELU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # layer1 ~ layer4
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    @property
    def in_channels(self):
        return getattr(self, "_in_channels", 16)

    @in_channels.setter
    def in_channels(self, value):
        self._in_channels = value

    def forward(self, x):
        """
        x shape: [B, 1, L]  (기본 in_channels=1)
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        f0 = x

        x = self.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        
        # [f0, f1, f2, f3, f4] 에 각각 [16, 32, 64, 128, 256] 채널
        return [f0, f1, f2, f3, f4]


############################
# 4) ImageEncoder (2D)
############################
class ImageEncoder(nn.Module):
    """
    2D Conv 기반 이미지 인코더:
      conv1 -> 16 channels
      layer1 -> 32 channels
      layer2 -> 64 channels
      layer3 -> 128 channels
      layer4 -> 256 channels
    """
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2]):
        super(ImageEncoder, self).__init__()
        self._in_channels = 16  # conv1 출력 채널
        
        # conv1
        self.conv1 = nn.Conv2d(3, self._in_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(self._in_channels)
        self.act   = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # layer1 ~ layer4
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    @property
    def in_channels(self):
        return getattr(self, "_in_channels", 16)

    @in_channels.setter
    def in_channels(self, value):
        self._in_channels = value

    def forward(self, x):
        """
        x shape: [B, 3, H, W]
        """
        f0 = self.conv1(x)
        f0 = self.norm1(f0)
        f0 = self.act(f0)

        x = self.maxpool(f0)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        
        return [f0, f1, f2, f3, f4]

    
class DepthDecoder(nn.Module):
    def __init__(self, skip_channels=[16, 32, 64, 128, 256]):
        super(DepthDecoder, self).__init__()
        self.num_stages = len(skip_channels) - 1

        self.up_convs = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        
        for i in range(self.num_stages):
            in_ch = skip_channels[-1 - i]
            out_ch = skip_channels[-2 - i]
            
            self.up_convs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='reflect'),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                ))
            
            self.fuse_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1, padding_mode='reflect'),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                ))
        self.out_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(skip_channels[0], 1, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GELU(),
        )

    def forward(self, feats):
        d = feats[-1]

        for i in range(self.num_stages):
            d = self.up_convs[i](d)
            skip = feats[-2 - i] 
            d = torch.cat([d, skip], dim=1)
            d = self.fuse_convs[i](d)
        out = self.out_conv(d)
        return out


class ConfidenceDecoder(nn.Module):
    def __init__(self, skip_channels=[256, 128, 64, 32, 16]):
        super(ConfidenceDecoder, self).__init__()
        self.num_stages = len(skip_channels)
        
        self.blocks = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        
        for i in range(self.num_stages):
            if i < self.num_stages - 1:
                in_ch = skip_channels[i]
                out_ch = skip_channels[i+1]
                block = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='reflect'),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                )
            else:
                in_ch = skip_channels[i]
                block = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_ch, 1, kernel_size=3, padding=1, padding_mode='reflect'),
                )
            self.blocks.append(block)
            
            if i > 0:
                if i < self.num_stages - 1:
                    proj = nn.Conv2d(skip_channels[i], skip_channels[i+1], kernel_size=1, bias=False)
                else:
                    proj = nn.Conv2d(skip_channels[i], 1, kernel_size=1, bias=False)
                self.residual_projs.append(proj)

    def forward(self, x):
        up_former = None
        for i, block in enumerate(self.blocks):
            out = block(x)
            if i > 0:
                residual = self.residual_projs[i-1](up_former)
                residual = F.interpolate(residual, size=out.shape[2:], mode='bilinear', align_corners=False)
                out = out + residual
            up_former = out
            x = out
        return out
    

class FusionBlock(nn.Module):
    def __init__(self, embed_dim, height, drop_path_rate=0.1, pos_drop_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.height = height

        # fuse linear: concat(radar, image) → embed_dim
        self.fuse_linear = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Pre-Norm for attention
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Pre-Norm for FFN
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # final Conv-FFN
        self.ffn_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(embed_dim * 4),
            nn.GELU(),
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # stochastic depth (DropPath)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, image, radar):
        """
        image: (B, D, H, W)
        radar: (B, D, W)
        returns: fused feature (B, D, H, W)
        """
        B, D, H, W = image.shape

        # 1) image → (N, H, D), N = W * B
        img = image.permute(3, 0, 2, 1).reshape(W * B, H, D)

        # 2) radar → (N, H, D)
        rd = radar.permute(0, 2, 1).unsqueeze(2)                # (B, W, 1, D)
        rd = rd.permute(1, 0, 2, 3).reshape(W * B, 1, D)
        rd = rd.expand(-1, H, -1)                               # (N, H, D)

        # 3) fuse: concat → linear
        fuse = torch.cat([rd, img], dim=2)                     # (N, H, 2D)
        fuse = self.fuse_linear(fuse)                          # (N, H, D)

        # 5) Pre-Norm + Self-Attention → residual
        fuse = self.norm_attn(fuse)
        attn_out, _ = self.attn(fuse,
                                fuse,
                                fuse)
        x = fuse + self.drop_path(attn_out)

        # 6) Pre-Norm + FFN(Linear) → residual
        ffn_out = self.ffn_linear(self.norm_ffn(x))
        x = x + self.drop_path(ffn_out)

        # 7) (N, H, D) → (B, D, H, W) → Conv-FFN + residual
        x = x.view(W, B, H, D).permute(1, 3, 2, 0)
        x = self.ffn_conv(x) + x

        return x
    

class JustDepth(torch.nn.Module):
    def __init__(self, opt):
        super(JustDepth, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        
        self.f_width = 50
        self.f_height = 28
        
        self.radar_encoder = RadarEncoder()
        self.image_encoder = ImageEncoder()
        
        self.fusion_block = FusionBlock(embed_dim=channels, height=self.f_height, drop_path_rate=drop_path, pos_drop_rate=0.1)
        
        if self.n_blocks == 0:
            self.graph_backbone = nn.Identity()
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
            num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
            max_dilation = self.f_height * self.f_width // max(num_knn)
            
            if opt.use_dilation:
                self.graph_backbone = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                    bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                        FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                        ) for i in range(self.n_blocks)])
            else:
                self.graph_backbone = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                    bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                        FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                        ) for i in range(self.n_blocks)])

        self.confidence_decoder = ConfidenceDecoder()
        self.depth_decoder = DepthDecoder()
        
        self.model_init()
        
    def model_init(self):
        for m in self.modules():
            # Conv / Linear 계열
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
                    
    def eval(self):
        super().eval()
        self.confidence_decoder = torch.nn.Identity()
        return self

    def forward(self, images, radar, get_confidence=True):
        # Inage Encoder
        image_features = self.image_encoder(images)
        image_latent = image_features[-1]
        
        # Radar Encoder
        radar_features = self.radar_encoder(radar.squeeze(2))
        radar_latent = radar_features[-1]
        
        # Fusion Block
        fused_feature = self.fusion_block(image_latent, radar_latent)
        
        # Confidence Decoder
        if get_confidence:
            confidence_map = self.confidence_decoder(fused_feature)

        x = fused_feature + image_latent
        
        # GCN
        for i in range(self.n_blocks):
            x = self.graph_backbone[i](x)

        # Depth Decoder
        features = image_features[:-1] + [x]
        depth_map = self.depth_decoder(features)
        
        if get_confidence:
            return depth_map, confidence_map, image_features[0], fused_feature
        
        return depth_map


@register_model
def justdepth(**kwargs):
    class OptInit:
        def __init__(self, drop_path_rate=0.1, drop_rate=0.1, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 8 # number of basic blocks in the graph_backbone
            self.n_filters = 256 # number of channels of deep features
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate


    opt = OptInit(**kwargs)
    model = JustDepth(opt)
    model.default_cfg = default_cfgs['justdepth']
    return model
