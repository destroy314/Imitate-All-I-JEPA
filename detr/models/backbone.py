# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from typing import List
from ..util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding

from detr.encoders.images_hl_dyh.images_hl_dyh import MultiImageObsEncoder

from jepa.evals.video_classification_frozen.eval import init_model


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class VjepaBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

        # from jepa/configs/evals/vith16_384_k400_16x8x3.yaml
        patch_size = 16
        self.body = init_model(
            crop_size=384,
            device=device,
            pretrained="jepa/cheakpoints/vith16-384.pth.tar",
            model_name="vit_huge",
            patch_size=patch_size,
            tubelet_size=2,
            frames_per_clip=16,
            uniform_power=True,
            checkpoint_key="target_encoder",
            use_SiLU=False,
            tight_SiLU=False,
            use_sdpa=True)
        for param in self.body.parameters():
            param.requires_grad = False
        self.body.eval()
        self.num_channels = 1280

        self.img_size = 384
        self.patch_num = (self.img_size // patch_size)**2

    def forward(self, tensor):
        #tensor: 4,16,3,480,640
        #needed: b,c=3,t=16,h=384,w=384
        tensor = tensor.permute(0, 2, 1, 3, 4)
        # crop 480x640 -> 480x480
        start = (640 - 480) // 2
        tensor = tensor[..., start:start + 480]
        # reshape 480x480 -> 384x384
        batch_size, channels, time_steps, height, width = tensor.shape
        tensor = tensor.reshape(batch_size * time_steps, channels, height, width)
        tensor = F.interpolate(tensor, size=(self.img_size, self.img_size), mode='bilinear')
        tensor = tensor.reshape(batch_size, channels, time_steps, self.img_size, self.img_size)
        # 4,4608,1280
        xs = self.body(tensor)
        xs = F.layer_norm(xs, (xs.size(-1),))  # normalize over feature-dim
        # 4,4608,1280 -> 4,1280,4608,1 变成4维来让transformer连接addition_input detr/models/transformer.py#L51
        xs = xs.permute(0, 2, 1)
        xs = xs.unsqueeze(-1)
        return {'0': xs}

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class BackboneYHD(BackboneBase):
    """YHD backbone with frozen BatchNorm."""
    def __init__(self, cfg, return_interm_layers):

        import hydra

        backbone: MultiImageObsEncoder = hydra.utils.instantiate(cfg.encoder)
        num_channels = 512
        train_backbone = False
        return_interm_layers = True
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    # TODO: build_optimizer function will use lr_backbone
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    if args.backbone == 'vjepa':
        backbone = VjepaBackbone()
    else:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

def build_yhd_backbone(config, args):
    position_embedding = build_position_encoding(args)
    return_interm_layers = args.masks
    backbone = BackboneYHD(config, return_interm_layers)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
