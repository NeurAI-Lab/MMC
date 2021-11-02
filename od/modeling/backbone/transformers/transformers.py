import torch
import torch.nn as nn
from functools import partial
from od.modeling import registry
from od.modeling.backbone.transformers.attention import Attention
from od.modeling.backbone.transformers.droppath import DropPath
from od.modeling.backbone.transformers.mlp import Mlp
from od.modeling.backbone.transformers.ntuple import to_2tuple
from od.modeling.backbone.transformers.trunc_normal import trunc_normal_
from od.modeling.backbone.transformers.post_backbone_v1 import PostBackboneV1
from od.modeling.backbone.transformers.post_backbone_v2 import PostBackboneV2
from od.modeling.backbone.transformers.post_backbone_v4 import PostBackboneV4


import math



def load(checkpoint,model):
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and k in state_dict and  checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed

    model.load_state_dict(checkpoint_model, strict=False)

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x



class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm,upsample="",postbackbone="v1"):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.postbackbone=postbackbone
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        #v1 normal conv after last layer
        #v2 taken all layers but not implemented yet
        #v3 choose o the on before last from all and apply conv
        #v4 choose the dist+cls token from all layers then apply conv
        if postbackbone=="v1" or postbackbone=="v3" :
            self.after_backbone=PostBackboneV1(int(math.sqrt(self.patch_embed.num_patches)),embed_dim,upsample=upsample,postbackbone=postbackbone)
        elif postbackbone=="v2":
            self.after_backbone=PostBackboneV2(int(math.sqrt(self.patch_embed.num_patches)),embed_dim,upsample=upsample)
        elif postbackbone == "v4":
            self.after_backbone = PostBackboneV4(int(math.sqrt(self.patch_embed.num_patches)),embed_dim,
                                                 upsample=upsample)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        outs=[]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            if self.postbackbone == "v2" or self.postbackbone == "v3":
                outs.append(x[:,1:])

            x = blk(x)

        x = self.norm(x)

        outs.append(x[:,1:])

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        x=self.after_backbone(x)
        return x




class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)
        outs=[]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            if self.postbackbone == "v2" or  self.postbackbone == "v3":
                outs.append(x[:,2:])
            if self.postbackbone == "v4":
                outs.append(x)
            x = blk(x)

        x = self.norm(x)
        if self.postbackbone == "v1" or self.postbackbone == "v2" or self.postbackbone == "v3":
            outs.append(x[:,2:])
        if self.postbackbone == "v4":
            outs.append(x)
        return outs
    def forward(self, x):
        x = self.forward_features(x)

        x = self.after_backbone(x)

        return x


@registry.BACKBONES.register('deit_tiny_patch16_224')
def deit_tiny_patch16_224(cfg,pretrained=False, **kwargs):
    """
    if post_backbone_v1:
    output_channels: (192, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    ( 32, 16, 8, 4, 2)
    """
    model = VisionTransformer(img_size=cfg.INPUT.IMAGE_SIZE,
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),upsample=cfg.MODEL.BACKBONE.UPSAMPLE,postbackbone=cfg.MODEL.BACKBONE.POSTBACKBONE, **kwargs)
    model.default_cfg = cfg
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        load(checkpoint, model)
    return model


@registry.BACKBONES.register('deit_small_patch16_224')
def deit_small_patch16_224(cfg,pretrained=False, **kwargs):
    """
    if post_backbone_v1:
    output_channels: (384, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    ( 32, 16, 8, 4, 2)
    """
    model = VisionTransformer(img_size=cfg.INPUT.IMAGE_SIZE,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),upsample=cfg.MODEL.BACKBONE.UPSAMPLE,postbackbone=cfg.MODEL.BACKBONE.POSTBACKBONE, **kwargs)
    model.default_cfg = cfg
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        load(checkpoint, model)
    return model

@registry.BACKBONES.register('deit_base_patch16_224')
def deit_base_patch16_224(cfg,pretrained=False, **kwargs):
    """
    if post_backbone_v1:
    output_channels: (768, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    ( 32, 16, 8, 4, 2)
    """
    model = VisionTransformer(img_size=cfg.INPUT.IMAGE_SIZE,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),upsample=cfg.MODEL.BACKBONE.UPSAMPLE,postbackbone=cfg.MODEL.BACKBONE.POSTBACKBONE, **kwargs)
    model.default_cfg = cfg
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        load(checkpoint, model)
    return model


@registry.BACKBONES.register('deit_tiny_distilled_patch16_224')
def deit_tiny_distilled_patch16_224(cfg,pretrained=False, **kwargs):
    """
    if post_backbone_v1:
    output_channels: (192, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    ( 32, 16, 8, 4, 2)
    """
    model = DistilledVisionTransformer(img_size=cfg.INPUT.IMAGE_SIZE,
        patch_size=16, in_chans=6 if cfg.KD.CONCAT_INPUT else 3, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),upsample=cfg.MODEL.BACKBONE.UPSAMPLE,postbackbone=cfg.MODEL.BACKBONE.POSTBACKBONE, **kwargs)
    model.default_cfg = cfg
    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
        #     map_location="cpu", check_hash=True
        # )
        # load(checkpoint, model)
        model_dict = model.state_dict()
        pretrained_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
                                                             map_location="cpu", check_hash=True)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict
                           and (model_dict[k].shape == pretrained_dict[k].shape)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@registry.BACKBONES.register('deit_small_distilled_patch16_224')
def deit_small_distilled_patch16_224(cfg,pretrained=False, **kwargs):
    """
    if post_backbone_v1:
    output_channels: (384, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    ( 32, 16, 8, 4, 2)
    """
    model = DistilledVisionTransformer(img_size=cfg.INPUT.IMAGE_SIZE,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),upsample=cfg.MODEL.BACKBONE.UPSAMPLE,postbackbone=cfg.MODEL.BACKBONE.POSTBACKBONE, **kwargs)
    model.default_cfg = cfg
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        load(checkpoint, model)
    return model


@registry.BACKBONES.register('deit_base_distilled_patch16_224')
def deit_base_distilled_patch16_224(cfg,pretrained=False, **kwargs):
    """
    if post_backbone_v1:
    output_channels: (768, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    ( 32, 16, 8, 4, 2)
    """
    model = DistilledVisionTransformer(img_size=cfg.INPUT.IMAGE_SIZE,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),upsample=cfg.MODEL.BACKBONE.UPSAMPLE,postbackbone=cfg.MODEL.BACKBONE.POSTBACKBONE, **kwargs)
    model.default_cfg = cfg
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        load(checkpoint, model)
    return model

@registry.BACKBONES.register('deit_base_patch16_384')
def deit_base_patch16_384(cfg, pretrained=True, **kwargs):
    """
    if post_backbone_v1:
    output_channels: (768, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    ( 32, 16, 8, 4, 2)
    """
    model = VisionTransformer(img_size=cfg.INPUT.IMAGE_SIZE, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),upsample=cfg.MODEL.BACKBONE.UPSAMPLE,postbackbone=cfg.MODEL.BACKBONE.POSTBACKBONE, **kwargs)
    model.default_cfg = cfg
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        load(checkpoint, model)
    return model

@registry.BACKBONES.register('deit_base_distilled_patch16_384')
def deit_base_distilled_patch16_384(cfg,pretrained=False, **kwargs):
    """
    if post_backbone_v1:
    output_channels: (768, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    ( 32, 16, 8, 4, 2)
    """
    model = DistilledVisionTransformer(img_size=cfg.INPUT.IMAGE_SIZE, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),upsample=cfg.MODEL.BACKBONE.UPSAMPLE,postbackbone=cfg.MODEL.BACKBONE.POSTBACKBONE, **kwargs)
    model.default_cfg = cfg
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        load(checkpoint, model)
    return model



