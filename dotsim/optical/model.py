"""Residual optical model (Sec. III-B): predicts the change of the tactile image from the change of the simulated
depth and normal images, relative to the undeformed frame."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50

from .data import create_dataloader_from_webdataset


def convert_bn_to_gn(module, num_groups=32):
    """Replace every BatchNorm2d with a GroupNorm (one group when the channels don't divide by ``num_groups``)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            current_num_groups = num_groups if num_channels % num_groups == 0 else 1
            setattr(module, name, nn.GroupNorm(num_groups=current_num_groups, num_channels=num_channels,
                                               eps=child.eps, affine=child.affine))
        else:
            convert_bn_to_gn(child, num_groups=num_groups)


def remove_dropout(model):
    for name, module in model.named_modules():
        if not isinstance(module, nn.Dropout):
            continue
        parts = name.split(".")
        if len(parts) > 1:
            setattr(model.get_submodule(".".join(parts[:-1])), parts[-1], nn.Identity())
        else:
            setattr(model, name, nn.Identity())


class DeepLabModel(pl.LightningModule):
    """DeepLabV3-ResNet50 (ImageNet backbone) taking depth + normal (6 channels) and predicting RGB.

    With ``cfg.use_diff`` the inputs and the target are differences to the first frame of the sequence, and the
    prediction is added to that frame's real image.
    """

    def __init__(self, cfg):
        super().__init__()
        backbone = ResNet50_Weights.IMAGENET1K_V1 if cfg.get("pretrained_backbone", True) else None
        self.model = deeplabv3_resnet50(weights=None, weights_backbone=backbone,
                                        num_classes=256, aux_loss=None)
        self.model.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.cfg = cfg
        self.use_diff = cfg.use_diff
        self.refine_conv = nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0)
        convert_bn_to_gn(self.model)
        remove_dropout(self.model)

    def train_dataloader(self):
        return create_dataloader_from_webdataset(
            self.cfg.data_root,
            split="train",
            batch_size=self.cfg.get("batch_size", 8),
            num_workers=self.cfg.get("num_workers", 8),
            shuffle_buffer=self.cfg.get("shuffle_buffer", 100),
            repeat=1,
        )

    def val_dataloader(self):
        return create_dataloader_from_webdataset(
            self.cfg.data_root,
            split="val",
            batch_size=1,
            num_workers=self.cfg.get("val_num_workers", 4),
            shuffle_buffer=0,
        )

    def forward(self, imgs):
        pred = self.model(imgs)["out"]
        return self.refine_conv(pred)

    def _step(self, batch):
        input_depth, gt, input_normal, ref_rgb, ref_depth, ref_normal = batch
        raw_depth = input_depth + ref_depth
        raw_normal = input_normal + ref_normal
        raw_gt = gt + ref_rgb
        if self.use_diff:
            pred = self(torch.cat([input_depth, input_normal], dim=1))
            loss = F.mse_loss(pred, gt)
        else:
            pred = self(torch.cat([raw_depth, raw_normal], dim=1))
            loss = F.mse_loss(pred, raw_gt)
        return loss, pred, raw_depth, raw_normal, raw_gt, ref_rgb

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)[0]
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, raw_depth, raw_normal, raw_gt, ref_rgb = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)
        if isinstance(self.logger, WandbLogger):
            if self.use_diff:
                pred = pred + ref_rgb
            self.logger.log_image(key="rendered_depth", images=[raw_depth[0]], step=self.current_epoch)
            self.logger.log_image(key="rendered_normal", images=[raw_normal[0]], step=self.current_epoch)
            self.logger.log_image(key="pred", images=[pred[0]], step=self.current_epoch)
            self.logger.log_image(key="gt", images=[raw_gt[0]], step=self.current_epoch)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam([*self.model.parameters(), *self.refine_conv.parameters()],
                                lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
