# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, logger=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        if self.cfg.MODEL.STAGE == "stage1":
            features = self.backbone(images.tensors)
            if self.cfg.DATASETS.DIR_LOAD_PRECOMPUTE_DETECTION_BOX and self.training:
                proposals, proposal_losses = self.rpn(images, features, targets[0])
            else:
                proposals, proposal_losses = self.rpn(images, features, targets)
        else:
            with torch.no_grad():
                features = self.backbone(images.tensors)
                if self.cfg.DATASETS.DIR_LOAD_PRECOMPUTE_DETECTION_BOX and self.training:
                    proposals, proposal_losses = self.rpn(images, features, targets[0])
                else:
                    proposals, proposal_losses = self.rpn(images, features, targets)

        x, result, detector_losses = self.roi_heads(features, proposals, targets, logger)


        if self.training:
            losses = {}
            losses.update(detector_losses)
            if not self.cfg.MODEL.RELATION_ON:
                # During the relationship training stage, the rpn_head should be fixed, and no loss.
                losses.update(proposal_losses)
            return losses

        return result
