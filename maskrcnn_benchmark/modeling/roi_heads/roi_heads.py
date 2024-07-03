# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import os
from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .attribute_head.attribute_head import build_roi_attribute_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .relation_head.relation_head import build_roi_relation_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None, logger=None):
        losses = {}
        if self.cfg.DATASETS.DIR_LOAD_PRECOMPUTE_DETECTION_BOX and self.training:
            tmp_target = targets[0]
        else:
            tmp_target = targets
        if self.cfg.MODEL.STAGE == "stage1":
            x, detections, loss_box = self.box(features, proposals, tmp_target)
        else:
            with torch.no_grad():
                x, detections, loss_box = self.box(features, proposals, tmp_target)
        if self.cfg.DATASETS.INFER_BBOX:
            dst_path = os.path.join(self.cfg.OUTPUT_DIR, "bbox")
            os.makedirs(dst_path, exist_ok=True)
            data = dict()
            name = None
            for det, tgt in zip(detections, targets):
                w, h = tgt.get_field("src_w"), tgt.get_field("src_h")
                det_resize = det.resize((w, h))
                item = {
                    'size': (w, h),
                    'bbox': det_resize.bbox.cpu(),
                    'pred_scores': det_resize.get_field('pred_scores').cpu(),
                    'pred_labels': det_resize.get_field('pred_labels').cpu(),
                    'predict_logits': det_resize.get_field('predict_logits').cpu(),
                    'labels': det_resize.get_field('labels').cpu(),
                }
                data[tgt.get_field("fp_name")] = item
                name = tgt.get_field("fp_name")
            torch.save(data, os.path.join(dst_path, name+'.pkl'))
        if self.cfg.DATASETS.DIR_LOAD_PRECOMPUTE_DETECTION_BOX and self.training:
            detections = targets[1]
            targets = targets[0]
        if not self.cfg.MODEL.RELATION_ON:
            # During the relationship training stage, the bbox_proposal_network should be fixed, and no loss. 
            losses.update(loss_box)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            # Attribute head don't have a separate feature extractor
            z, detections, loss_attribute = self.attribute(features, detections, targets)
            losses.update(loss_attribute)

        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)

        if self.cfg.MODEL.RELATION_ON:
            # it may be not safe to share features due to post processing
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            if self.training:
                features, detections, targets = self.mask_non_proposal(features, detections, targets)
            x, detections, loss_relation = self.relation(features, detections, targets, logger)
            losses.update(loss_relation)

        return x, detections, losses

    @staticmethod
    def mask_non_proposal(features, detections, targets):
        feat, det, tgt = [], [], []
        mask = torch.ones(len(detections), dtype=torch.bool, device=features[0].device)
        for i, (d, t) in enumerate(zip(detections, targets)):
            if len(d) != 0:
                det.append(d)
                tgt.append(t)
            else:
                mask[i] = False
        feat = [x[mask] for x in features]
        return feat, det, tgt


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))
    if cfg.MODEL.RELATION_ON:
        roi_heads.append(("relation", build_roi_relation_head(cfg, in_channels)))
    if cfg.MODEL.ATTRIBUTE_ON:
        roi_heads.append(("attribute", build_roi_attribute_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
