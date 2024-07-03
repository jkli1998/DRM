# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list
import copy


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, cfg, size_divisible=0, need_to_list=True, mode="None"):
        self.cfg = cfg
        self.mode = mode
        self.size_divisible = size_divisible
        self.need_to_list = need_to_list

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        if self.need_to_list:
            images = to_image_list(transposed_batch[0], self.size_divisible)
            if self.cfg.DATASETS.DIR_LOAD_PRECOMPUTE_DETECTION_BOX and self.mode == 'train':
                view1_targets = [x[0] for x in transposed_batch[1]]
                view2_targets = [x[1] for x in transposed_batch[1]]
                targets = [view1_targets, view2_targets]
            else:
                targets = transposed_batch[1]
            img_ids = transposed_batch[2]
        else:
            images = list(transposed_batch[0])
            if self.cfg.DATASETS.DIR_LOAD_PRECOMPUTE_DETECTION_BOX and self.mode == 'train':
                view1_targets = [x[0] for x in transposed_batch[1]]
                view2_targets = [x[1] for x in transposed_batch[1]]
                targets = [view1_targets, view2_targets]
            else:
                targets = list(transposed_batch[1])
            img_ids = list(transposed_batch[2])
        return images, targets, img_ids


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))


class ContrastiveBatchCollator(object):
    def __init__(self, cfg, size_divisible=0):
        self.size_divisible = size_divisible
        self.cfg = cfg

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        view1_images = [x[0] for x in transposed_batch[0]]
        view2_images = [x[1] for x in transposed_batch[0]]

        view1_targets = [x[0] for x in transposed_batch[1]]
        view2_targets = [x[1] for x in transposed_batch[1]]

        view1_images.extend(view2_images)
        images = to_image_list(view1_images, self.size_divisible)

        view1_targets.extend(view2_targets)
        if self.cfg.DATASETS.DIR_LOAD_PRECOMPUTE_DETECTION_BOX:
            targets = [x[0] for x in view1_targets]
            pre_com = [x[1] for x in view1_targets]
            view1_targets = [targets, pre_com]

        w_img_ids, s_img_ids = [], []
        for x in transposed_batch[2]:
            w_img_ids.append(x)
            s_img_ids.append(x)
        w_img_ids.extend(s_img_ids)
        return images, view1_targets, w_img_ids



