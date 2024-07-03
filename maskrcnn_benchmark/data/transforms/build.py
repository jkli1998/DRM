# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T

def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.TRAIN_AUG.MIN_SIZE_TRAIN
        max_size = cfg.TRAIN_AUG.MAX_SIZE_TRAIN
        base_strong = [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(cfg.TRAIN_AUG.RANDOM_FLIP),
        ]
        train_pipeline_add = [
            T.RandSolarize(
                prob=cfg.TRAIN_AUG.SOLARIZE_PROB,
                magnitude=cfg.TRAIN_AUG.SOLARIZE_MAGN
            ),
            T.RandBrightness(
                prob=cfg.TRAIN_AUG.BRIGHTNESS_PROB,
                magnitude=cfg.TRAIN_AUG.BRIGHTNESS_MAGN,
            ),
            T.RandContrast(
                prob=cfg.TRAIN_AUG.CONTRAST_PROB,
                magnitude=cfg.TRAIN_AUG.CONTRAST_MAGN,
            ),
            T.RandSharpness(
                prob=cfg.TRAIN_AUG.SHARPNESS_PROB,
                magnitude=cfg.TRAIN_AUG.SHARPNESS_MAGN,
            ),
        ]
        pipeline_add = [
            T.RandTranslate(
                prob=cfg.TRAIN_AUG.TRANSLATE_PROB,
                x=cfg.TRAIN_AUG.TRANSLATE_RATIO_X,
            ),
            T.RandTranslate(
                prob=cfg.TRAIN_AUG.TRANSLATE_PROB,
                y=cfg.TRAIN_AUG.TRANSLATE_RATIO_Y,
            ),
            T.RandRotate(
                prob=cfg.TRAIN_AUG.ROTATE_PROB,
                angle=cfg.TRAIN_AUG.ROTATE_ANGLE,
            ),
            T.RandShear(
                prob=cfg.TRAIN_AUG.SHEAR_PROB,
                x=cfg.TRAIN_AUG.SHEAR_RATIO_X,
            ),
            T.RandShear(
                prob=cfg.TRAIN_AUG.SHEAR_PROB,
                y=cfg.TRAIN_AUG.SHEAR_RATIO_Y,
            ),
        ]

        cutout_add = [
            T.RandErase(
                prob=cfg.TRAIN_AUG.CUTOUT_PROB,
                n_iterations=cfg.TRAIN_AUG.CUTOUT_ITER,
                size=cfg.TRAIN_AUG.CUTOUT_SIZE,
                squared=cfg.TRAIN_AUG.CUTOUT_SQUARED,
            ),
        ]

        normalize_add = [
            T.ToTensor(),
            T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN,
                std=cfg.INPUT.PIXEL_STD,
                to_bgr255=cfg.INPUT.TO_BGR255,
            ),
        ]
        pipeline = base_strong + train_pipeline_add + pipeline_add + cutout_add + normalize_add
        transform = T.Compose(pipeline)
    else:
        min_size = cfg.TRAIN_AUG.MIN_SIZE_TEST
        max_size = cfg.TRAIN_AUG.MAX_SIZE_TEST
        transform = T.Compose(
            [
                T.Resize(min_size, max_size),
                T.ToTensor(),
                T.Normalize(
                    mean=cfg.INPUT.PIXEL_MEAN,
                    std=cfg.INPUT.PIXEL_STD,
                    to_bgr255=cfg.INPUT.TO_BGR255,
                ),
            ]
        )
    return transform

