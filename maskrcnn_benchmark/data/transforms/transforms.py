# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import mmcv
import random
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import ImageEnhance, ImageOps, Image

from maskrcnn_benchmark.structures.bounding_box import BoxList


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        if isinstance(target, BoxList):
            target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target


# --new add augments--
PARAMETER_MAX = 10


class RandAug(object):
    """refer to https://github.com/microsoft/SoftTeacher"""
    def __init__(
        self,
        prob: float = 0.25,
        magnitude: int = 10,
        random_magnitude: bool = True,
        magnitude_limit: int = 10,
    ):
        assert 0 <= prob <= 1, f"probability should be in (0,1) but get {prob}"
        assert (
            magnitude <= PARAMETER_MAX
        ), f"magnitude should be small than max value {PARAMETER_MAX} but get {magnitude}"

        self.prob = prob
        self.magnitude = magnitude
        self.magnitude_limit = magnitude_limit
        self.random_magnitude = random_magnitude
        self.buffer = None
        self.record = False

    def __call__(self, image, target):
        if np.random.random() < self.prob:
            magnitude = self.magnitude
            if self.random_magnitude:
                magnitude = np.random.randint(1, magnitude)
            image, target = self.apply(image, target, magnitude)
        # clear buffer
        return image, target

    def apply(self, image, target, magnitude: int = None):
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}(prob={self.prob},magnitude={self.magnitude}," \
               f"max_magnitude={self.magnitude_limit},random_magnitude={self.random_magnitude})"

    def get_aug_info(self, **kwargs):
        aug_info = dict(type=self.__class__.__name__)
        aug_info.update(
            dict(
                prob=1.0,
                random_magnitude=False,
                record=False,
                magnitude=self.magnitude,
            )
        )
        aug_info.update(kwargs)
        return aug_info

    def enable_record(self, mode: bool = True):
        self.record = mode


def int_parameter(level, maxval, max_level=None):
    if max_level is None:
        max_level = PARAMETER_MAX
    return int(level * maxval / max_level)


def float_parameter(level, maxval, max_level=None):
    if max_level is None:
        max_level = PARAMETER_MAX
    return float(level) * maxval / max_level


class RandSolarize(RandAug):
    def apply(self, image, target, magnitude: int = None):
        ratio = min(int_parameter(magnitude, 256, self.magnitude_limit), 255)
        rand_solarize = torchvision.transforms.RandomSolarize(threshold=ratio, p=self.prob)
        return rand_solarize(image), target


def _enhancer_impl(enhancer):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of
    PIL."""

    def impl(pil_img, level, max_level=None):
        # v = float_parameter(level, 1.8, max_level) + 0.1 # going to 0 just destroys it
        # multi-aug need to change as the paper do
        v = float_parameter(level, 1.0, max_level)
        return enhancer(pil_img).enhance(v)

    return impl


class RandEnhance(RandAug):
    op = None

    def apply(self, image, target, magnitude: int = None):
        img = _enhancer_impl(self.op)(
            image, magnitude, self.magnitude_limit
        )
        return img, target


class RandContrast(RandEnhance):
    op = ImageEnhance.Contrast


class RandBrightness(RandEnhance):
    op = ImageEnhance.Brightness


class RandSharpness(RandEnhance):
    op = ImageEnhance.Sharpness


class GeometricAugmentation(object):
    def __init__(
        self,
        img_fill_val=125,
        min_size=0,
        prob: float = 0.25,
        random_magnitude: bool = True,
    ):
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, "img_fill_val as tuple must have 3 elements."
            img_fill_val = tuple([float(val) for val in img_fill_val])
        assert np.all(
            [0 <= val <= 255 for val in img_fill_val]
        ), "all elements of img_fill_val should between range [0,255]."
        self.img_fill_val = img_fill_val
        self.min_size = min_size
        self.prob = prob
        self.random_magnitude = random_magnitude

    def __call__(self, image, target):
        if not target.has_field("aug_mask"):
            aug_mask = torch.ones(len(target), dtype=torch.bool, device=target.bbox.device)
            target.add_field("aug_mask", aug_mask)
        if np.random.random() < self.prob:
            magnitude: dict = self.get_magnitude(image, target)
            image, target = self.apply(image, target, **magnitude)
            target = self._filter_invalid(image, target, min_size=self.min_size)
        return image, target

    def get_magnitude(self, image, target) -> dict:
        raise NotImplementedError()

    def apply(self, image, target, **kwargs):
        raise NotImplementedError()

    def _filter_invalid(self, image, target, min_size=0):
        """Filter bboxes and masks too small or translated out of image."""
        if min_size is None:
            return target
        assert target.mode == 'xyxy'
        bbox = target.bbox.numpy()
        min_x, min_y, max_x, max_y = np.split(bbox, bbox.shape[-1], axis=-1)
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        valid_inds = (bbox_w > min_size) & (bbox_h > min_size)
        # --change--
        """
        valid_inds = torch.from_numpy(np.nonzero(valid_inds)[0])
        target = target[valid_inds]
        """
        aug_mask = torch.tensor(valid_inds, dtype=torch.bool).squeeze(-1)
        if target.has_field("aug_mask"):
            src_aug_mask = target.get_field("aug_mask")
            aug_mask = src_aug_mask & aug_mask
        target.add_field("aug_mask", aug_mask)
        valid_inds = aug_mask.numpy()
        min_x[~valid_inds] = 0.
        min_y[~valid_inds] = 0.
        max_x[~valid_inds] = 1.
        max_y[~valid_inds] = 1.
        bbox = np.concatenate([min_x, min_y, max_x, max_y], axis=-1).astype(bbox.dtype)
        target.bbox = torch.tensor(bbox, dtype=target.bbox.dtype, device=target.bbox.device)
        return target

    def __repr__(self):
        return f"""{self.__class__.__name__}(
        img_fill_val={self.img_fill_val},
        min_size={self.magnitude},
        prob: float = {self.prob},
        random_magnitude: bool = {self.random_magnitude},
        )"""


class RandTranslate(GeometricAugmentation):
    def __init__(self, x=None, y=None, **kwargs):
        super().__init__(**kwargs)
        # default: x = (-0.1, 0.1) or y = (-0.1, 0.1)
        self.x = x
        self.y = y
        if self.x is None and self.y is None:
            self.prob = 0.0

    def get_magnitude(self, image, target):
        magnitude = {}
        if self.random_magnitude:
            if isinstance(self.x, (list, tuple)):
                assert len(self.x) == 2
                x = np.random.random() * (self.x[1] - self.x[0]) + self.x[0]
                magnitude["x"] = x
            if isinstance(self.y, (list, tuple)):
                assert len(self.y) == 2
                y = np.random.random() * (self.y[1] - self.y[0]) + self.y[0]
                magnitude["y"] = y
        else:
            if self.x is not None:
                assert isinstance(self.x, (int, float))
                magnitude["x"] = self.x
            if self.y is not None:
                assert isinstance(self.y, (int, float))
                magnitude["y"] = self.y
        return magnitude

    def apply(self, image, target, x=None, y=None):
        # ratio to pixel
        h, w, c = np.array(image).shape
        if x is not None:
            x = w * x
        if y is not None:
            y = h * y
        if x is not None:
            # translate horizontally
            image, target = self._translate(image, target, x)
        if y is not None:
            # translate veritically
            image, target = self._translate(image, target, y, direction="vertical")
        return image, target

    def _translate(self, image, target, offset, direction="horizontal"):
        image = self._translate_img(image, offset, direction=direction)
        target = self._translate_bboxes(image, target, offset, direction=direction)
        return image, target

    def _translate_img(self, image, offset, direction="horizontal"):
        convert_tag = False
        if not isinstance(image, np.ndarray):
            convert_tag = True
            image = np.array(image)
        image = image.copy()
        img = mmcv.imtranslate(
                image, offset, direction, self.img_fill_val
            ).astype(image.dtype)
        if convert_tag:
            img = Image.fromarray(img)
        return img

    def _translate_bboxes(self, image, target, offset, direction="horizontal"):
        """Shift bboxes horizontally or vertically, according to offset."""
        h, w, c = np.array(image).shape
        assert target.mode == 'xyxy'
        min_x, min_y, max_x, max_y = target.bbox.split(1, dim=-1)
        if direction == "horizontal":
            min_x = torch.maximum(torch.zeros_like(min_x), min_x + offset)
            max_x = torch.minimum(torch.zeros_like(max_x).fill_(w), max_x + offset)
        elif direction == "vertical":
            min_y = torch.maximum(torch.zeros_like(min_y), min_y + offset)
            max_y = torch.minimum(torch.zeros_like(max_y).fill_(h), max_y + offset)
        target.bbox = torch.cat((min_x, min_y, max_x, max_y), dim=-1)
        # the boxes translated outside of image will be filtered along with
        # the corresponding masks, by invoking ``_filter_invalid``.
        return target

    def __repr__(self):
        repr_str = super().__repr__()
        return ("\n").join(
            repr_str.split("\n")[:-1]
            + [f"x={self.x}", f"y={self.y}"]
            + repr_str.split("\n")[-1:]
        )


class RandRotate(GeometricAugmentation):
    def __init__(self, angle=None, center=None, scale=1, **kwargs):
        super().__init__(**kwargs)
        # default: angle = (-30, 30)
        self.angle = angle
        self.center = center
        self.scale = scale
        if self.angle is None:
            self.prob = 0.0

    def get_magnitude(self, image, target):
        magnitude = {}
        if self.random_magnitude:
            if isinstance(self.angle, (list, tuple)):
                assert len(self.angle) == 2
                angle = (
                    np.random.random() * (self.angle[1] - self.angle[0]) + self.angle[0]
                )
                magnitude["angle"] = angle
        else:
            if self.angle is not None:
                assert isinstance(self.angle, (int, float))
                magnitude["angle"] = self.angle
        return magnitude

    def apply(self, image, target, angle: float = None):
        h, w, _ = np.array(image).shape
        center = self.center
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        image = self._rotate_img(image, angle, center, self.scale)
        rotate_matrix = cv2.getRotationMatrix2D(center, -angle, self.scale)
        target = self._rotate_bboxes(image, target, rotate_matrix)
        return image, target

    def _rotate_img(self, image, angle, center=None, scale=1.0):
        """Rotate the image.

        Args:
            results (dict): Result dict from loading pipeline.
            angle (float): Rotation angle in degrees, positive values
                mean clockwise rotation. Same in ``mmcv.imrotate``.
            center (tuple[float], optional): Center point (w, h) of the
                rotation. Same in ``mmcv.imrotate``.
            scale (int | float): Isotropic scale factor. Same in
                ``mmcv.imrotate``.
        """
        convert_tag = False
        if not isinstance(image, np.ndarray):
            convert_tag = True
            image = np.array(image)
        img_rotated = mmcv.imrotate(
            image, angle, center, scale, border_value=self.img_fill_val
        ).astype(image.dtype)
        if convert_tag:
            img_rotated = Image.fromarray(img_rotated)
        return img_rotated

    def _rotate_bboxes(self, image, target, rotate_matrix):
        """Rotate the bboxes."""
        h, w, c = np.array(image).shape
        assert target.mode == 'xyxy'
        bbox = target.bbox.numpy()
        min_x, min_y, max_x, max_y = np.split(bbox, bbox.shape[-1], axis=-1)
        coordinates = np.stack(
            [[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]]
        )  # [4, 2, nb_bbox, 1]
        # pad 1 to convert from format [x, y] to homogeneous
        # coordinates format [x, y, 1]
        coordinates = np.concatenate(
            (
                coordinates,
                np.ones((4, 1, coordinates.shape[2], 1), coordinates.dtype),
            ),
            axis=1,
        )  # [4, 3, nb_bbox, 1]
        coordinates = coordinates.transpose((2, 0, 1, 3))  # [nb_bbox, 4, 3, 1]
        rotated_coords = np.matmul(rotate_matrix, coordinates)  # [nb_bbox, 4, 2, 1]
        rotated_coords = rotated_coords[..., 0]  # [nb_bbox, 4, 2]
        min_x, min_y = (
            np.min(rotated_coords[:, :, 0], axis=1),
            np.min(rotated_coords[:, :, 1], axis=1),
        )
        max_x, max_y = (
            np.max(rotated_coords[:, :, 0], axis=1),
            np.max(rotated_coords[:, :, 1], axis=1),
        )
        min_x, min_y = (
            np.clip(min_x, a_min=0, a_max=w),
            np.clip(min_y, a_min=0, a_max=h),
        )
        max_x, max_y = (
            np.clip(max_x, a_min=min_x, a_max=w),
            np.clip(max_y, a_min=min_y, a_max=h),
        )
        rotate_bbox = np.stack([min_x, min_y, max_x, max_y], axis=-1).astype(bbox.dtype)
        rotate_bbox = torch.tensor(rotate_bbox, dtype=target.bbox.dtype, device=target.bbox.device)
        target.bbox = rotate_bbox
        return target

    def __repr__(self):
        repr_str = super().__repr__()
        return ("\n").join(
            repr_str.split("\n")[:-1]
            + [f"angle={self.angle}", f"center={self.center}", f"scale={self.scale}"]
            + repr_str.split("\n")[-1:]
        )


class RandShear(GeometricAugmentation):
    def __init__(self, x=None, y=None, interpolation="bilinear", **kwargs):
        super().__init__(**kwargs)
        # default: x = (-30, 30) or y = (-30, 30)
        self.x = x
        self.y = y
        self.interpolation = interpolation
        if self.x is None and self.y is None:
            self.prob = 0.0

    def get_magnitude(self, image, target):
        magnitude = {}
        if self.random_magnitude:
            if isinstance(self.x, (list, tuple)):
                assert len(self.x) == 2
                x = np.random.random() * (self.x[1] - self.x[0]) + self.x[0]
                magnitude["x"] = x
            if isinstance(self.y, (list, tuple)):
                assert len(self.y) == 2
                y = np.random.random() * (self.y[1] - self.y[0]) + self.y[0]
                magnitude["y"] = y
        else:
            if self.x is not None:
                assert isinstance(self.x, (int, float))
                magnitude["x"] = self.x
            if self.y is not None:
                assert isinstance(self.y, (int, float))
                magnitude["y"] = self.y
        return magnitude

    def apply(self, image, target, x=None, y=None):
        if x is not None:
            # translate horizontally
            image, target = self._shear(image, target, np.tanh(-x * np.pi / 180))
        if y is not None:
            # translate veritically
            image, target = self._shear(image, target, np.tanh(y * np.pi / 180), direction="vertical")
        return image, target

    def _shear(self, image, target, magnitude, direction="horizontal"):
        image = self._shear_img(image, magnitude, direction, interpolation=self.interpolation)
        target = self._shear_bboxes(image, target, magnitude, direction=direction)
        # fill_val defaultly 0 for BitmapMasks and None for PolygonMasks.
        return image, target

    def _shear_img(
        self, image, magnitude, direction="horizontal", interpolation="bilinear"
    ):
        """Shear the image.

        Args:
            results (dict): Result dict from loading pipeline.
            magnitude (int | float): The magnitude used for shear.
            direction (str): The direction for shear, either "horizontal"
                or "vertical".
            interpolation (str): Same as in :func:`mmcv.imshear`.
        """
        convert_tag = False
        if not isinstance(image, np.ndarray):
            convert_tag = True
            image = np.array(image)
        img_sheared = mmcv.imshear(
            image,
            magnitude,
            direction,
            border_value=self.img_fill_val,
            interpolation=interpolation,
        ).astype(image.dtype)
        if convert_tag:
            img_sheared = Image.fromarray(img_sheared)
        return img_sheared

    def _shear_bboxes(self, image, target, magnitude, direction="horizontal"):
        """Shear the bboxes."""
        h, w, c = np.array(image).shape
        if direction == "horizontal":
            shear_matrix = np.stack([[1, magnitude], [0, 1]]).astype(
                np.float32
            )  # [2, 2]
        else:
            shear_matrix = np.stack([[1, 0], [magnitude, 1]]).astype(np.float32)

        assert target.mode == 'xyxy'
        bbox = target.bbox.numpy()
        min_x, min_y, max_x, max_y = np.split(bbox, bbox.shape[-1], axis=-1)
        coordinates = np.stack(
            [[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]]
        )  # [4, 2, nb_box, 1]
        coordinates = (
            coordinates[..., 0].transpose((2, 1, 0)).astype(np.float32)
        )  # [nb_box, 2, 4]
        new_coords = np.matmul(
            shear_matrix[None, :, :], coordinates
        )  # [nb_box, 2, 4]
        min_x = np.min(new_coords[:, 0, :], axis=-1)
        min_y = np.min(new_coords[:, 1, :], axis=-1)
        max_x = np.max(new_coords[:, 0, :], axis=-1)
        max_y = np.max(new_coords[:, 1, :], axis=-1)
        min_x = np.clip(min_x, a_min=0, a_max=w)
        min_y = np.clip(min_y, a_min=0, a_max=h)
        max_x = np.clip(max_x, a_min=min_x, a_max=w)
        max_y = np.clip(max_y, a_min=min_y, a_max=h)

        shear_bbox = np.stack([min_x, min_y, max_x, max_y], axis=-1).astype(bbox.dtype)
        shear_bbox = torch.tensor(shear_bbox, dtype=target.bbox.dtype, device=target.bbox.device)
        target.bbox = shear_bbox
        return target

    def __repr__(self):
        repr_str = super().__repr__()
        return ("\n").join(
            repr_str.split("\n")[:-1]
            + [f"x_magnitude={self.x}", f"y_magnitude={self.y}"]
            + repr_str.split("\n")[-1:]
        )


class RandErase(GeometricAugmentation):
    def __init__(
        self,
        n_iterations=None,
        size=None,
        squared: bool = True,
        patches=None,
        **kwargs,
    ):
        kwargs.update(min_size=None)
        super().__init__(**kwargs)
        # default: n_iter = (1, 5), size = [0, 0.2], squared = True
        self.n_iterations = n_iterations
        self.size = size
        self.squared = squared
        self.patches = patches

    def get_magnitude(self, image, target):
        magnitude = {}
        if self.random_magnitude:
            n_iterations = self._get_erase_cycle()
            patches = []
            h, w, c = np.array(image).shape
            for i in range(n_iterations):
                # random sample patch size in the image
                ph, pw = self._get_patch_size(h, w)
                # random sample patch left top in the image
                px, py = np.random.randint(0, w - pw), np.random.randint(0, h - ph)
                patches.append([px, py, px + pw, py + ph])
            magnitude["patches"] = patches
        else:
            assert self.patches is not None
            magnitude["patches"] = self.patches

        return magnitude

    def _get_erase_cycle(self):
        if isinstance(self.n_iterations, int):
            n_iterations = self.n_iterations
        else:
            assert (
                isinstance(self.n_iterations, (tuple, list))
                and len(self.n_iterations) == 2
            )
            n_iterations = np.random.randint(*self.n_iterations)
        return n_iterations

    def _get_patch_size(self, h, w):
        if isinstance(self.size, float):
            assert 0 < self.size < 1
            return int(self.size * h), int(self.size * w)
        else:
            assert isinstance(self.size, (tuple, list))
            assert len(self.size) == 2
            assert 0 <= self.size[0] < 1 and 0 <= self.size[1] < 1
            w_ratio = np.random.random() * (self.size[1] - self.size[0]) + self.size[0]
            h_ratio = w_ratio

            if not self.squared:
                h_ratio = (
                    np.random.random() * (self.size[1] - self.size[0]) + self.size[0]
                )
            return int(h_ratio * h), int(w_ratio * w)

    def apply(self, image, target, patches=None):
        if patches is None:
            return image, target
        for patch in patches:
            image = self._erase_image(image, patch, fill_val=self.img_fill_val)
        return image, target

    def _erase_image(self, image, patch, fill_val=128):
        convert_tag = False
        if not isinstance(image, np.ndarray):
            convert_tag = True
            image = np.array(image)
        x1, y1, x2, y2 = patch
        tmp = image.copy()
        tmp[y1:y2, x1:x2, :] = fill_val
        img_erase = tmp
        if convert_tag:
            img_erase = Image.fromarray(img_erase)
        return img_erase

