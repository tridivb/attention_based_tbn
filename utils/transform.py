import torchvision
import cv2
import numpy as np
import numbers
import math
import torch


class RandomCrop(object):
    def __init__(self, size):
        assert isinstance(size, (int, tuple))

        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, frames):

        h, w = frames.shape[1], frames.shape[2]
        th, tw = self.size

        out_images = np.zeros(
            frames.shape[0], self.size[0], self.size[1], frames.shape[3]
        )

        x1 = np.random.randint(0, w - tw)
        y1 = np.random.randint(0, h - th)

        out_imgs = []
        for frame_no in range(frames.shape[0]):
            assert frames.shape[2] == w and frames.shape[1] == h
            if w == tw and h == th:
                out_images.extend(frames[frame_no])[None, ...]
            else:
                out_images.extend(
                    frames[frame_no, x1 : x1 + tw, y1 : y1 + th, :][None, ...]
                )

        out_images = np.stack(out_images, axis=0)

        return out_images


class CenterCrop(object):
    def __init__(self, size):
        assert isinstance(size, (int, tuple))

        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, img_stack):
        
        h, w = self.size
        out_images = []

        for img_no in range(img_stack.shape[0]):
            img = img_stack[img_no]
            x1 = (img.shape[1] - w) // 2
            y1 = (img.shape[0] - h) // 2
            img = img[y1:y1+h, x1:x1+w, :]
            out_images.extend(img[None, ...])

        out_images = np.stack(out_images, axis=0)
        
        return out_images


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(
                        ret[i]
                    )  # invert flow pixel values when flipping
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = (
            crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        )

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(
            False, image_w, image_h, crop_w, crop_h
        )
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == "L" and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):
    def __init__(
        self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True
    ):
        self.scales = scales if scales is not None else [1, 0.875, 0.75, 0.66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = (
            input_size if not isinstance(input_size, int) else [input_size, input_size]
        )
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [
            img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
            for img in img_group
        ]
        ret_img_group = [
            img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
            for img in crop_img_group
        ]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(x - self.input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            self.input_size[0] if abs(x - self.input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1]
            )

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h
        )
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) / 4
        h_step = (image_h - crop_h) / 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class RandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):

        assert isinstance(size, (int, tuple))

        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size
            
        self.interpolation = interpolation

    def __call__(self, img_stack):

        assert (isinstance(img_stack), np.ndarray)
        

        for attempt in range(10):
            area = img_stack.shape[1] * img_stack.shape[2]
            target_area = np.random.uniform(0.08, 1.0) * area
            aspect_ratio = np.random.uniform(3.0 / 4, 4.0 / 3)

            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))

            if np.random.random() < 0.5:
                w, h = h, w

            if w <= img_stack.shape[2] and h <= img_stack.shape[1]:
                x1 = np.random.randint(0, img_stack.shape[2] - w)
                y1 = np.random.randint(0, img_stack.shape[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_images = []
            for img_no in range(img_stack.shape[0]):
                img = img_stack[img_no, x1 : x1 + w, y1 : y1 + h, :]
                assert img.shape[:-1] == (w, h)
                out_images.extend(
                    cv2.resize(
                        img, (self.size, self.size), interpolation=self.interpolation
                    )
                )
            return out_images
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = RandomCrop(self.size)
            return crop(scale(img_stack))


class ToTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __call__(self, images):
        assert isinstance(images, np.ndarray)
        assert len(images.shape) == 4

        out_images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        out_images /= 255

        return out_images


class IdentityTransform(object):
    def __call__(self, data):
        return data
