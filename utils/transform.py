import cv2
import numpy as np
import numbers
import math
import torch
import torchvision


class RandomCrop(object):
    def __init__(self, size):
        assert isinstance(size, (int, tuple))

        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, img_list):
        assert isinstance(img_list, list)
        th, tw = self.size
        h, w = img_list[0].shape[0:2]

        x1 = np.random.randint(0, w - tw)
        y1 = np.random.randint(0, h - th)

        out_images = []
        for img in img_list:
            if w == tw and h == th:
                out_images.extend([img])
            else:
                assert img.shape[0] == h and img.shape[1] == w
                out_images.extend([img[y1 : y1 + th, x1 : x1 + tw]])

        return out_images


class CenterCrop(object):
    def __init__(self, size):
        assert isinstance(size, (int, tuple))

        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, img_list):
        assert isinstance(img_list, list)
        h, w = self.size
        out_images = []

        for img in img_list:
            x1 = (img.shape[1] - w) // 2
            y1 = (img.shape[0] - h) // 2
            img = img[y1 : y1 + h, x1 : x1 + w]
            out_images.extend([img])

        return out_images


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, prob=0.5, is_flow=False):
        self.is_flow = is_flow
        self.prob = prob

    def __call__(self, img_list, is_flow=False):

        assert isinstance(img_list, list)
        p = np.random.random()

        if p < self.prob:
            out_images = []
            for img in img_list:
                out_images.extend([np.fliplr(img)])
            return out_images
        else:
            return img_list


class Rescale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR, is_flow=False):
        assert isinstance(size, (int, tuple))
        self.size = size
        self.interpolation = interpolation
        self.is_flow = is_flow

    def __call__(self, img_list):

        assert isinstance(img_list, list)

        h, w = img_list[0].shape[0:2]

        if isinstance(self.size, int):
            if h > w:
                new_h, new_w = self.size * h / w, self.size
            else:
                new_h, new_w = self.size, self.size * w / h
        else:
            assert len(self.size) == 2
            new_h, new_w = self.size

        new_h, new_w = int(new_h), int(new_w)

        out_images = []
        for img in img_list:
            if (new_h, new_w) == img.shape[0:2]:
                out_images.extend([img])
            else:
                assert img.shape[0] == h and img.shape[1] == w
                res_img = cv2.resize(
                    img, (new_w, new_h), interpolation=self.interpolation
                )
                out_images.extend([res_img])
        return out_images


class MultiScaleCrop(object):
    def __init__(
        self,
        input_size,
        scales=None,
        max_distort=1,
        fix_crop=True,
        more_fix_crop=True,
        is_flow=False,
    ):
        self.scales = scales if scales is not None else [1, 0.875, 0.75, 0.66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        assert isinstance(input_size, (int, tuple))
        self.input_size = (
            input_size if isinstance(input_size, tuple) else (input_size, input_size)
        )
        self.is_flow = is_flow
        self.interpolation = cv2.INTER_LINEAR

    def __call__(self, img_list):

        assert isinstance(img_list, list)

        im_size = img_list[0].shape[0:2]

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)

        out_images = []
        for img in img_list:
            img = img[offset_h : offset_h + crop_h, offset_w : offset_w + crop_w]
            out_images.extend([img])

        rescale = Rescale(
            self.input_size, interpolation=self.interpolation, is_flow=self.is_flow
        )
        out_images = rescale(out_images)
        return out_images

    def _sample_crop_size(self, im_size):
        img_h, img_w = im_size[0], im_size[1]

        # find a crop size
        base_size = min(img_w, img_h)
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

        rand_idx = np.random.randint(len(pairs))
        crop_pair = pairs[rand_idx]
        if not self.fix_crop:
            w_offset = np.random.randint(0, img_w - crop_pair[0])
            h_offset = np.random.randint(0, img_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                img_w, img_h, crop_pair[0], crop_pair[1]
            )

        return crop_pair[0], crop_pair[1], int(w_offset), int(h_offset)

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h
        )
        rand_idx = np.random.randint(len(offsets))
        return offsets[rand_idx]

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

    def __call__(self, img_list):

        assert isinstance(img_list, np.ndarray)
        img_h, img_w = img_list[0].shape[0:2]

        for attempt in range(10):

            area = img_list[0].shape[0] * img_list[0].shape[1]
            target_area = np.random.uniform(0.08, 1.0) * area
            aspect_ratio = np.random.uniform(3.0 / 4, 4.0 / 3)

            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))

            if np.random.random() < 0.5:
                w, h = h, w

            if w <= img_w and h <= img_h:
                x1 = np.random.randint(0, img_w - w)
                y1 = np.random.randint(0, img_h - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_images = []
            for img in img_list:
                img = img[x1 : x1 + w, y1 : y1 + h]
                assert img.shape[0:2] == (h, w)
                res_img = cv2.resize(
                    img, (self.size, self.size), interpolation=self.interpolation
                )
                out_images.extend([res_img])
            return out_images
        else:
            # Fallback
            scale = Rescale(self.size, interpolation=self.interpolation)
            crop = RandomCrop(self.size)
            return crop(scale(img_list))


class Stack(object):
    def __init__(self, modality, length=5):
        self.modality = modality
        self.length = 2 * length

    def __call__(self, img_list):
        assert isinstance(img_list, list)

        out_frames = []
        for img in img_list:
            if self.modality == "RGB":
                img = img.reshape(1, img.shape[0], img.shape[1], 3)
            else:
                img = img.reshape(1, img.shape[0], img.shape[1], 1)
            out_frames.extend(img)

        if self.modality == "Flow":
            flow_frames = []
            for idx in range(0, len(out_frames), self.length):
                img = np.concatenate(out_frames[idx : idx + self.length], axis=2)
                flow_frames.extend([img])
            return np.stack(flow_frames, axis=0)
        else:
            return np.stack(out_frames, axis=0)


class ToTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __call__(self, img_arr):
        assert isinstance(img_arr, np.ndarray)

        out_images = torch.from_numpy(img_arr).permute(0, 3, 1, 2).contiguous().float()
        out_images /= 255

        return out_images


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()

    def __call__(self, tensor):
        if self.mean.size(0) < tensor.size(1):
            self.mean = self.mean.repeat(tensor.size(1))
        if self.std.size(0) < tensor.size(1):
            self.std = self.std.repeat(tensor.size(1))

        rep_mean = self.mean.reshape(1, self.mean.size(0), 1, 1)
        rep_std = self.std.reshape(1, self.std.size(0), 1, 1)

        tensor = (tensor - rep_mean) / rep_std

        return tensor


class TransferTensorDict(object):
    def __init__(self, device):
        super(TransferTensorDict).__init__()
        assert isinstance(device, torch.device)
        self.device = device

    def __call__(self, tensor_dict):
        assert isinstance(tensor_dict, dict)

        for key in tensor_dict.keys():
            tensor_dict[key] = tensor_dict[key].to(self.device)

        return tensor_dict


class IdentityTransform(object):
    def __call__(self, data):
        return data
