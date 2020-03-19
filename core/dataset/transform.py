import cv2
import numpy as np
import numbers
import math
import torch
import torchvision


class RandomCrop(object):
    """
    Randomly Crop an image

    Args
    ----------
    size: int, tuple
        Crop size
    """

    def __init__(self, size):
        assert isinstance(size, (int, tuple))

        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, img_list):
        """
        Args
        ----------
        img_list: list
            List of input images

        Returns
        ----------
        out_images: list
            List of cropped output images
            
        """

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
    """
    Center Crop an image

    Args
    ----------
    size: int, tuple
        Crop size
    """

    def __init__(self, size):
        assert isinstance(size, (int, tuple))

        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, img_list):
        """
        Args
        ----------
        img_list: list
            List of input images

        Returns
        ----------
        out_images: list
            List of cropped output images

        """

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
    """
    Randomly horizontally flips the given Image with a probability

    Args
    ----------
    prob: float
        Probability of flipping an image

    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img_list):
        """
        Args
        ----------
        img_list: list
            List of input images

        Returns
        ----------
        out_images: list
            List of output images

        """

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
    """ Rescales the input image to the given 'size'.
    
    Args
    ----------
    size: int, tuple
        size of smaller edge or new size of image
    interpolation: cv2 interpolation, default = cv2.INTER_LINEAR
        Interpolation type

    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, (int, tuple))
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_list):
        """
        Args
        ----------
        img_list: list
            List of input images

        Returns
        ----------
        out_images: list
            List of rescaled output images

        """

        assert isinstance(img_list, list)

        h, w = img_list[0].shape[0:2]

        if isinstance(self.size, int):
            # image is rescaled with size as the smaller edge.
            # For example, if height > width, then image will be
            # rescaled to (size * height / width, size)
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
    """ Crop the input image at multiple scales
    
    Args
    ----------
    input_size: int, tuple
        new size of input image
    scales: list, default = [1, 0.875, 0.75, 0.66]
        list of scales to crop at
    max_distort: int default = 1
        Distortion limit
    fix_crop: bool, default = True
        Crop at fixed offset
    more_fix_crop: bool default = True
        More fix crop

    """

    def __init__(
        self,
        input_size,
        scales=[1, 0.875, 0.75, 0.66],
        max_distort=1,
        fix_crop=True,
        more_fix_crop=True,
    ):
        self.scales = scales
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        assert isinstance(input_size, (int, tuple))
        self.input_size = (
            input_size if isinstance(input_size, tuple) else (input_size, input_size)
        )
        self.interpolation = cv2.INTER_LINEAR

    def __call__(self, img_list):
        """
        Args
        ----------
        img_list: list
            List of input images

        Returns
        ----------
        out_images: list
            List of output images

        """

        assert isinstance(img_list, list)

        im_size = img_list[0].shape[0:2]

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)

        out_images = []
        for img in img_list:
            img = img[offset_h : offset_h + crop_h, offset_w : offset_w + crop_w]
            out_images.extend([img])

        rescale = Rescale(self.input_size, interpolation=self.interpolation)
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


class Stack(object):
    """ Stack a given list of input images into a numpy array as per modality
    
    Args
    ----------
    modality: str
        Modality of input images
    length: int, default = 10
        number of optical flow frames to be stacked
    
    """

    def __init__(self, modality, length=10):
        self.modality = modality
        self.length = length

    def __call__(self, img_list):
        """
        Args
        ----------
        img_list: list
            List of input images

        Returns
        ----------
        out_images: np.ndarray
            Array of stacked output images

        """
        assert isinstance(img_list, list)

        out_images = []
        for img in img_list:
            if self.modality == "RGB":
                img = img.reshape(1, img.shape[0], img.shape[1], 3)
            else:
                img = img.reshape(1, img.shape[0], img.shape[1], 1)
            out_images.extend(img)

        if self.modality == "Flow":
            flow_frames = []
            for idx in range(0, len(out_images), self.length):
                img = np.concatenate(out_images[idx : idx + self.length], axis=2)
                flow_frames.extend([img])
            return np.stack(flow_frames, axis=0)
        else:
            return np.stack(out_images, axis=0)


class ToTensor(object):
    """ 
    Converts a numpy.ndarray (N x H x W x C) 
    to a torch.FloatTensor of shape (N x C x H x W)

    Args
    ----------
    is_audio: bool, default = False
        Flag to check if modality is audio or not

    """

    def __init__(self, is_audio=False):
        self.is_audio = is_audio

    def __call__(self, img_arr):
        """
        Args
        ----------
        img_arr: np.ndarray
            Array of input images

        Returns
        ----------
        out_images: torch.Tensor
            Tensor of output images

        """

        assert isinstance(img_arr, np.ndarray)

        out_images = torch.from_numpy(img_arr).permute(0, 3, 1, 2).contiguous().float()
        if not self.is_audio:
            out_images /= 255

        return out_images


class Normalize(object):
    """ 
    Normalize a bunch of given images with mean and standard deviation

    Args
    ----------
    mean: list
        list of mean values
    std: list
        list of standard deviations

    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()

    def __call__(self, tensor):
        """
        Args
        ----------
        tensor: torch.Tensor
            Tensor of input images

        Returns
        ----------
        tensor: torch.Tensor
            Tensor of normalized output images

        """

        if self.mean.size(0) < tensor.size(1):
            self.mean = self.mean.repeat(tensor.size(1))
        if self.std.size(0) < tensor.size(1):
            self.std = self.std.repeat(tensor.size(1))

        rep_mean = self.mean.reshape(1, self.mean.size(0), 1, 1)
        rep_std = self.std.reshape(1, self.std.size(0), 1, 1)

        tensor = (tensor - rep_mean) / rep_std

        return tensor


class TransferTensorDict(object):
    """ 
    Transfer a dictionary of tensors to gpu/cpu

    Args
    ----------
    device: torch.device
        Device name where to transfer tensors

    """

    def __init__(self, device):
        super(TransferTensorDict).__init__()
        assert isinstance(device, torch.device)
        self.device = device

    def __call__(self, tensor_dict):
        """
        Args
        ----------
        tensor_dict: dict
            Dictionary of tensors

        Returns
        ----------
        tensor_dict: dict
            Dictionary of tensors transferred to desired device

        """

        assert isinstance(tensor_dict, dict)

        for key in tensor_dict.keys():
            if isinstance(tensor_dict[key], dict):
                tensor_dict[key] = self.__call__(tensor_dict[key])
            else:
                tensor_dict[key] = tensor_dict[key].to(self.device)

        return tensor_dict
