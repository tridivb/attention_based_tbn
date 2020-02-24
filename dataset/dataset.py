import time
import math
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch import Tensor
import torchvision.transforms.functional as F
import librosa as lr
import pandas as pd
from collections import OrderedDict

from .epic_record import EpicVideoRecord
from .transform import *


class Video_Dataset(Dataset):
    """
    Video Dataset class

    Args
    ----------
    cfg: OmegaConf dict
        Dictonary of config parameters
    vid_list: list
        List of videos to process
    annotation_file: str
        Relative path of annotation file containing data of trimmed action segments
    modality: list, default = ["RGB"]
        List of modalities
    transform: list, default = ["ToTensor()"]
        List of transforms to apply
    mode: str, default = "train"
        Mode of dataloader
    
    """

    def __init__(
        self,
        cfg,
        vid_list,
        annotation_file,
        modality=["RGB"],
        transform=["ToTensor()"],
        mode="train",
    ):
        self.cfg = cfg
        self.root_dir = cfg.data.data_dir
        self.rgb_prefix = cfg.data.rgb_dir_prefix
        self.flow_prefix = cfg.data.flow_dir_prefix
        self.audio_prefix = cfg.data.audio_dir_prefix

        self.vis_file_ext = cfg.data.frame_file_ext
        self.aud_file_ext = cfg.data.audio_file_ext

        self.aud_sampling_rate = cfg.data.sampling_rate

        self.modality = modality
        self.mode = mode

        self.read_flow_pickle = cfg.data.read_flow_pickle
        self.read_audio_pickle = cfg.data.read_audio_pickle
        self.use_attention = cfg.model.use_attention

        self.transform = transform

        if mode == "train":
            self.num_segments = cfg.train.num_segments
        elif mode == "val":
            self.num_segments = cfg.val.num_segments
        elif mode == "test":
            self.num_segments = cfg.test.num_segments

        self.frame_len = {}
        for m in self.modality:
            if m == "Flow":
                self.frame_len[m] = cfg.data.flow_win_length
            else:
                self.frame_len[m] = 1

        if annotation_file.endswith("csv"):
            self.annotations = pd.read_csv(annotation_file)
        elif annotation_file.endswith("pkl"):
            self.annotations = pd.read_pickle(annotation_file)
        if vid_list:
            self.annotations = self.annotations.query("video_id in @vid_list")

    def __len__(self):
        """
        Get length of the dataset

        Returns
        ----------
        len: int
            Number of trimmed action segments to be processed
        """

        return self.annotations.shape[0]

    def __getitem__(self, index):
        """
        Get dataset items

        Args
        ----------
        index: int
            Index of dataset to retrieve
        
        Returns
        ----------
        data: dict
            Dictionary of frames for each modality
        target: dict/int
            Dictionary of target labels for each class
        action_id: str
            Video id of the untrimmed video for the trimmed action segment

        """

        data = OrderedDict()
        target = OrderedDict()

        vid_record = EpicVideoRecord(self.annotations.iloc[index])
        vid_id = vid_record.untrimmed_video_name

        indices = {}
        for m in self.modality:
            # TODO This is ugly. Make it robust when using attention or create a different dataset
            # class for sampling with attention
            indices[m] = self._get_offsets(vid_record, m)
            # Read individual flow files
            if m == "Flow" and not self.read_flow_pickle:
                frame_indices = (
                    indices[m].repeat(self.frame_len[m])
                    + np.tile(np.arange(self.frame_len[m]), self.num_segments)
                ).astype(np.int64)
                data[m], _ = self._get_frames(vid_record, m, vid_id, frame_indices)
            elif m == "Audio" and self.use_attention:
                indices[m] = indices["RGB"]
                data[m], gt_attn_wts = self._get_frames(
                    vid_record, m, vid_id, indices[m]
                )
            else:
                data[m], _ = self._get_frames(vid_record, m, vid_id, indices[m])
            data[m] = self._transform_data(data[m], m)

        target["class"] = vid_record.label
        if self.use_attention:
            target["weights"] = gt_attn_wts

        if self.mode == "train":
            return data, target
        else:
            return data, target, vid_record.action_id

    def _get_offsets(self, vid_record, modality):
        """
        Helper function to get offsets for each temporal binding window

        Args
        ----------
        vid_record: EpicVideoRecord object
            Object of EpicVideoRecord class
        modality: str
            Input modality to process
        
        Returns
        ----------
        indices: np.ndarray
            Array of indices for number of segments or temporal binding window.

        """

        if self.mode == "train":
            seg_len = (
                vid_record.num_frames[modality] - self.frame_len[modality] + 1
            ) // self.num_segments
        else:
            seg_len = vid_record.num_frames[modality] // self.num_segments
        if seg_len > 0:
            if self.mode == "train":
                # randomly select an offset for the segment
                offsets = np.random.randint(seg_len, size=self.num_segments)
            else:
                # choose the center as the offset of the segment
                offsets = seg_len // 2
                # Center the flow window during validation
                if modality == "Flow":
                    offsets = offsets - (self.frame_len[modality] // 2)

            indices = (
                vid_record.start_frame[modality]
                + np.arange(0, self.num_segments) * seg_len
                + offsets
            ).astype(int)
        else:
            indices = vid_record.start_frame[modality] + np.zeros(
                (self.num_segments), dtype=int
            )

        return indices

    def _get_frames(self, record, modality, vid_id, indices):
        """
        Helper function to get list of frames for a specific modality

        Args
        ----------
        modality: str
            Input modality to process
        vid_id: str
            Untrimmed Video id
        indices: np.ndarray
            Array of indices to process
        
        Returns
        ----------
        frames: list
            List of arrays for images or spectrograms

        """

        frames = []
        gt_attn_wts = []
        if modality == "Audio":
            aud_sample = self._read_audio_sample(vid_id)
        else:
            aud_sample = None

        for ind in indices:
            if modality == "Audio":
                frame, gt_attn_wt = self._read_frames(
                    record, ind, vid_id, modality, aud_sample
                )
                if self.use_attention:
                    gt_attn_wts.extend(gt_attn_wt)
            else:
                frame = self._read_frames(record, ind, vid_id, modality, aud_sample)
            frames.extend(frame)

        if len(gt_attn_wts) > 0:
            gt_attn_wts = torch.stack(gt_attn_wts)

        return frames, gt_attn_wts

    def _read_frames(self, record, frame_idx, vid_id, modality, aud_sample=None):
        """
        Helper function to get read images or get spectrogram for an index

        Args
        ----------
        frame_idx: int
            Index of the frame to be read
        vid_id: str
            Untrimmed Video id
        modality: str
            Input modality to process
        
        Returns
        ----------
        img/spec: list
            List of one array containing the image or spectrogram

        """

        if modality == "RGB":
            rgb_file_name = "img_{:010d}.{}".format(frame_idx, self.vis_file_ext)
            rgb_path = os.path.join(self.root_dir, self.rgb_prefix, vid_id)
            img = cv2.imread(os.path.join(rgb_path, rgb_file_name))
            # Convert to rgb
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return [img]
        elif modality == "Flow":
            return self._read_flow_frames(frame_idx, vid_id)
        elif modality == "Audio":
            spec, gt_attn_wts = self._get_audio_segment(
                record.start_frame["Audio"],
                record.end_frame["Audio"],
                frame_idx,
                aud_sample,
            )
            return [spec], [gt_attn_wts]

    def _read_flow_frames(self, frame_idx, vid_id):
        """
        Helper function to read optical flow images from compressed image or numpy files

        Args
        ----------
        frame_idx: int
            Index of the frame to be read
        vid_id: str
            Untrimmed Video id
        
        Returns
        ----------
        img: list
            List of stacked optical flow frames

        """
        if self.read_flow_pickle:
            # Read an array of stacked optical flow frames
            flow_file_name = "frame_{:010d}.npz".format(frame_idx)
            flow_path = os.path.join(self.root_dir, self.flow_prefix, vid_id)
            img = None
            try:
                with np.load(os.path.join(flow_path, flow_file_name)) as data:
                    img = data["flow"]
                    img = [img[:, :, c] for c in range(img.shape[2])]
                    data.close()
            except Exception as e:
                raise Exception(
                    "Failed to load flow file {} with error {}.".format(
                        os.path.join(flow_path, flow_file_name), e
                    )
                )
            return img
        else:
            # Read individual optical flow frames into a list
            flow_file_name = [
                "x_{:010d}.{}".format(frame_idx, self.vis_file_ext),
                "y_{:010d}.{}".format(frame_idx, self.vis_file_ext),
            ]
            flow_path = os.path.join(self.root_dir, self.rgb_prefix, vid_id)
            img_x = cv2.imread(os.path.join(flow_path, flow_file_name[0]), 0)
            img_y = cv2.imread(os.path.join(flow_path, flow_file_name[1]), 0)
            return [img_x, img_y]

    def _read_audio_sample(self, vid_id):
        """
        Helper function to read audio data from raw or numpy files

        Args
        ----------
        vid_id: str
            Untrimmed Video id
        
        Returns
        ----------
        sample: np.ndarray
            Array of 1D audio data

        """

        # Read untrimmed audio sample
        if self.read_audio_pickle:
            # Read from numpy file
            npy_file = os.path.join(
                self.root_dir, self.audio_prefix, "{}.npy".format(vid_id),
            )
            try:
                sample = np.load(npy_file)
            except Exception as e:
                raise Exception(
                    "Failed to read audio sample {} with error {}".format(npy_file, e)
                )
        else:
            # Read from raw file
            aud_file = os.path.join(
                self.root_dir,
                self.audio_prefix,
                "{}.{}".format(vid_id, self.aud_file_ext),
            )
            try:
                sample, _ = lr.core.load(
                    aud_file, sr=self.aud_sampling_rate, mono=True,
                )
            except Exception as e:
                raise Exception(
                    "Failed to read audio sample {} with error {}".format(aud_file, e)
                )

        return sample

    def _get_audio_segment(self, ann_start_frame, ann_end_frame, frame_idx, aud_sample):
        """
        Helper function to trim sampled audio and return a spectrogram

        Args
        ----------
        frame_idx: int
            Center Index of the temporal audio window to be read
        aud_sample: np.ndarray
            Untrimmed 1D audio sample
        
        Returns
        ----------
        spec: np.ndarray
            Array of audio spectrogram

        """

        min_len = int(self.cfg.data.audio_length * self.aud_sampling_rate)
        max_len = aud_sample.shape[0]

        if max_len < min_len:
            aud_sample = np.pad(aud_sample, (0, min_len - max_len))

        # Find the starting temporal offset of the audio sample
        # if self.use_attention:
        #     start_sec = float(ann_start_frame / self.cfg.data.vid_fps)
        # else:
        start_sec = float(frame_idx / self.cfg.data.vid_fps) - (
            self.cfg.data.audio_length / 2
        )
        # Find the starting frame of the audio sample array
        start_frame = int(max(0, start_sec * self.aud_sampling_rate))
        if start_frame + min_len > max_len:
            start_frame = max_len - min_len

        sample = aud_sample[start_frame : start_frame + min_len]

        spec = self._get_spectrogram(sample)
        gt_attn_wts = self._get_attn_weights(frame_idx, start_sec)

        return spec, gt_attn_wts

    def _get_spectrogram(self, sample, window_size=10, step_size=5, eps=1e-6):
        """
        Helper function to create 2D audio spectogram from 1D audio sample

        Args
        ----------
        sample: np.ndarray
            1D array of untrimmed audio sample
        window_size: int, default = 10
            Temporal size in millisecond of window function for STFT
        step_size: int, default = 5
            Hop size in millisecond between each sample for STFT
        eps: double default = 1e-6
            Correction term to prevent divide by zero
        
        Returns
        ----------
        spec: np.ndarray
            Array of audio spectrogram

        """

        nperseg = int(round(window_size * self.aud_sampling_rate / 1e3))
        noverlap = int(round(step_size * self.aud_sampling_rate / 1e3))

        spec = lr.stft(
            sample,
            n_fft=511,
            window="hann",
            hop_length=noverlap,
            win_length=nperseg,
            pad_mode="constant",
        )

        spec = np.log(np.real(spec * np.conj(spec)) + eps)
        return spec

    def _transform_data(self, img_stack, modality):
        """
        Helper function to transform input data

        Args
        ----------
        img_stack: list
            list of stacked input frames
        modality: str
            Modality of input frames
        
        Returns
        ----------
        img_stack: Tensor
            A tensor of transformed input frames

        """

        img_stack = self.transform[modality](img_stack)

        return img_stack

    def _get_attn_weights(self, index, start_time):
        """
        """
        gt_attn_wts = cv2.getGaussianKernel(25, sigma=1.5)
        mean_loc = gt_attn_wts.shape[0] // 2
        ind_time = float(index / self.cfg.data.vid_fps)
        diff = ind_time - start_time
        new_mean_loc = round(diff * 25 / self.cfg.data.audio_length)
        if new_mean_loc <= gt_attn_wts.shape[0]:
            gt_attn_wts = np.roll(gt_attn_wts, new_mean_loc - mean_loc)
            if new_mean_loc - 6 > 0:
                gt_attn_wts[: new_mean_loc - 6] = 1e-6
            if new_mean_loc + 6 < gt_attn_wts.shape[0]:
                gt_attn_wts[new_mean_loc + 6 :] = 1e-6
        else:
            gt_attn_wts = np.zeros(gt_attn_wts.shape, dtype=np.float32) + 1e-6
        return torch.tensor(gt_attn_wts).float()
