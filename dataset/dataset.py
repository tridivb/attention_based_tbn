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

from .epic_record import EpicVideoRecord
from .transform import *


class Video_Dataset(Dataset):
    def __init__(
        self,
        cfg,
        vid_list: list,
        annotation_file: str,
        modality: list = ["RGB"],
        transform=None,
        mode: str = "train",
    ):
        self.cfg = cfg
        self.root_dir = cfg.DATA.DATA_DIR
        self.frame_dir = cfg.DATA.FRAME_DIR_PREFIX
        self.audio_dir = cfg.DATA.AUDIO_DIR_PREFIX

        self.vis_file_ext = cfg.DATA.FRAME_FILE_EXT
        self.aud_file_ext = cfg.DATA.AUDIO_FILE_EXT

        self.modality = modality
        self.mode = mode

        self.sampling_rate = cfg.DATA.AUDIO_SAMPLING_RATE
        self.read_pickle = cfg.DATA.READ_AUDIO_PICKLE

        self.transform = transform

        if mode == "train":
            self.num_segments = cfg.TRAIN.NUM_SEGMENTS
        elif mode == "val":
            self.num_segments = cfg.VAL.NUM_SEGMENTS
        elif mode == "test":
            self.num_segments = cfg.TEST.NUM_SEGMENTS

        self.frame_len = {}
        for m in self.modality:
            if m == "Flow":
                self.frame_len[m] = self.cfg.DATA.FLOW_WIN_LENGTH
            else:
                self.frame_len[m] = 1

        if annotation_file.endswith("csv"):
            self.annotations = pd.read_csv(annotation_file)
        elif annotation_file.endswith("pkl"):
            self.annotations = pd.read_pickle(annotation_file)
        self.annotations = self.annotations.query("video_id in @vid_list")

    def __len__(self) -> int:
        return self.annotations.shape[0]

    def __getitem__(self, index: int) -> (Tensor, int):
        """
        
        """

        data = {}

        vid_record = EpicVideoRecord(self.annotations.iloc[index])
        vid_id = vid_record.untrimmed_video_name

        self.frame_path = os.path.join(
            self.cfg.DATA.DATA_DIR, self.cfg.DATA.FRAME_DIR_PREFIX, vid_id
        )
        self.audio_path = os.path.join(
            self.cfg.DATA.DATA_DIR, self.cfg.DATA.AUDIO_DIR_PREFIX, vid_id
        )

        indices = {}
        for m in self.modality:
            indices[m] = self._get_offsets(vid_record, m)
            if self.frame_len[m] > 1:
                frame_indices = (
                    indices[m].repeat(self.frame_len[m])
                    + np.tile(np.arange(self.frame_len[m]), self.num_segments)
                ).astype(np.int64)
                data[m] = self._get_frames(vid_record, m, vid_id, frame_indices)
            else:
                data[m] = self._get_frames(vid_record, m, vid_id, indices[m])
            data[m] = self._transform_data(data[m], m)

        target = vid_record.label

        if self.mode == "train":
            return data, target
        else:
            return data, target, vid_record.action_id

    def _get_offsets(self, vid_record, modality):
        seg_len = (
            vid_record.num_frames[modality] - self.frame_len[modality] + 1
        ) // self.num_segments
        if seg_len > 0:
            if self.mode == "train":
                offsets = np.random.randint(seg_len, size=self.num_segments)
            else:
                offsets = seg_len // 2
            
            indices = (
                vid_record.start_frame[modality]
                + np.arange(0, self.num_segments) * seg_len
                + offsets
            ).astype(int)
        else:
            indices = vid_record.start_frame[modality] + np.zeros((self.num_segments))

        return indices

    def _get_frames(self, vid_record, modality, vid_id, indices):

        frames = []

        for ind in indices:
            frames.extend(self._read_frames(vid_record, ind, vid_id, modality))
        return frames

    def _read_frames(self, vid_record, frame_idx, vid_id, modality):
        if modality == "RGB":
            rgb_file_name = "img_{:010d}.{}".format(frame_idx, self.vis_file_ext)
            rgb_path = os.path.join(
                self.cfg.DATA.DATA_DIR, self.cfg.DATA.FRAME_DIR_PREFIX, vid_id
            )
            img = cv2.imread(os.path.join(rgb_path, rgb_file_name))
            # Convert to rgb
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return [img]
        elif modality == "Flow":
            flow_file_name = [
                "x_{:010d}.{}".format(frame_idx, self.vis_file_ext),
                "y_{:010d}.{}".format(frame_idx, self.vis_file_ext),
            ]
            flow_path = os.path.join(
                self.cfg.DATA.DATA_DIR, self.cfg.DATA.FRAME_DIR_PREFIX, vid_id
            )
            img_x = cv2.imread(os.path.join(flow_path, flow_file_name[0]), 0)
            img_y = cv2.imread(os.path.join(flow_path, flow_file_name[1]), 0)
            return [img_x, img_y]
        elif modality == "Audio":
            spec = self._get_audio(vid_record, frame_idx, vid_id)
            return [spec]

    def _get_audio(self, vid_record, frame_idx, vid_id):
        min_len = int(self.cfg.DATA.AUDIO_LENGTH * self.cfg.DATA.SAMPLING_RATE)

        start_sec = round((frame_idx / self.cfg.DATA.VID_FPS) - (self.cfg.DATA.AUDIO_LENGTH / 2), 3)
        if start_sec + self.cfg.DATA.AUDIO_LENGTH > round(vid_record.end_frame["Audio"] / self.cfg.DATA.VID_FPS, 3):
            start_sec = round(vid_record.end_frame["Audio"] / self.cfg.DATA.VID_FPS, 3) - self.cfg.DATA.AUDIO_LENGTH
        start_frame = int(max(0, start_sec * self.cfg.DATA.SAMPLING_RATE))

        if self.read_pickle:
            npy_file = os.path.join(
                self.cfg.DATA.DATA_DIR,
                self.cfg.DATA.AUDIO_DIR_PREFIX,
                "{}.npy".format(vid_id),
            )
            try:
                sample = np.load(npy_file)
            except Exception as e:
                print(
                    "Failed to read audio sample {} with error {}".format(npy_file, e)
                )
        else:
            aud_file = os.path.join(
                self.cfg.DATA.DATA_DIR,
                self.cfg.DATA.AUDIO_DIR_PREFIX,
                "{}.{}".format(vid_id, self.aud_file_ext),
            )
            try:
                sample, _ = lr.core.load(
                    aud_file,
                    sr=self.cfg.DATA.SAMPLING_RATE,
                    mono=True,
                )
            except Exception as e:
                print(
                    "Failed to read audio sample {} with error {}".format(aud_file, e)
                )

        if sample.shape[0] < min_len:
            sample = np.pad(sample, (0, min_len - sample.shape[0]), mode="constant")
        if sample.shape[0] >= min_len:
            sample = sample[start_frame : start_frame + min_len]

        spec = self._get_spectrogram(sample)

        return spec

    def _get_spectrogram(self, sample, window_size=10, step_size=5, eps=1e-6):
        nperseg = int(round(window_size * self.cfg.DATA.SAMPLING_RATE / 1e3))
        noverlap = int(round(step_size * self.cfg.DATA.SAMPLING_RATE / 1e3))

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

        img_stack = self.transform[modality](img_stack)

        return img_stack
