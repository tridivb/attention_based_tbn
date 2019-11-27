import math
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch import Tensor
import torchvision.transforms.functional as F
import librosa as lr
from parse import parse
import pandas as pd

from .epic_record import EpicVideoRecord
from .transform import *


class Video_Dataset(Dataset):
    def __init__(
        self,
        cfg,
        vid_list: list,
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
        self.num_segments = cfg.DATA.NUM_SEGMENTS
        self.sampling_rate = cfg.DATA.AUDIO_SAMPLING_RATE

        self.transform = transform

        annotation_file = cfg.DATA.ANNOTATION_FILE

        self.frame_len = {}
        for m in self.modality:
            if m == "Flow":
                self.frame_len[m] = self.cfg.DATA.FLOW_WIN_LENGTH
            else:
                self.frame_len[m] = 1

        if mode in ["train", "val"]:
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

        for m in self.modality:
            data[m] = self._get_frames(m, vid_id, indices)
            data[m] = self._transform_data(data[m], m)

        data["target"] = vid_record.label

        return data

    def _get_offsets(self, vid_record, modality):
        seg_len = vid_record.num_frames[modality] // self.num_segments
        if self.mode == "train":
            if seg_len > 0:
                offsets = np.random.randint(seg_len, size=self.num_segments)
                indices = (
                    vid_record.start_frame[modality]
                    + np.arange(0, self.num_segments) * seg_len
                    + offsets
                )
            else:
                indices = vid_record.start_frame[modality] + np.zeros(
                    (self.num_segments)
                )
        elif self.mode == "val":
            if seg_len > 0:
                offsets = (
                    (seg_len - self.frame_len[modality] + 1)
                    // 2
                    * np.ones((self.num_segments))
                )
                indices = (
                    vid_record.start_frame[modality]
                    + np.arange(0, self.num_segments) * seg_len
                    + offsets
                )
            else:
                indices = vid_record.start_frame[modality] + np.zeros(
                    (self.num_segments)
                )

        return indices

    def _get_frames(self, modality, vid_id, indices):

        frames = []

        for ind in indices[modality]:
            frame = []
            for ind_offset in range(self.frame_len[modality]):
                frame.extend(self._read_frames(ind + ind_offset, vid_id, modality))
            if modality == "Audio":
                frame = frame[0]
            else:
                frame = np.concatenate(frame, axis=2)
            frames.extend(frame[None, ...])

        frames = np.stack(frames, axis=0)

        return frames

    def _read_frames(self, frame_idx, vid_id, modality):
        if modality == "RGB":
            rgb_file_name = "img_{:010d}.{}".format(frame_idx, self.vis_file_ext)
            rgb_path = os.path.join(
                self.cfg.DATA.DATA_DIR, self.cfg.DATA.FRAME_DIR_PREFIX, vid_id
            )
            img = cv2.imread(os.path.join(rgb_path, rgb_file_name))
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
            img = np.concatenate((img_x[..., None], img_y[..., None]), axis=2)
            return [img]
        elif modality == "Audio":
            sample = self._get_audio(frame_idx, vid_id)
            spec = self._get_spectrogram(sample)
            return [spec]

    def _get_audio(self, frame_idx, vid_id):
        start_sec = (frame_idx / self.cfg.DATA.IN_FPS) - (
            self.cfg.DATA.AUDIO_LENGTH / 2
        )
        start_sec = max(0, start_sec)

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
                offset=start_sec,
                duration=self.cfg.DATA.AUDIO_LENGTH,
            )
            min_len = self.cfg.DATA.AUDIO_LENGTH * self.cfg.DATA.SAMPLING_RATE
            if sample.shape[0] < min_len:
                sample = np.pad(sample, (0, min_len - sample.shape[0]), mode="constant")
            if sample.shape[0] > min_len:
                sample = sample[:min_len]

            return sample

        except Exception as e:
            print("Failed to read audio file {} with error {}".format(aud_file, e))

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
