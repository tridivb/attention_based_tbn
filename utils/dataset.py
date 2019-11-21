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
from epic_record import EpicVideoRecord

class Video_Dataset(Dataset):
    def __init__(
        self,
        cfg,
        vid_list: list,
        modality: list = ["RGB"],
        mode: str = "train"
    ):
        self.cfg = cfg
        self.root_dir = cfg.DATA.DATA_DIR
        self.frame_dir = cfg.DATA.FRAME_DIR_PREFIX
        self.audio_dir = cfg.DATA.AUDIO_DIR_PREFIX

        self.vid_list = vid_list
        self.vis_file_ext = cfg.DATA.FRAME_FILE_EXT
        self.aud_file_ext = cfg.DATA.AUDIO_FILE_EXT

        self.modality = modality
        self.mode = mode
        self.num_segments = cfg.DATA.NUM_SEGMENTS
        self.sampling_rate = cfg.DATA.AUDIO_SAMPLING_RATE

    def __len__(self) -> int:
        return self.vid_list.shape[0]

    def __getitem__(self, index: int) -> (Tensor, int):
        """
        
        """

        data = {}

        vid_record = EpicVideoRecord(self.vid_list.iloc[index])
        vid_id = vid_record.untrimmed_video_name()

        self.frame_path = os.path.join(self.cfg.DATA.DATA_DIR, self.cfg.DATA.FRAME_DIR_PREFIX, vid_id)
        self.audio_path = os.path.join(self.cfg.DATA.DATA_DIR, self.cfg.DATA.AUDIO_DIR_PREFIX, vid_id)

        indices = {}
        for m in self.modality:
            indices[m] = self._get_offsets(vid_record, m)
        
        frames = {}
        for m in self.modality:
            frames[m] = self._get_frames(m, vid_id, indices)

        data["input"] = frames
        data["target"] = vid_record.label

        return data

    def _get_offsets(self, vid_record, modality):
        win_len = vid_record.num_frames[modality] // self.num_segments
        if win_len > 0:
            offsets = np.random.randint(win_len, size=self.num_segments)
            indices = vid_record.start_frame[modality] + np.arange(0, self.num_segments) * win_len + offsets
        else:
            indices = vid_record.start_frame[modality] + np.zeros((self.num_segments))
        return indices

    def _get_frames(self, modality, vid_id, indices):
        frames = []
        if modality == "FLOW":
            win_len = self.cfg.DATA.FLOW_WIN_LENGTH
        else:
            win_len = 1
        for ind in indices:
            for ind_offset in enumerate(range(win_len)):
                frames.extend(self._read_frames(ind + ind_offset, vid_id, modality))

        return frames

    def _read_frames(self, frame_idx, vid_id, modality):
        if modality == "RGB":
            rgb_file_name = "img_{:010d}.{}".format(frame_idx, self.vis_file_ext)
            rgb_path = os.path.join(self.cfg.DATA.DATA_DIR, self.cfg.DATA.FRAME_DIR_PREFIX, vid_id)
            img = cv2.imread(os.path.join(rgb_path, rgb_file_name))
            return [img]
        elif modality == "FLOW":
            flow_file_name = ["x_{:010d}.{}".format(frame_idx, self.vis_file_ext), "y_{:010d}.{}".format(frame_idx, self.vis_file_ext)]
            flow_path = os.path.join(self.cfg.DATA.DATA_DIR, self.cfg.DATA.FRAME_DIR_PREFIX, vid_id)
            img = [cv2.imread(os.path.join(flow_path, flow_file_name[0]), 0), cv2.imread(os.path.join(flow_path, flow_file_name[1]), 0)]
            return img
        elif modality == "AUD":
            sample = self._get_audio(frame_idx, vid_id)
            spec = self._get_spectrogram(sample)
            return [spec]

    def _get_audio(self, frame_idx, vid_id):
        start_sec = (frame_idx / self.cfg.DATA.IN_FPS) - (self.cfg.DATA.AUDIO_LENGTH / 2)
        start_sec = max(0, start_sec)

        aud_file = os.path.join(self.cfg.DATA.DATA_DIR, self.cfg.DATA.AUDIO_DIR_PREFIX, "{}.{}".format(vid_id, self.aud_file_ext))

        try:
            sample, _ = lr.core.load(aud_file, sr=self.cfg.DATA.SAMPLING_RATE, mono=True, offset=start_sec, duration=self.cfg.DATA.AUDIO_LENGTH)
            min_len = self.cfg.DATA.AUDIO_LENGTH * self.cfg.DATA.SAMPLING_RATE
            if sample.shape[0] < min_len:
                sample = np.pad(sample, (0, min_len-sample.shape[0]), mode="constant")
            if sample.shape[0] > min_len:
                sample = sample[:min_len]
            
            return sample

        except Exception as e:
            print("Failed to read audio file {} with error {}".format(aud_file, e)) 


    def _get_spectrogram(self, sample, window_size=10, step_size=5, eps=1e-6):
        nperseg = int(round(window_size * self.cfg.DATA.SAMPLING_RATE / 1e3))
        noverlap = int(round(step_size * self.cfg.DATA.SAMPLING_RATE / 1e3))

        spec = lr.stft(sample, n_fft=511,
                            window='hann',
                            hop_length=noverlap,
                            win_length=nperseg,
                            pad_mode='constant')

        spec = np.log(np.real(spec * np.conj(spec)) + eps)
        return spec

    def _transform_data(self):
        raise Exception("Not implemented")