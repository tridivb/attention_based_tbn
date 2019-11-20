from .video_record import VideoRecord


class EpicVideoRecord(VideoRecord):
    def __init__(self, record):
        self._series = record

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    @property
    def start_frame(self):
        return {'RGB': self._series['start_frame'] - 1, "Flow": (self._series['start_frame'] - 1) // 2, "Spec": self._series['start_frame'] - 1}

    @property
    def end_frame(self):
        return {'RGB': self._series['stop_frame'] - 2, "Flow": (self._series['stop_frame'] - 2) // 2, "Spec": self._series['stop_frame'] - 2}

    @property
    def num_frames(self):
        return {'RGB': self.end_frame - self.start_frame, 'Flow': (self.end_frame - self.start_frame) / 2,
                'Spec': self.end_frame - self.start_frame}
    @property
    def label(self):
        if 'verb_class' in self._series.keys().tolist():
            label = {'verb': self._series['verb_class'], 'noun': self._series['noun_class']}
        else:  # Fake label to deal with the test sets (S1/S2) that dont have any labels
            label = -10000
        return label
