
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def segment_name(self):
        return NotImplementedError()

    @property
    def untrimmed_video_name(self):
        return NotImplementedError()

    @property
    def start_frame(self):
        return NotImplementedError()

    @property
    def end_frame(self):
        return NotImplementedError()

    @property
    def num_frames(self):
        return NotImplementedError()

    @property
    def label(self):
        return NotImplementedError()