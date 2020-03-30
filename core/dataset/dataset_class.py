class DatasetClasses(object):
    def __init__(self, ann_path):
        self._ann_path = ann_path

    @property
    def obj_class(self):
        return NotImplementedError()
