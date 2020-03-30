import os
import pandas as pd
from .dataset_class import DatasetClasses


class EpicClasses(DatasetClasses):
    def __init__(self, ann_path):
        super(EpicClasses).__init__()
        self._ann_path = ann_path

    @property
    def verbs(self):
        verb_classes = pd.read_csv(
            os.path.join(self._ann_path, "EPIC_verb_classes.csv")
        ).class_key.tolist()
        return verb_classes

    @property
    def nouns(self):
        noun_classes = pd.read_csv(
            os.path.join(self._ann_path, "EPIC_noun_classes.csv")
        ).class_key.tolist()
        return noun_classes
