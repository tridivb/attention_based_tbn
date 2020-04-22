import os
import ast
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
    def verb_df(self):
        df = pd.read_csv(os.path.join(self._ann_path, "EPIC_verb_classes.csv"))
        df["verbs"] = df["verbs"].apply(ast.literal_eval)
        df = df.explode("verbs")[["verb_id", "verbs"]]
        return df

    @property
    def noun_df(self):
        df = pd.read_csv(os.path.join(self._ann_path, "EPIC_noun_classes.csv"))
        df["nouns"] = df["nouns"].apply(ast.literal_eval)
        df = df.explode("nouns")[["noun_id", "nouns"]]
        return df

    @property
    def nouns(self):
        noun_classes = pd.read_csv(
            os.path.join(self._ann_path, "EPIC_noun_classes.csv")
        ).class_key.tolist()
        return noun_classes

    @property
    def actions(self):
        df = pd.read_csv(os.path.join(self._ann_path, "EPIC_many_shot_actions.csv"))
        df["action"] = df[["verb", "noun"]].agg(" ".join, axis=1)
        action_classes = df.action.to_list()
        return action_classes
