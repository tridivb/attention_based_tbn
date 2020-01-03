import torch
import torchvision
import numpy as np
import time
from omegaconf.dictconfig import DictConfig


class Plotter(object):
    """
    Class to initialize plotter for graphs and images
    """
    def __init__(self, writer):
        super(Plotter).__init__()
        self.writer = writer

    def plot_scalar(self, val, epoch, plot_name):
        """
        Plot scalar values in a 2D graph
        """
        assert isinstance(val, (int, float))
        assert isinstance(epoch, (int, float))
        assert isinstance(plot_name, str)

        self.writer.add_scalar(plot_name, val, epoch)

    def add_config(self, cfg):
        config_summary = ""
        for k, d in cfg.items():
            if isinstance(d, DictConfig):
                config_summary += k + "<br/>"
                for key, val in d.items():
                    config_summary += "&nbsp;&nbsp;&nbsp;&nbsp;" + key + ": " + str(val) + "<br/>"
            else:
                config_summary += k + ": " + str(d) + "<br/>"
        self.writer.add_text("Config", config_summary)
