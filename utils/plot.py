import torch
import torchvision
import numpy as np
import time


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

    # def plot_dict(self, dict_input, epoch, plot_header):
    #     assert isinstance(dict_input, dict)
    #     assert isinstance(epoch, (float, int))
    #     assert isinstance(plot_header, str)

    #     for k, val in dict_input.items():
    #         if isinstance(val, list):
    #             for v in val:
    #                 plot_name = "{} {}_{}".format(k, plot_header, )
    #                 self.writer.add_scalar(plot_name, v, epoch)
    #         elif isinstance(val, (float, int)):
    #             self.writer.add_scalar(plot_header, val, epoch)
