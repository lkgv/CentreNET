import numpy
import torch
import torch.nn as nn

class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, out, target):
        ret = out - target
        ret.data.map_(ret.data,
                      (lambda a, b:
                       (0.5 * a * a if abs(a) < 1
                        else abs(a) - 0.5)))
        return ret
