import glob
from pathlib import Path
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

import numpy as np
import torch

from APP  import PROJ_SETTINGS, EXPERIMENT_SETTINGS

def get_widget(parent:QFrame, name):
    for widget in parent.children():
        if widget:
            if widget.objectName() == name:
                return widget
            else:
                w = get_widget(widget, name)
                if w != None:
                    return w
    return None

def getcat(value):
    return torch.cat if isinstance(value, torch.Tensor) else np.concatenate

