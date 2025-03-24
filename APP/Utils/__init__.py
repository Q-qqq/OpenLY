import glob
from pathlib import Path
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import numpy as np
import torch

from APP.Data import readLabelFile
from ultralytics.data.utils import check_det_dataset, img2label_paths
from ultralytics.utils import ROOT

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


def append_formatted_text(text_edit, text, font_size, color=None, bold=False):
    # 获取当前光标并移动到文档末尾
    cursor = text_edit.textCursor()
    cursor.movePosition(QTextCursor.End)
    
    # 创建格式并设置属性
    fmt = QTextCharFormat()
    fmt.setFont(QFont("Cascadia Mono", font_size))  # 字体大小
    if color:
        fmt.setForeground(color)
    if bold:
        fmt.setFontWeight(QFont.Weight.Bold)       # 粗体
    
    # 插入带格式的文本并换行
    cursor.insertText(text + "\n", fmt)
    
    # 更新光标位置
    text_edit.setTextCursor(cursor)
    text_edit.ensureCursorVisible()

