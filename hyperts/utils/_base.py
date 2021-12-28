# -*- coding:utf-8 -*-
"""

"""
import time

_tool_boxes = []

def register_tstoolbox(tb, pos=None):
    if pos is None:
        _tool_boxes.append(tb)
    else:
        _tool_boxes.insert(pos, tb)

    global _tool_boxes_update_at
    _tool_boxes_update_at = time.time()

def get_tool_box(*data):
    for tb in _tool_boxes:
        if tb.accept(*data):
            return tb

    raise ValueError(f'No toolbox found for your data with types: {[type(x) for x in data]}. '
                     f'Registered tabular toolboxes are {[t.__name__ for t in _tool_boxes]}.')
