# 发布阅读说明：读取固定交互实体的仿真标签；只用于诊断记录，不参与策略输入或攻击训练。位置使用仿真世界坐标与米。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Read-only first-entity geometry, isolated from historical wrapper imports.

The original anchor/truth functions only need `require`. Its historical
sr_common import loads a chain of unrelated train wrappers named common;
provide only the reader's actual dependency while loading the original file.
No object definition, coordinates, contact rule or physics operation changes.
"""
from functools import lru_cache
from pathlib import Path
import sys
import types
from .common import config,module,require


@lru_cache(None)
def reader():
    key='sr_common';prior=sys.modules.get(key)
    shim=types.ModuleType(key);shim.require=require;shim.native=None
    try:
        sys.modules[key]=shim
        return module('jr04_geometry_reader',Path(config()['project'])/'research_runs/wrist_spatial_approach_pilot_v1/code/sr_geometry.py')
    finally:
        if prior is None:sys.modules.pop(key,None)
        else:sys.modules[key]=prior
