#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""轻量配置对象，供 ``data_utils.py`` 读取 spec 缓存开关。

上游 Bert-VITS2 的 ``config.py`` 在导入时就要读取 ``config.yml`` 并在 CWD 里
找配置文件，与本仓库“用 ``config.json`` 驱动 train”的模式不符。这里提供一个
最小替代：模块级 ``config`` 对象，其 ``train_ms_config.spec_cache`` 从 CWD 的
``config.json`` 中读取（不存在则默认关闭 spec 缓存）。
"""

import json
import os


class _TrainMsConfig:
    def __init__(self, spec_cache=False):
        self.spec_cache = spec_cache


class _Config:
    def __init__(self):
        self.train_ms_config = _TrainMsConfig()
        self._load()

    def _load(self):
        try:
            path = os.path.join(os.getcwd(), "config.json")
            with open(path, encoding="utf-8") as f:
                data = json.load(f).get("data", {})
            self.train_ms_config.spec_cache = bool(data.get("spec_cache", False))
        except Exception:
            # config.json 不存在或不可解析时，默认关闭 spec 缓存
            self.train_ms_config.spec_cache = False


config = _Config()