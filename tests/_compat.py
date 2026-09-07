"""Import-compatibility helpers.

``BookerBV2Tool/data_utils.py`` (and ``models.py``/``train.py``) were written
for the upstream Bert-VITS2 layout where the package root *is* the top-level
namespace: they use absolute imports such as ``import commons``,
``from config import config`` and ``from tools.log import logger``.  Some of
those modules (``config``, ``tools``) are not shipped inside this repository.

:func:`install_top_level_aliases` registers ``sys.modules`` aliases so that
``import BookerBV2Tool.data_utils`` succeeds against the real source modules.
"""

import logging
import sys
import types
from types import SimpleNamespace


def install_top_level_aliases():
    """Make ``data_utils`` importable and return the imported module."""
    from BookerBV2Tool import commons as real_commons
    from BookerBV2Tool import mel_processing as real_mel
    from BookerBV2Tool import utils as real_utils
    from BookerBV2Tool.text import cleaner as real_cleaner

    sys.modules.setdefault("commons", real_commons)
    sys.modules.setdefault("mel_processing", real_mel)
    sys.modules.setdefault("utils", real_utils)
    # data_utils does `from text import cleaned_text_to_sequence`.
    sys.modules.setdefault("text", real_cleaner)

    if "config" not in sys.modules:
        config_mod = types.ModuleType("config")
        config_mod.config = SimpleNamespace(
            train_ms_config=SimpleNamespace(spec_cache=False)
        )
        sys.modules["config"] = config_mod

    if "tools" not in sys.modules or "tools.log" not in sys.modules:
        tools = types.ModuleType("tools")
        tools_log = types.ModuleType("tools.log")
        tools_log.logger = logging.getLogger("tests.tools")
        tools.log = tools_log
        sys.modules["tools"] = tools
        sys.modules["tools.log"] = tools_log

    import BookerBV2Tool.data_utils

    return sys.modules["BookerBV2Tool.data_utils"]