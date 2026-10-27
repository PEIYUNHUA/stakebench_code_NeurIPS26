from __future__ import annotations

import os
import importlib
from pathlib import Path
from types import ModuleType

try:
    from . import common as common_module
    from . import config
    from .llm_adapter import LLMAdapter
    from .model_profiles import build_hf_model_map, default_model_list, resolve_profile
    from .prompt_factory import (
        build_task0_prompt,
        build_task1_prompt,
        build_task2_decrease_hold_prompt,
        build_task2_flip_hold_prompt,
        build_task2_prompt,
        build_task3_prompt,
    )
except ImportError:
    import common as common_module
    import config
    from llm_adapter import LLMAdapter
    from model_profiles import build_hf_model_map, default_model_list, resolve_profile
    from prompt_factory import (
        build_task0_prompt,
        build_task1_prompt,
        build_task2_decrease_hold_prompt,
        build_task2_flip_hold_prompt,
        build_task2_prompt,
        build_task3_prompt,
    )


_LEGACY: ModuleType | None = None
_ADAPTER: LLMAdapter | None = None


def resolve_out_dir() -> str:
    env = os.environ.get("COMMITBENCH_OUT_DIR")
    if env:
        return env
    base_dir = Path(__file__).resolve().parent.parent
    return str(base_dir / "result" / "commitbench_v18")


def bootstrap() -> tuple[ModuleType, LLMAdapter]:
    global _LEGACY, _ADAPTER
    if _LEGACY is not None and _ADAPTER is not None:
        return _LEGACY, _ADAPTER

    legacy = importlib.reload(common_module)
    legacy.OUT_DIR = resolve_out_dir()
    legacy.MODEL_LIST = default_model_list()
    legacy.HF_MODEL_MAP.update(build_hf_model_map())
    legacy.TRANSFORMERS_BATCH_SIZE = config.TRANSFORMERS_BATCH_SIZE

    adapter = LLMAdapter(legacy)
    adapter.set_log_dir(legacy.OUT_DIR)

    def _profile(model_name: str | None = None):
        return resolve_profile(model_name or legacy.MODEL)

    legacy.prompt_task0 = lambda question, text: build_task0_prompt(_profile(), question, text, zeroshot=False)
    legacy.prompt_task0_zeroshot = lambda question, text: build_task0_prompt(_profile(), question, text, zeroshot=True)
    legacy.prompt_task1 = lambda question, text, topic="": build_task1_prompt(_profile(), question, text, topic=topic)
    legacy.prompt_task2 = (
        lambda question, text, position_side, position_size: build_task2_prompt(
            _profile(), question, text, position_side, position_size
        )
    )
    legacy.prompt_task2_flip_hold = (
        lambda question, text, position_side, position_size: build_task2_flip_hold_prompt(
            _profile(), question, text, position_side, position_size
        )
    )
    legacy.prompt_task2_decrease_hold = (
        lambda question, text, position_side, position_size: build_task2_decrease_hold_prompt(
            _profile(), question, text, position_side, position_size
        )
    )
    legacy.prompt_task3_lead = lambda question, comments_text, current_odds: build_task3_prompt(
        _profile(), question, comments_text, current_odds
    )
    legacy.query_llm = adapter.query
    legacy.check_backend = adapter.check_backend
    legacy._vllm_query_batch = adapter.query_batch

    _LEGACY = legacy
    _ADAPTER = adapter
    return legacy, adapter
