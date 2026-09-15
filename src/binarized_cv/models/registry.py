from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from omegaconf import DictConfig, OmegaConf

from binarized_cv.models.base import BaseDetector

ModelClass = TypeVar("ModelClass", bound=type[BaseDetector])

MODEL_REGISTRY: dict[str, type[BaseDetector]] = {}


def register_model(name: str) -> Callable[[ModelClass], ModelClass]:
    """Class decorator: `@register_model("simple_fusion")` makes a detector
    buildable by that name via `build_model`, without the training/eval code
    needing to import the concrete class."""

    def decorator(cls: ModelClass) -> ModelClass:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model {name!r} is already registered to {MODEL_REGISTRY[name]!r}")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(name: str, **kwargs) -> BaseDetector:
    _ensure_builtin_models_registered()
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown model {name!r}. Available: {available}")
    return MODEL_REGISTRY[name](**kwargs)


def _ensure_builtin_models_registered() -> None:
    import binarized_cv.models.detectors  # noqa: F401


def build_model_from_config(model_cfg: DictConfig) -> BaseDetector:
    """Builds whatever model `model_cfg.name` names, passing every other
    field in `model_cfg` through as a constructor kwarg — so a new model
    only needs a `configs/model/<name>.yaml` with `name: <name>` plus its
    own kwargs; the training/eval scripts never hardcode per-model fields."""
    kwargs = OmegaConf.to_container(model_cfg, resolve=True)
    name = kwargs.pop("name")
    kwargs = {key: tuple(value) if isinstance(value, list) else value for key, value in kwargs.items()}
    return build_model(name, **kwargs)
