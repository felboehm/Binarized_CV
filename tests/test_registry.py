import pytest
from omegaconf import OmegaConf

from binarized_cv.models.base import BaseDetector
from binarized_cv.models.registry import (
    MODEL_REGISTRY,
    build_model,
    build_model_from_config,
    register_model,
)


def test_register_and_build_model():
    name = "_test_dummy_detector"

    @register_model(name)
    class DummyDetector(BaseDetector):
        modalities = ("rgb",)

        def forward(self, images, targets=None):
            return {} if targets is not None else []

    try:
        model = build_model(name)
        assert isinstance(model, DummyDetector)
    finally:
        del MODEL_REGISTRY[name]


def test_build_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model("does-not-exist")


def test_double_registration_raises():
    name = "_test_dup_detector"

    @register_model(name)
    class A(BaseDetector):
        modalities = ("rgb",)

        def forward(self, images, targets=None):
            return {}

    try:
        with pytest.raises(ValueError):

            @register_model(name)
            class B(BaseDetector):
                modalities = ("rgb",)

                def forward(self, images, targets=None):
                    return {}

    finally:
        del MODEL_REGISTRY[name]


def test_builtin_simple_fusion_model_is_registered():
    model = build_model("simple_fusion")
    assert "simple_fusion" in MODEL_REGISTRY
    assert model.modalities == ("rgb", "ir")


def test_build_model_from_config_passes_through_arbitrary_kwargs_and_converts_lists_to_tuples():
    name = "_test_config_built_detector"
    seen_kwargs = {}

    @register_model(name)
    class ConfiguredDetector(BaseDetector):
        modalities = ("rgb",)

        def __init__(self, img_size, num_classes):
            super().__init__()
            seen_kwargs["img_size"] = img_size
            seen_kwargs["num_classes"] = num_classes

        def forward(self, images, targets=None):
            return {}

    try:
        model_cfg = OmegaConf.create({"name": name, "img_size": [32, 32], "num_classes": 1})
        model = build_model_from_config(model_cfg)

        assert isinstance(model, ConfiguredDetector)
        assert seen_kwargs == {"img_size": (32, 32), "num_classes": 1}
    finally:
        del MODEL_REGISTRY[name]
