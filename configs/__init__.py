# Original code from: https://github.com/groomata/vision/blob/main/src/groovis/configs/__init__.py
import importlib.util
import os
from typing import Any

from hydra_zen import MISSING, load_from_yaml, make_config, make_custom_builds_fn, store, to_yaml
from hydra_zen.third_party.beartype import validates_with_beartype
from omegaconf import OmegaConf
from rich import print


def print_yaml(config: Any):
    print(to_yaml(config))


partial_builds = make_custom_builds_fn(
    populate_full_signature=True,
    zen_partial=True,
    zen_wrappers=validates_with_beartype,
)
full_builds = make_custom_builds_fn(
    populate_full_signature=True,
    zen_wrappers=validates_with_beartype,
)


defaults = [
    "_self_",
    {"data": "mnist"},
    {"model": "mnist"},
    {"callbacks": "default"},
    {"logger": "wandb"},
    {"trainer": "cpu"},
    {"paths": "default"},
    {"extras": "default"},
    {"experiment": None},
    {"hparams_search": None},
    {"debug": None},
]

# Config = load_from_yaml("configs/train.yaml")

TrainConfig = make_config(
    data=None,
    model=None,
    callbacks=None,
    logger=None,
    trainer=None,
    paths=None,
    extras=None,
    hydra=None,
    experiment=None,
    hparams_search=None,
    debug=None,
    task_name=MISSING,
    tags=["dev"],
    train=True,
    test=False,
    ckpt_path=MISSING,
    seed=MISSING,
    hydra_defaults=defaults,
)
EvalConfig = make_config(
    task_name=MISSING,
    tags=["dev"],
    train=False,
    test=True,
    ckpt_path=MISSING,
    seed=MISSING,
    hydra_defaults=defaults,
)

# store(TrainConfig, name="train")
# store(EvalConfig, name="eval")


def register_configs():
    """Register all configs under config directory to hydra-zen."""
    for root, dirs, files in os.walk(os.path.dirname(__file__)):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                path = os.path.join(root, file)
                spec = importlib.util.spec_from_file_location(file, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "_register_configs"):
                    module._register_configs()


def register_everything() -> None:
    """Register all configs and eval resolver to hydra global Configstore."""
    register_configs()
    store.add_to_hydra_store()
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
