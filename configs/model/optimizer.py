# Original code from: https://github.com/groomata/vision/blob/main/src/groovis/configs/optimizer.py

# import torch_optimizer
# from aihwkit.optim import AnalogSGD
from hydra_zen import make_custom_builds_fn, store
from torch import optim

pbuilds = make_custom_builds_fn(
    zen_partial=True
)  # To make it pickleable, we need to give up beartype support

SGDConfig = pbuilds(optim.SGD, lr=0.001, momentum=0.9)
RMSPropConfig = pbuilds(optim.RMSprop, lr=0.001)
AdamConfig = pbuilds(optim.Adam, lr=0.001)
AdamWConfig = pbuilds(optim.AdamW, lr=0.001)
RAdamConfig = pbuilds(optim.RAdam, lr=0.001)
# LARSConfig = pbuilds(torch_optimizer.LARS, lr=0.001)
# AnalogSGDConfig = pbuilds(AnalogSGD, lr=0.01, momentum=0.9)


def _register_configs():
    optim_store = store(group="model/optimizer")
    optim_store(SGDConfig, name="sgd")
    optim_store(RMSPropConfig, name="rmsprop")
    optim_store(AdamConfig, name="adam")
    optim_store(AdamWConfig, name="adamw")
    optim_store(RAdamConfig, name="radam")
    # optim_store(name="lars", LARSConfig)
    # optim_store(AnalogSGDConfig, name="analog_sgd")
