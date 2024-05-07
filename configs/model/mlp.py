from ast import mod

from hydra_zen import MISSING, builds, make_config, store

from configs import full_builds
from src.models.components.simple_dense_net import SimpleDenseNet
from src.models.mnist_module import MNISTLitModule

MNISTBackboneConfig = full_builds(SimpleDenseNet)

mnist_narrow_backbone = MNISTBackboneConfig(lin1_size=128, lin2_size=128, lin3_size=64)
mnist_wide_backbone = MNISTBackboneConfig(lin1_size=512, lin2_size=512, lin3_size=256)

ModuleConfig = make_config(net=MISSING, optimizer=MISSING, scheduler=MISSING, compile=False)

MNISTModuleConfig = builds(
    MNISTLitModule,
    net=mnist_narrow_backbone,
    scheduler=None,
    builds_bases=(ModuleConfig,),
    hydra_defaults=[{"optimizer": "adam"}, "_self_"],
)  # beartype not supported, so we use builds instead of full_builds

MNISTModuleConfigXYCE = builds(
    MNISTLitModule,
    net=mnist_narrow_backbone,
    scheduler=None,
    builds_bases=(ModuleConfig,),
    hydra_defaults=[{"optimizer": "adam"}, "_self_"],
)  # beartype not supported, so we use builds instead of full_builds

mnist_module = MNISTModuleConfig()


def _register_configs():
    model_store = store(group="model")
    model_store(mnist_module, name="mnist")
    model_store(MNISTModuleConfig(net=mnist_wide_backbone), name="mnist-wide")
    model_store(MNISTModuleConfigXYCE(net=mnist_wide_backbone), name="mnist-xyce")
