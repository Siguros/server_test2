from ast import mod

from hydra_zen import MISSING, builds, make_config, store

from configs import full_builds
from src.models.classifier_module import ClassifierLitModule
from src.models.components.simple_dense_net import SimpleDenseNet

MLPBackboneConfig = builds(SimpleDenseNet, populate_full_signature=True)

mnist_narrow_backbone = MLPBackboneConfig(cfg=[784, 128, 64, 10])
mnist_wide_backbone = MLPBackboneConfig(cfg=[784, 256, 256, 10])
xor_backbone = MLPBackboneConfig(cfg=[2, 1, 2], batch_norm=False)

ModuleConfig = make_config(net=MISSING, optimizer=MISSING, scheduler=MISSING, compile=False)

MNISTModuleConfig = builds(
    ClassifierLitModule,
    net=mnist_narrow_backbone,
    scheduler=None,
    num_classes=10,
    builds_bases=(ModuleConfig,),
    hydra_defaults=[{"optimizer": "adam"}, "_self_"],
)  # beartype not supported, so we use builds instead of full_builds

XORModuleConfig = builds(
    ClassifierLitModule,
    net=xor_backbone,
    scheduler=None,
    num_classes=2,
    builds_bases=(ModuleConfig,),
    hydra_defaults=[{"optimizer": "sgd"}, "_self_"],
)

mnist_module = MNISTModuleConfig()
xor_mpdule = XORModuleConfig()


def _register_configs():
    model_store = store(group="model")
    model_store(mnist_module, name="mnist")
    model_store(MNISTModuleConfig(net=mnist_wide_backbone), name="mnist-wide")
    model_store(xor_mpdule, name="xor")
