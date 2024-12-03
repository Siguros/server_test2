from hydra_zen import MISSING, ZenField, builds, make_config, store
from torch.nn import MSELoss, ReLU, Sigmoid

from configs import full_builds
from src.models.classifier_module import BinaryClassifierLitModule, ClassifierLitModule
from src.models.components.resnet import ResNet

resnet_backboneconfig = builds(
    ResNet, in_ch=3, hidden_ch=16, num_layer=3, num_classes=10, populate_full_signature=True
)

ModuleConfig = make_config(net=MISSING, optimizer=MISSING, scheduler=MISSING, compile=False)

CIFAR10ModuleConfig = builds(
    ClassifierLitModule,
    net=resnet_backboneconfig,
    scheduler=None,
    num_classes=10,
    builds_bases=(ModuleConfig,),
    hydra_defaults=[{"optimizer": "adam"}, "_self_"],
)  # beartype not supported, so we use builds instead of full_builds


CIFAR10Module = CIFAR10ModuleConfig()


def _register_configs():
    model_store = store(group="model")
    model_store(CIFAR10Module, name="resnet_cifar10")
