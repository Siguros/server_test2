from ast import mod

from aihwkit.nn import AnalogLinear, AnalogSequential  # Import aihwkit components
from aihwkit.simulator.configs import ConstantStepDevice, SingleRPUConfig
from hydra_zen import MISSING, ZenField, builds, make_config, store
from torch.nn import MSELoss, ReLU, Sigmoid

from configs import full_builds
from src.models.classifier_module import BinaryClassifierLitModule, ClassifierLitModule
from src.models.components.resnet import resnet
from src.models.components.simple_dense_net import SimpleDenseNet

rpu_config = full_builds(SingleRPUConfig, device=builds(ConstantStepDevice))

AnalogBackboneConfig = builds(
    AnalogSequential,
    builds(AnalogLinear, in_features=784, out_features=128, bias=True, rpu_config="${rpu_config}"),
    builds(ReLU),
    builds(AnalogLinear, in_features=128, out_features=64, bias=True, rpu_config="${rpu_config}"),
    builds(ReLU),
    builds(AnalogLinear, in_features=64, out_features=10, bias=True, rpu_config="${rpu_config}"),
)

MLPBackboneConfig = builds(SimpleDenseNet, populate_full_signature=True)

mnist_narrow_backbone = MLPBackboneConfig(cfg=[784, 128, 64, 10])
mnist_wide_backbone = MLPBackboneConfig(cfg=[784, 256, 256, 10])
xor_backbone = MLPBackboneConfig(
    cfg=[2, 10, 1], batch_norm=False, bias=False
)  # , activation=Sigmoid)

resnet_backboneconfig = builds(
    resnet, in_ch=3, hidden_ch=16, num_layer=3, num_classes=10, populate_full_signature=True
)

xor_onehot_backbone = MLPBackboneConfig(cfg=[2, 10, 2], batch_norm=False, bias=False)

ModuleConfig = make_config(net=MISSING, optimizer=MISSING, scheduler=MISSING, compile=False)

MNISTModuleConfig = builds(
    ClassifierLitModule,
    net=mnist_narrow_backbone,
    scheduler=None,
    num_classes=10,
    builds_bases=(ModuleConfig,),
    hydra_defaults=[{"optimizer": "adam"}, "_self_"],
)  # beartype not supported, so we use builds instead of full_builds

CIFAR10ModuleConfig = builds(
    ClassifierLitModule,
    net=resnet_backboneconfig,
    scheduler=None,
    num_classes=10,
    builds_bases=(ModuleConfig,),
    hydra_defaults=[{"optimizer": "adam"}, "_self_"],
)  # beartype not supported, so we use builds instead of full_builds


XORModuleConfig = builds(
    BinaryClassifierLitModule,
    net=xor_backbone,
    scheduler=None,
    num_classes=1,
    # criterion=MSELoss,
    builds_bases=(ModuleConfig,),
    hydra_defaults=[{"optimizer": "sgd"}, "_self_"],
    populate_full_signature=True,
)

XOROneHotModuleConfig = builds(
    ClassifierLitModule,
    net=xor_onehot_backbone,
    scheduler=None,
    num_classes=2,
    builds_bases=(ModuleConfig,),
    hydra_defaults=[{"optimizer": "sgd"}, "_self_"],
)

MNISTModuleConfigXYCE = builds(
    ClassifierLitModule,
    net=mnist_narrow_backbone,
    scheduler=None,
    builds_bases=(ModuleConfig,),
    hydra_defaults=[{"optimizer": "adam"}, "_self_"],
)  # beartype not supported, so we use builds instead of full_builds

# Define the Analog MNIST Module

AnalogMNISTModuleConfig = builds(
    ClassifierLitModule,
    net=AnalogBackboneConfig,
    scheduler=None,
    num_classes=10,
    builds_bases=(ModuleConfig,),
    hydra_defaults=[
        {"optimizer": "adam"},
        "_self_",
    ],
)


mnist_module = MNISTModuleConfig()
xor_module = XORModuleConfig()
xor_oh_module = XOROneHotModuleConfig()

analog_mnist_module = AnalogMNISTModuleConfig()
CIFAR10Module = CIFAR10ModuleConfig()


def _register_configs():
    model_store = store(group="model")
    model_store(mnist_module, name="mnist")
    model_store(MNISTModuleConfig(net=mnist_wide_backbone), name="mnist-wide")
    model_store(analog_mnist_module, name="analog-mnist")
    model_store(CIFAR10Module, name="resnet_cifar10")
