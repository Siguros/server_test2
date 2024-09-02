from hydra_zen import MISSING, builds, make_config, store
from torch.nn import MSELoss, Sigmoid

from configs.eqprop.solver import ParamAdjusterConfig
from src._eqprop import EqPropBackbone
from src.models.classifier_module import (
    BinaryClassifierLitModule,
    ClassifierLitModule,
    MSELitModule,
)
from src.models.components.simple_dense_net import SimpleDenseNet

MLPBackboneConfig = builds(SimpleDenseNet, populate_full_signature=True)

mnist_narrow_backbone = MLPBackboneConfig(cfg=[784, 128, 64, 10])
mnist_wide_backbone = MLPBackboneConfig(cfg=[784, 256, 256, 10])
xor_backbone = MLPBackboneConfig(
    cfg=[2, 10, 1], batch_norm=False, bias=False
)  # , activation=Sigmoid)
xor_onehot_backbone = MLPBackboneConfig(cfg=[2, 10, 2], batch_norm=False, bias=False)

EqPropBackboneConfig = builds(
    EqPropBackbone,
    scale_input=2,
    scale_output=2,
    cfg=MISSING,
    param_adjuster=ParamAdjusterConfig(),
    populate_full_signature=True,
)
eqprop_backbone = EqPropBackboneConfig(
    cfg="${eval:'[784*${.scale_input}, 128, 10*${.scale_output}]'}"
)

ModuleConfig = make_config(net=MISSING, optimizer=MISSING, scheduler=MISSING, compile=False)

CEModuleConfig = builds(
    ClassifierLitModule,
    scheduler=None,
    num_classes=MISSING,
    builds_bases=(ModuleConfig,),
    hydra_defaults=[{"optimizer": "adam"}, {"net": "mnist-narrow"}, "_self_"],
)  # beartype not supported, so we use builds instead of full_builds

MSEModuleConfig = builds(
    MSELitModule,
    scheduler=None,
    num_classes=MISSING,
    builds_bases=(ModuleConfig,),
    hydra_defaults=[{"optimizer": "adam"}, {"net": "mnist-narrow"}, "_self_"],
)

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

basic_module = CEModuleConfig(num_classes=10)
mse_module = MSEModuleConfig(num_classes=10)
xor_module = XORModuleConfig()
xor_oh_module = XOROneHotModuleConfig()


def _register_configs():
    backbone_store = store(group="model/net")
    backbone_store(eqprop_backbone, name="eqprop")
    backbone_store(mnist_narrow_backbone, name="mnist-narrow")
    backbone_store(mnist_wide_backbone, name="mnist-wide")

    model_store = store(group="model")
    model_store(basic_module, name="mnist")
    model_store(mse_module, name="mnist-mse")
