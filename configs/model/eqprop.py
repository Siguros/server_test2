from hydra_zen import MISSING, builds, make_config, store

from configs.eqprop.solver import AnalogEqPropSolverConfig, ParamAdjusterConfig, xor_adjuster
from src._eqprop import AnalogEP2, EqPropBinaryLitModule, EqPropLitModule, EqPropMSELitModule

EqPropBackboneConfig = builds(
    AnalogEP2,
    batch_size="${data.batch_size}",
    beta=0.01,
    bias=False,
    positive_w=True,
    min_w=1e-6,
    max_w_gain=0.08,
    scale_input=2,
    scale_output=2,
    cfg=MISSING,
    solver=AnalogEqPropSolverConfig,
    populate_full_signature=True,
    hydra_convert="partial",
)


eqprop_mnist = EqPropBackboneConfig(
    cfg="${eval:'[784*${.scale_input}, 128, 10*${.scale_output}]'}",
)

eqprop_xor = EqPropBackboneConfig(
    beta=0.001,
    scale_input=1,
    scale_output=2,
    min_w=0.0001,
    max_w=0.1,
    max_w_gain=None,
    cfg="${eval:'[3*${.scale_input}, 2, 1*${.scale_output}]'}",  # change to 2
)

eqprop_xor_onehot = EqPropBackboneConfig(
    beta=0.001,
    scale_input=1,
    scale_output=2,
    min_w=0.0001,
    max_w=0.1,
    max_w_gain=None,
    cfg="${eval:'[2*${.scale_input}, 10, 2*${.scale_output}]'}",
)


EqPropModuleConfig = make_config(
    scheduler=None,
    optimizer=MISSING,
    net=MISSING,
    compile=False,
    param_adjuster=MISSING,
)


ep_defaults = [
    "_self_",
    {"optimizer": "sgd"},
    {"net/solver/strategy": "proxqp"},
    {"net/solver/strategy/activation": "ots"},
]
EqPropXORModuleConfig = builds(
    EqPropBinaryLitModule,
    net=eqprop_xor,
    num_classes=1,
    param_adjuster=xor_adjuster,
    builds_bases=(EqPropModuleConfig,),
    hydra_defaults=ep_defaults,
    populate_full_signature=True,
)

EqPropXOROHModuleConfig = builds(
    EqPropLitModule,
    net=eqprop_xor_onehot,
    num_classes=2,
    scheduler=None,
    param_adjuster=xor_adjuster,
    builds_bases=(EqPropModuleConfig,),
    hydra_defaults=ep_defaults,
)

EqPropMNISTModuleConfig = builds(
    EqPropLitModule,
    net=eqprop_mnist,
    param_adjuster=ParamAdjusterConfig(),
    builds_bases=(EqPropModuleConfig,),
    hydra_defaults=ep_defaults,
)

ep_mnist_adamw = builds(
    EqPropLitModule,
    net=eqprop_mnist,
    builds_bases=(EqPropModuleConfig,),
    hydra_defaults=["_self_", {"override /optimizer": "adamw"}],
)

EqPropMNISTMSEModuleConfig = builds(
    EqPropMSELitModule,
    net=eqprop_mnist,
    param_adjuster=ParamAdjusterConfig(),
    builds_bases=(EqPropModuleConfig,),
    hydra_defaults=ep_defaults,
)


def _register_configs():
    model_store = store(group="model")
    model_store(EqPropXORModuleConfig, name="ep-xor")
    model_store(EqPropXOROHModuleConfig, name="ep-xor-onehot")
    model_store(EqPropMNISTModuleConfig, name="ep-mnist")
    model_store(ep_mnist_adamw, name="ep-mnist-adamw")
    model_store(EqPropMNISTMSEModuleConfig, name="ep-mnist-mse")
