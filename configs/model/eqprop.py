from dataclasses import dataclass

from hydra_zen import MISSING, builds, make_config, store

from configs import full_builds, partial_builds
from src._eqprop import (
    AnalogEP2,
    DummyAnalogEP2,
    EqPropBinaryLitModule,
    EqPropLitModule,
    EqPropMSELitModule,
)
from src.core.eqprop import eqprop_util, solver, strategy

IdealRectifierConfig = full_builds(eqprop_util.IdealRectifier)
P3OTSConfig = full_builds(eqprop_util.P3OTS, Is=1e-6, Vth=0.02, Vl=0, Vr=0)
p3ots_real = P3OTSConfig(Is=4.352e-6, Vth=0.026, Vl=0, Vr=0)
symrelu = full_builds(eqprop_util.SymReLU, Vl=-0.6, Vr=0.6)

ParamAdjusterConfig = full_builds(eqprop_util.AdjustParams)
xor_adjuster = ParamAdjusterConfig(L=1e-7, clamp=True, normalize=False)

AbstractStrategyConfig = full_builds(strategy.AbstractStrategy)

GDStrategyConfig = full_builds(
    strategy.GradientDescentStrategy,
    alpha=1,
    amp_factor="${model.net.solver.amp_factor}",
    max_iter=50,
    atol=1e-6,
    activation=MISSING,
)

IdealQPStrategyConfig = full_builds(
    strategy.IdealQPStrategy,
    amp_factor="${model.net.solver.amp_factor}",
    activation=MISSING,
)

NewtonStrategyConfig = full_builds(
    strategy.NewtonStrategy,
    clip_threshold=0.5,
    amp_factor="${model.net.solver.amp_factor}",
    eps=1e-6,
    max_iter=25,
    atol=1e-6,
    activation=MISSING,
    add_nonlin_last=False,
)

XyceStrategyConfig = full_builds(
    strategy.XyceStrategy,
    amp_factor="${model.net.solver.amp_factor}",
    SPICE_params={
        "A": "${model.net.solver.amp_factor}",
        "beta": "${model.net.beta}",
        "Diode": {
            "Path": "./src/core/spice/1N4148.lib",
            "ModelName": "1N4148",
            "Rectifier": "BidRectifier",
        },
        "noise": 0,
    },
    mpi_commands=["mpirun", "-use-hwthread-cpus", "-np", "1", "-cpu-set"],
    activation=None,
)


AnalogEqPropSolverConfig = partial_builds(
    solver.AnalogEqPropSolver,
    amp_factor=1.0,
    beta="${model.net.beta}",
    strategy=MISSING,
)

EqPropBackboneConfig = builds(
    AnalogEP2,
    batch_size="${data.batch_size}",
    beta=0.01,
    bias=False,
    positive_w=True,
    min_w=1e-6,
    max_w_gain=0.28,
    scale_input=2,
    scale_output=2,
    cfg=MISSING,
    solver=AnalogEqPropSolverConfig,
    populate_full_signature=True,
    hydra_convert="partial",
)

DummyEqPropBackboneConfig = builds(
    DummyAnalogEP2,
    batch_size="${data.batch_size}",
    builds_bases=(EqPropBackboneConfig,),
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
    cfg="${eval:'[2*${.scale_input}, 2, 1*${.scale_output}]'}",
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

dummy_eqprop_xor = DummyEqPropBackboneConfig(
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
    {"net/solver/strategy": "newton"},
    {"net/solver/strategy/activation": "p3ots"},
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
    activation_store = store(group="model/net/solver/strategy/activation")
    activation_store(IdealRectifierConfig, name="ideal")
    activation_store(P3OTSConfig, name="p3ots")
    activation_store(p3ots_real, name="p3ots-real")
    activation_store(symrelu, name="symrelu")

    strategy_store = store(group="model/net/solver/strategy")
    strategy_store(GDStrategyConfig, name="gd")
    strategy_store(NewtonStrategyConfig, name="newton")
    strategy_store(IdealQPStrategyConfig, name="ideal-qp")
    strategy_store(XyceStrategyConfig, name="Xyce")

    model_store = store(group="model")
    model_store(EqPropXORModuleConfig, name="ep-xor")
    model_store(EqPropXOROHModuleConfig, name="ep-xor-onehot")
    model_store(EqPropMNISTModuleConfig, name="ep-mnist")
    model_store(ep_mnist_adamw, name="ep-mnist-adamw")
    model_store(EqPropMNISTMSEModuleConfig, name="ep-mnist-mse")
