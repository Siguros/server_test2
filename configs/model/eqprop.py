from dataclasses import dataclass

from hydra_zen import MISSING, builds, make_config, store

from configs import full_builds, partial_builds
from src._eqprop import AnalogEP2, EqPropLitModule, EqPropMSELitModule
from src.core.eqprop import eqprop_util, solver, strategy

P3OTSConfig = full_builds(eqprop_util.P3OTS, Is=1e-6, Vth=0.02, Vl=0, Vr=0)
p3ots_real = P3OTSConfig(Is=4.352e-6, Vth=0.026, Vl=0, Vr=0)
symrelu = full_builds(eqprop_util.SymReLU, Vl=-0.6, Vr=0.6)

AbstractStrategyConfig = full_builds(strategy.AbstractStrategy)

GDStrategyConfig = full_builds(
    strategy.GradientDescentStrategy,
    alpha=1,
    amp_factor="${model.net.solver.amp_factor}",
    max_iter=50,
    atol=1e-6,
    activation=P3OTSConfig,
)

NewtonStrategyConfig = full_builds(
    strategy.NewtonStrategy,
    clip_threshold=0.5,
    amp_factor="${model.net.solver.amp_factor}",
    eps=1e-6,
    max_iter=25,
    atol=1e-6,
    activation=P3OTSConfig,
)

AnalogEqPropSolverConfig = partial_builds(
    solver.AnalogEqPropSolver,
    amp_factor=1.0,
    beta="${model.net.beta}",
    strategy=MISSING,
)

EqPropBackboneConfig = partial_builds(
    AnalogEP2,
    batch_size="${data.batch_size}",
    input_size=MISSING,
    lin1_size=MISSING,
    output_size=MISSING,
    beta=0.01,
    solver=AnalogEqPropSolverConfig,
)

eqprop_mnist = EqPropBackboneConfig(
    input_size="${eval:'784*${model.scale_input}'}",
    lin1_size=128,
    output_size="${eval:'10*${model.scale_output}'}",
)

eqprop_xor = EqPropBackboneConfig(
    input_size=2,
    lin1_size=2,
    output_size=2,
)

EqPropModuleConfig = make_config(
    scheduler=None,
    optimizer=MISSING,
    net=MISSING,
    scale_input=2,
    scale_output=2,
    positive_w=True,
    bias=False,
    clip_weights=True,
    normalize_weights=False,
    min_w=1e-6,
    max_w_gain=0.28,
)
ep_defaults = ["_self_", {"optimizer": "sgd"}, {"net/solver/strategy": "newton"}]

EqPropXORModuleConfig = builds(
    EqPropLitModule,
    net=eqprop_xor,
    builds_bases=(EqPropModuleConfig,),
    hydra_defaults=ep_defaults,
)

EqPropMNISTModuleConfig = builds(
    EqPropLitModule,
    net=eqprop_mnist,
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
    builds_bases=(EqPropModuleConfig,),
)


def _register_configs():
    activation_store = store(group="model/net/solver/strategy/activation")
    activation_store(P3OTSConfig, name="p3ots")
    activation_store(p3ots_real, name="p3ots-real")
    activation_store(symrelu, name="symrelu")

    strategy_store = store(group="model/net/solver/strategy")
    strategy_store(GDStrategyConfig, name="gd")
    strategy_store(NewtonStrategyConfig, name="newton")

    model_store = store(group="model")
    model_store(EqPropXORModuleConfig, name="ep-xor")
    model_store(EqPropMNISTModuleConfig, name="ep-mnist")
    model_store(ep_mnist_adamw, name="ep-mnist-adamw")
    model_store(EqPropMNISTMSEModuleConfig, name="ep-mnist-mse")
