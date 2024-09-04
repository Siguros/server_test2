from hydra_zen import MISSING, builds, store

from configs import full_builds
from src.core.eqprop.python import activation, solver, strategy
from src.utils import eqprop_utils

IdealRectifierConfig = full_builds(activation.IdealRectifier)
OTSConfig = full_builds(activation.OTS, Is=1e-6, Vth=0.026, Vl=0.1, Vr=0.9)
P3OTSConfig = full_builds(activation.P3OTS, Is=1e-6, Vth=0.02, Vl=-0.5, Vr=0.5)
p3ots_real = P3OTSConfig(Is=4.352e-6, Vth=0.026, Vl=0, Vr=0)
symrelu = full_builds(activation.SymReLU, Vl=-0.6, Vr=0.6)

ParamAdjusterConfig = full_builds(eqprop_utils.AdjustParams)
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

QPStrategyConfig = full_builds(
    strategy.QPStrategy,
    amp_factor="${model.net.solver.amp_factor}",
    activation=MISSING,
    solver_type="proxqp",
    add_nonlin_last=False,
)

ProxQPStrategyConfig = full_builds(
    strategy.ProxQPStrategy,
    amp_factor="${model.net.solver.amp_factor}",
    activation=MISSING,
    add_nonlin_last=False,
    num_threads=None,
)


NewtonStrategyConfig = full_builds(
    strategy.NewtonStrategy,
    clip_threshold=0.5,
    amp_factor="${model.net.solver.amp_factor}",
    eps=1e-6,
    max_iter=25,
    atol=1e-7,
    activation=MISSING,
    add_nonlin_last=False,
    momentum=0.1,
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


AnalogEqPropSolverConfig = builds(
    solver.AnalogEqPropSolver,
    amp_factor=1.0,
    beta="${model.net.beta}",
    strategy=MISSING,
)


def _register_configs():
    activation_store = store(group="model/net/solver/strategy/activation")
    activation_store(IdealRectifierConfig, name="ideal")
    activation_store(OTSConfig, name="ots")
    activation_store(P3OTSConfig, name="p3ots")
    activation_store(p3ots_real, name="p3ots-real")
    activation_store(symrelu, name="symrelu")

    strategy_store = store(group="model/net/solver/strategy")
    strategy_store(GDStrategyConfig, name="gd")
    strategy_store(NewtonStrategyConfig, name="newton")
    strategy_store(QPStrategyConfig, name="qp")
    strategy_store(ProxQPStrategyConfig, name="proxqp")
    strategy_store(XyceStrategyConfig, name="Xyce")
