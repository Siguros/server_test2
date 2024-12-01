from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice, IdealDevice
from hydra_zen import builds, store

# Define different rpu_config options
ConstantStepConfig = builds(SingleRPUConfig, device=builds(ConstantStepDevice))
IdealPulseConfig = builds(SingleRPUConfig, device=builds(IdealDevice))


# Store them in Hydra Zen's model store
def _register_configs():
    rpu_store = store(group="model/net/rpu_config")
    rpu_store(ConstantStepConfig, name="constant_step")
    rpu_store(IdealPulseConfig, name="ideal_pulse")
