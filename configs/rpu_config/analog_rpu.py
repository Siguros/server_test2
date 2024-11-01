from aihwkit.simulator.configs import ConstantStepDevice, ExpStepDevice, SingleRPUConfig
from hydra_zen import MISSING, ZenField, builds, make_config, store

from configs import full_builds

# Create a configuration for ConstantStepDevice with full signature
ConstantStepDevicecfg = full_builds(
    ConstantStepDevice,
    populate_full_signature=True,  # Make all parameters configurable externally
)

ExpStepDevicecfg = full_builds(
    ExpStepDevice,
    populate_full_signature=True,  # Make all parameters configurable externally
)


# Create a configuration for SingleRPUConfig, allowing the device to be configurable
ConstantstepRPUConfigcfg = full_builds(
    SingleRPUConfig,
    device=ConstantStepDevicecfg,  # Reference the ConstantStepDevice configuration
)

ExpStepDeviceRPUConfigcfg = full_builds(
    SingleRPUConfig,
    device=ExpStepDevicecfg,  # Reference the ConstantStepDevice configuration
)


# Register the config in the ConfigStore under the group "rpu_config"
def _register_configs():
    rpu_store = store(group="rpu_config")
    rpu_store(ConstantstepRPUConfigcfg, name="constant_step")
