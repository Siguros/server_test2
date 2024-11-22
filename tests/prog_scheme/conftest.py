"""This file prepares fixtures for test_kalman.py."""

import pytest
from aihwkit.simulator.configs import (
    ConstantStepDevice,
    IdealDevice,
    LinearStepDevice,
    SingleRPUConfig,
)
from aihwkit.simulator.configs.utils import UpdateParameters
from aihwkit.simulator.parameters.enums import PulseType
from aihwkit.simulator.tiles.analog import AnalogTile
from omegaconf import DictConfig, OmegaConf

from src.prog_scheme.kalman import DeviceKF, LinearDeviceEKF
from src.prog_scheme.utils import generate_target_weights


@pytest.fixture(scope="session")
def prog_cfg() -> DictConfig:
    """Return a programming method related configuration."""
    cfg = OmegaConf.create()
    cfg.input_size = 5
    cfg.output_size = 3
    cfg.rank = 2
    cfg.batch_size = 1
    cfg.tol = 1e-8
    cfg.max_iter = 100
    # method wise config
    cfg.svd_every_k_iter = 1
    cfg.read_noise_std = 0.01
    cfg.update_noise_std = 0.01
    return cfg


@pytest.fixture(scope="session")
def rpu_config():
    """Return a default RPU configuration."""
    return SingleRPUConfig()


@pytest.fixture(scope="session")
def ideal_rpu_config():
    """Return a RPU configuration with ideal device."""
    device = IdealDevice()
    return SingleRPUConfig(device=device)


@pytest.fixture(scope="session")
def lin_rpu_config():
    """Return a RPU configuration with LinearStepDevice."""
    device = LinearStepDevice()
    update = UpdateParameters(
        pulse_type=PulseType.MEAN_COUNT,
        desired_bl=127,
        update_bl_management=False,
        update_management=False,
    )
    return SingleRPUConfig(device=device, update=update)


@pytest.fixture(scope="session")
def kf_rpu_config(lin_rpu_config, prog_cfg):
    """Return a RPU configuration for Kalman Filter.

    Remove all device non-linearities.
    """
    # lin_rpu_config.forward.is_perfect = True
    lin_rpu_config.forward.out_noise = prog_cfg.read_noise_std
    lin_rpu_config.forward.inp_noise = 0.0
    lin_rpu_config.forward.inp_res = 0
    lin_rpu_config.forward.out_res = 0
    # update Attributes
    lin_rpu_config.update.desired_bl = 2**11
    # LinearStepDevice Attributes
    lin_rpu_config.device.apply_write_noise_on_set = False
    lin_rpu_config.device.gamma_down = 0.1
    lin_rpu_config.device.gamma_down_dtod = 0.0
    lin_rpu_config.device.gamma_up = 0.3
    lin_rpu_config.device.gamma_up_dtod = 0.0
    lin_rpu_config.device.write_noise_std = prog_cfg.update_noise_std
    # PulsedDevice Attributes
    lin_rpu_config.device.w_max = 1
    lin_rpu_config.device.w_min = -1
    lin_rpu_config.device.w_max_dtod = 0.0
    lin_rpu_config.device.w_min_dtod = 0.0
    lin_rpu_config.device.dw_min = 1e-4
    lin_rpu_config.device.dw_min_std = 0.0
    lin_rpu_config.device.dw_min_dtod = 0.0
    lin_rpu_config.device.mult_noise = False
    lin_rpu_config.device.up_down_dtod = 0.0

    return lin_rpu_config


@pytest.fixture(scope="session")
def const_rpu_config_idealized():
    """Return a RPU configuration behaves ideally with ConstStepDevice."""
    device = ConstantStepDevice()
    update = UpdateParameters(
        pulse_type=PulseType.MEAN_COUNT,
        desired_bl=2**11 - 1,
        update_bl_management=False,
        update_management=False,
    )
    rpu_config = SingleRPUConfig(device=device, update=update)
    # lin_rpu_config.forward.is_perfect = True
    rpu_config.forward.out_noise = 0
    rpu_config.forward.inp_noise = 0.0
    rpu_config.forward.inp_res = 0
    rpu_config.forward.out_res = 0
    # ConstStepDevice Attributes
    # PulsedDevice Attributes
    rpu_config.device.w_max = 1
    rpu_config.device.w_min = -1
    rpu_config.device.w_max_dtod = 0.0
    rpu_config.device.w_min_dtod = 0.0
    rpu_config.device.dw_min_std = 0.0
    rpu_config.device.dw_min = 1e-5
    rpu_config.device.dw_min_dtod = 0.0
    rpu_config.device.up_down = 0.0
    rpu_config.device.up_down_dtod = 0.0
    return rpu_config


@pytest.fixture(scope="session")
def analogtile(rpu_config, prog_cfg):
    """Return an AnalogTile set with target weights."""
    atile = AnalogTile(
        out_size=prog_cfg.input_size, in_size=prog_cfg.output_size, rpu_config=rpu_config
    )
    target_w = generate_target_weights(prog_cfg.input_size, prog_cfg.output_size, prog_cfg.rank)
    atile.target_weights = target_w.T
    atile.tile.set_learning_rate(1)
    return atile


@pytest.fixture(scope="class")
def ideal_analogtile(ideal_rpu_config, prog_cfg):
    """Return an AnalogTile set with target weights for ideal rpu configuration."""
    atile = AnalogTile(
        out_size=prog_cfg.input_size, in_size=prog_cfg.output_size, rpu_config=ideal_rpu_config
    )
    target_w = generate_target_weights(prog_cfg.input_size, prog_cfg.output_size, prog_cfg.rank)
    atile.target_weights = target_w.T
    atile.tile.set_learning_rate(1)
    return atile


@pytest.fixture(scope="class")
def idealized_analogtile(const_rpu_config_idealized):
    """Return an AnalogTile set with target weights for ideal rpu configuration."""
    atile = AnalogTile(out_size=2, in_size=2, rpu_config=const_rpu_config_idealized)
    atile.tile.set_learning_rate(1)
    return atile


@pytest.fixture(scope="class")
def kf_analogtile(kf_rpu_config, prog_cfg):
    """Return an AnalogTile set with target weights for Kalman Filter rpu configuration."""
    atile = AnalogTile(
        out_size=prog_cfg.input_size, in_size=prog_cfg.output_size, rpu_config=kf_rpu_config
    )
    target_w = generate_target_weights(prog_cfg.input_size, prog_cfg.output_size, prog_cfg.rank)
    atile.target_weights = target_w.T
    atile.tile.set_learning_rate(1)
    return atile


@pytest.fixture(scope="function")
def base_kf(prog_cfg):
    dim = prog_cfg.input_size * prog_cfg.output_size
    return DeviceKF(
        dim=dim, read_noise_std=prog_cfg.read_noise_std, update_noise_std=prog_cfg.update_noise_std
    )


@pytest.fixture(scope="function")
def linear_ekf(kf_rpu_config, prog_cfg):
    device_params = kf_rpu_config.device.__dict__
    dim = prog_cfg.input_size * prog_cfg.output_size
    return LinearDeviceEKF(
        dim=dim,
        read_noise_std=prog_cfg.read_noise_std,
        update_noise_std=prog_cfg.update_noise_std,
        iterative_update=False,
        **device_params
    )
