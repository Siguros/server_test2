"""This file prepares fixtures for test_kalman.py."""

import pytest
from aihwkit.simulator.configs import LinearStepDevice, SingleRPUConfig
from aihwkit.simulator.tiles.analog import AnalogTile
from omegaconf import DictConfig, OmegaConf

from src.prog_scheme.kalman import LinearDeviceEKF
from src.prog_scheme.utils import generate_target_weights


@pytest.fixture(scope="session")
def prog_cfg() -> DictConfig:
    """Return a programming method related configuration."""
    cfg = OmegaConf.create()
    cfg.input_size = 10
    cfg.output_size = 5
    cfg.rank = 3
    cfg.batch_size = 1
    cfg.tol = 1e-8
    cfg.max_iter = 100
    # method wise config
    cfg.svd_every_k_iter = 1
    cfg.read_noise_std = 0.1
    cfg.update_noise_std = 0.1
    return cfg


@pytest.fixture(scope="session")
def rpu_config():
    """Return a default RPU configuration."""
    return SingleRPUConfig()


@pytest.fixture(scope="session")
def lin_rpu_config():
    """Return a RPU configuration with LinearStepDevice."""
    device = LinearStepDevice()
    return SingleRPUConfig(device=device)


@pytest.fixture(scope="session")
def kf_rpu_config(lin_rpu_config, prog_cfg):
    """Return a RPU configuration for Kalman Filter.

    Remove all device non-linearities.
    """
    lin_rpu_config.forward.out_noise = prog_cfg.read_noise_std
    lin_rpu_config.forward.inp_noise = 0.0
    # LinearStepDevice Attributes
    lin_rpu_config.device.apply_write_noise_on_set = False
    lin_rpu_config.device.gamma_down = 0.1
    lin_rpu_config.device.gamma_down_dtod = 0.0
    lin_rpu_config.device.gamma_up = 0.5
    lin_rpu_config.device.gamma_up_dtod = 0.0
    lin_rpu_config.device.write_noise_std = prog_cfg.update_noise_std
    # PulsedDevice Attributes
    lin_rpu_config.device.w_max = 1
    lin_rpu_config.device.w_min = -1
    lin_rpu_config.device.w_max_dtod = 0.0
    lin_rpu_config.device.w_min_dtod = 0.0
    lin_rpu_config.device.dw_min_std = 0.0
    lin_rpu_config.device.dw_min_dtod = 0.0
    lin_rpu_config.device.mult_noise = False
    lin_rpu_config.device.up_down_dtod = 0.0
    lin_rpu_config.forward.inp_res = 0
    lin_rpu_config.forward.out_res = 0
    return lin_rpu_config


@pytest.fixture(scope="session")
def analogtile(rpu_config, prog_cfg):
    """Return an AnalogTile set with target weights."""
    atile = AnalogTile(
        out_size=prog_cfg.input_size, in_size=prog_cfg.output_size, rpu_config=rpu_config
    )
    target_w = generate_target_weights(prog_cfg.input_size, prog_cfg.output_size, prog_cfg.rank)
    atile.target_weights = target_w.T
    return atile


@pytest.fixture(scope="session")
def kf_analogtile(kf_rpu_config, prog_cfg):
    """Return an AnalogTile set with target weights for Kalman Filter rpu configuration."""
    atile = AnalogTile(
        out_size=prog_cfg.input_size, in_size=prog_cfg.output_size, rpu_config=kf_rpu_config
    )
    target_w = generate_target_weights(prog_cfg.input_size, prog_cfg.output_size, prog_cfg.rank)
    atile.target_weights = target_w.T
    return atile


@pytest.fixture(scope="session")
def linear_ekf(kf_rpu_config, prog_cfg):
    device_params = kf_rpu_config.device
    dim = prog_cfg.input_size * prog_cfg.output_size
    return LinearDeviceEKF(
        dim=dim,
        read_noise_std=prog_cfg.read_noise_std,
        update_noise_std=prog_cfg.update_noise_std,
        **device_params
    )
