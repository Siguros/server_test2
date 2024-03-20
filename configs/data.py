# from hydra_zen import store
# from hydra.core.config_store import ConfigStore
from hydra_zen import store

import src.data as data
from configs import full_builds

XORModuleConfig = full_builds(data.XORDataModule)

MNISTModuleConfig = full_builds(
    data.MNISTDataModule,
    data_dir="${paths.data_dir}",
    train_val_test_split=[55000, 5000, 10000],
    num_workers=0,
    pin_memory=False,
)
CIFAR10ModuleConfig = full_builds(
    data.CIFAR10DataModule,
    data_dir="${paths.data_dir}",
    train_val_test_split=[45000, 5000, 10000],
    num_workers=2,
    pin_memory=False,
)


def _register_configs():
    data_store = store(group="data")
    data_store(XORModuleConfig, name="xor")
    data_store(MNISTModuleConfig, name="mnist")
    data_store(CIFAR10ModuleConfig, name="cifar10")
