import platform
import shutil

import pkg_resources
from lightning.fabric.accelerators import TPUAccelerator


def _package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    :param package_name: The name of the package to be checked.

    :return: `True` if the package is available. `False` otherwise.
    """
    try:
        return pkg_resources.require(package_name) is not None
    except pkg_resources.DistributionNotFound:
        return False


def _lib_available(lib_name: str) -> bool:
    """Check if a library is available in your environment."""
    result = shutil.which(lib_name)
    if result:
        return True
    else:
        return False


_TPU_AVAILABLE = TPUAccelerator.is_available()

_IS_WINDOWS = platform.system() == "Windows"

_SH_AVAILABLE = not _IS_WINDOWS and _package_available("sh")

_DEEPSPEED_AVAILABLE = not _IS_WINDOWS and _package_available("deepspeed")
_FAIRSCALE_AVAILABLE = not _IS_WINDOWS and _package_available("fairscale")

_WANDB_AVAILABLE = _package_available("wandb")
_NEPTUNE_AVAILABLE = _package_available("neptune")
_COMET_AVAILABLE = _package_available("comet_ml")
_MLFLOW_AVAILABLE = _package_available("mlflow")
_SCIPY_AVAILABLE = _package_available("scipy")
_XYCE_AVAILABLE = _lib_available("Xyce") and _package_available("pyspice")
_OPTUNA_PLUGIN_AVAILABLE = _package_available("hydra-optuna-sweeper")
