from hydra_zen import store
from torch.nn import CrossEntropyLoss, MSELoss

from configs import full_builds, partial_builds

MSELossConfig = full_builds(MSELoss)
mse_sum = MSELossConfig(reduction="sum")
CELossConfig = full_builds(CrossEntropyLoss)


def _register_configs():
    pass
    # loss_store = store(group="model/loss")

    # loss_store(MSELossConfig, name="mse")
    # loss_store(mse_sum, name="mse-sum")
    # loss_store(CELossConfig, name="ce")
