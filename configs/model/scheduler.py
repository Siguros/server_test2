# Original code from: https://github.com/groomata/vision/blob/main/src/groovis/configs/scheduler.py

from hydra_zen import store
from torch.optim import lr_scheduler

from configs import partial_builds

OneCycleLRConfig = partial_builds(
    lr_scheduler.OneCycleLR,
    max_lr="${optimizer.lr}",
    pct_start=0.2,
    anneal_strategy="linear",
    div_factor=1e3,
    final_div_factor=1e6,
)
StepLRConfig = partial_builds(
    lr_scheduler.StepLR,
    step_size=30,
    gamma=0.1,
)
ReduceLROnPlateauConfig = partial_builds(
    lr_scheduler.ReduceLROnPlateau,
    mode="min",
    factor=0.1,
    patience=10,
    threshold=0.0001,
    threshold_mode="rel",
    cooldown=0,
    min_lr=0,
    eps=1e-08,
    verbose=False,
)


def _register_configs():
    sched_store = store(group="model/scheduler")

    sched_store(OneCycleLRConfig, name="onecycle")
    sched_store(StepLRConfig, name="steplr")
    sched_store(ReduceLROnPlateauConfig, name="reducelronplateau")
