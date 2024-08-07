import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import open_dict

from src._eqprop.eqprop_backbone import AnalogEP2
from src.core.eqprop.python import activation, strategy


@pytest.fixture(scope="module")
def second_order_strategy(toymodel) -> strategy.SecondOrderStrategy:
    """A pytest fixture for instantiating a second order EqProp strategy.

    Args:
        toymodel (_type_): _description_

    Returns:
        strategy.SecondOrderStrategy: _description_
    """
    st = strategy.NewtonStrategy(
        activation=activation.SymReLU(Vl=-0.6, Vr=0.6),
        amp_factor=1.0,
        add_nonlin_last=False,
        max_iter=1,
        atol=1e-4,
        clip_threshold=1,
    )
    st.set_strategy_params(toymodel)
    return st


@pytest.fixture(scope="module")
def toy_backbone(cfg_train_global) -> AnalogEP2:
    """A pytest fixture for instantiating a toy EqProp backbone.

    Args:
        cfg_train_global (_type_): _description_

    Returns:
        AnalogEP2: _description_
    """
    cfg = cfg_train_global.copy()
    HydraConfig().set_config(cfg)
    with open_dict(cfg):
        # cfg.paths.out
        cfg.model.net.batch_size = 1
        cfg.model.net.dims = [2, 1, 1]
        cfg.model.net.beta = 0.1
        cfg.model.net.solver.strategy.add_nonlin_last = False
        cfg.model.net.solver.strategy.clip_threshold = 1
        cfg.model.net.solver.strategy.max_iter = 20
        cfg.model.net.solver.strategy.activation = {
            "_target_": "src.utils.eqprop_utils.SymReLU",
            "Vl": -0.6,
            "Vr": 0.6,
        }
    backbone_: AnalogEP2 = instantiate(cfg.model.net)
    backbone = backbone_(hyper_params={"bias": True})
    backbone.model[0].weight.data = torch.tensor([[1.0, 1.0]])
    backbone.model[0].bias.data = torch.tensor([1.0])
    backbone.model[1].weight.data = torch.tensor([[2.0]])
    backbone.model[1].bias.data = torch.tensor([0.0])
    return backbone
