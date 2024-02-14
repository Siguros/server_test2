import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import open_dict

from src.eqprop import eqprop_util, strategy
from src.models.components.eqprop_backbone import AnalogEP2


@pytest.fixture(scope="module")
def second_order_strategy(toymodel) -> strategy.SecondOrderStrategy:
    st = strategy.NewtonStrategy(
        activation=eqprop_util.SymReLU(Vl=-0.6, Vr=0.6),
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
    cfg = cfg_train_global.model.net.copy()
    HydraConfig().set_config(cfg)
    with open_dict(cfg):
        cfg.batch_size = 1
        cfg.dims = [2, 1, 1]
        cfg.beta = 0.1
        cfg.solver.max_iter = 20
        cfg.solver.strategy.activation = {
            "_target_": "src.eqprop.eqprop_util.SymReLU",
            "Vl": -0.6,
            "Vr": 0.6,
        }
    backbone: AnalogEP2 = instantiate(cfg)
    backbone.model[0].weight.data = torch.tensor([[1.0, 1.0]])
    backbone.model[0].bias.data = torch.tensor([1.0])
    backbone.model[1].weight.data = torch.tensor([[2.0]])
    backbone.model[1].bias.data = torch.tensor([0.0])
    return backbone
