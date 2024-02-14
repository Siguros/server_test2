from pathlib import Path

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.data.mnist_datamodule import MNISTDataModule
from tests.helpers.run_if import RunIf


def load_batch():
    data_dir = "data/"
    dm = MNISTDataModule(data_dir=data_dir, batch_size=1)
    dm.prepare_data()
    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    dm_it = iter(dm.train_dataloader())
    yield next(dm_it)


@pytest.fixture(scope="function")
def eqprop_torch(batch, model, beta):
    # run 1 free & nudge phase together
    vout_torch = model(batch)
    return vout_torch


@RunIf(xyce=True)
@pytest.fixture(scope="function")
def eqprop_xyce(batch, model, beta):
    # reshape vout to match torch version
    vout_xyce = model(batch)
    return vout_xyce


def test_eqprop_precision(cfg_train, ckpt_path):
    """Compare eqprop_torch with eqprop_xyce.

    1 free & nudge phase each.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.ckpt_path = str(ckpt_path / "checkpoints" / "last.ckpt")

    # compare when initialized with same weights
    beta = cfg_train.model.beta
    model = torch.load(cfg_train.ckpt_path)
    batch = load_batch()
    vout_torch = eqprop_torch(batch, model, beta)
    vout_xyce = eqprop_xyce(batch, model, beta)
    assert vout_torch.shape == vout_xyce.shape, "Shapes do not match. check eqprop_xyce."
    assert torch.allclose(
        vout_torch, vout_xyce, atol=cfg_train.model.net.solver.atol
    ), "Values do not match. check eqprop solver."


class TestAnalogEP2:
    @pytest.mark.parametrize("x", torch.tensor([[-1.0, -1.0]]))
    def test_free(self, toy_backbone, x):
        ypred = toy_backbone.solver(x)
        assert torch.allclose(ypred.squeeze(), torch.tensor(-1.2))

    @pytest.mark.parametrize("x", torch.tensor([[-1.0, -1.0]]))
    def test_nudge(self, toy_backbone, x):
        criterion = torch.nn.MSELoss()
        loss = criterion(toy_backbone.model.ypred, torch.tensor([0.0]))
        loss.backward()
        assert torch.allclose(loss.grad, torch.tensor([-1.2]))
        ypred = toy_backbone.solver(x, nudge_phase=True)
        assert torch.allclose(ypred, [])

    @pytest.mark.parametrize("x", torch.tensor([[-1.0, -1.0]]))
    def test_update(self, toy_backbone, x):
        toy_backbone.eqprop(x)
        net = toy_backbone.model
        assert torch.allclose(net.lin1.weight.grad, torch.tensor([[0.0, 0.0]]))
        assert torch.allclose(net.lin1.bias.grad, torch.tensor([0.0]))
        assert torch.allclose(net.last.weight.grad, torch.tensor([[0.0]]))
        assert torch.allclose(net.last.bias.grad, torch.tensor([0.0]))
