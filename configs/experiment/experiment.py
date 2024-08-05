from hydra_zen import just, make_config, store
from torch.nn import MSELoss

xor_config = make_config(
    hydra_defaults=[
        "_self_",
        {"override /data": "xor"},
        {"override /model": "xor"},
    ],
    trainer=dict(detect_anomaly=True, max_epochs=500),
    callbacks=dict(early_stopping=None),
    model=dict(optimizer=dict(lr=0.001, momentum=0.9)),
)

xor_onehot_config = make_config(
    bases=(xor_config,),
    hydra_defaults=[
        "_self_",
        {"override /data": "xor"},
        {"override /model": "xor-onehot"},
    ],
)
ep_xor_config = make_config(
    bases=(xor_config,),
    model=dict(optimizer=dict(lr=0.001, momentum=0.0), net=dict(bias=False)),
    hydra_defaults=[
        "_self_",
        {"override /data": "xor"},
        {"override /model": "ep-xor"},
        {"override /model/net/solver/strategy": "qp"},
    ],
)

ep_xor_mse_config = make_config(
    bases=(xor_config,),
    model=dict(
        optimizer=dict(lr=0.001, momentum=0.0), net=dict(bias=False), criterion=just(MSELoss)
    ),
    hydra_defaults=[
        "_self_",
        {"override /data": "xor"},
        {"override /model": "ep-xor"},
        {"override /model/net/solver/strategy": "qp"},
    ],
)

ep_xor_mse_xyce_config = make_config(
    bases=(xor_config,),
    model=dict(
        optimizer=dict(lr=0.001, momentum=0.0), net=dict(bias=True), criterion=just(MSELoss)
    ),
    hydra_defaults=[
        "_self_",
        {"override /data": "xor"},
        {"override /model": "ep-xor"},
        {"override /model/net/solver/strategy": "Xyce"},
    ],
)

ep_xor_onehot_config = make_config(
    bases=(xor_config,),
    hydra_defaults=[
        "_self_",
        {"override /data": "xor"},
        {"override /model": "ep-xor-onehot"},
        {"override /model/net/solver/strategy": "newton"},
    ],
)


ep_xor_dummy_config = make_config(
    bases=(xor_onehot_config,),
    hydra_defaults=[
        "_self_",
        {"override /data": "xor"},
        {"override /model": "ep-xor-dummy"},
    ],
)

ep_mnist_config = make_config(
    hydra_defaults=[
        "_self_",
        {"override /data": "mnist"},
        {"override /model": "ep-mnist"},
        {"override /model/net/solver/strategy": "proxqp"},
    ],
    model=dict(optimizer=dict(lr=0.1, momentum=0.9)),
    data=dict(batch_size=64),
)


def _register_configs():
    exp_store = store(group="experiment", package="_global_")
    exp_store(xor_config, name="xor")
    exp_store(xor_onehot_config, name="xor-onehot")
    exp_store(ep_xor_config, name="ep-xor")
    exp_store(ep_xor_mse_config, name="ep-xor-mse")
    exp_store(ep_xor_onehot_config, name="ep-xor-onehot")
    exp_store(ep_xor_dummy_config, name="ep-xor-dummy")
    exp_store(ep_mnist_config, name="ep-mnist")
    exp_store(ep_xor_mse_xyce_config, name="ep-xor-mse-xyce")
