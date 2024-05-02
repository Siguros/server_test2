from hydra_zen import make_config, store

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
    model=dict(optimizer=dict(lr=0.001, momentum=0.0), net=dict(bias=True)),
    hydra_defaults=[
        "_self_",
        {"override /data": "xor"},
        {"override /model": "ep-xor"},
        {"override /model/net/solver/strategy": "newton"},
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
    ],
    model=dict(optimizer=dict(lr=0.1, momentum=0.9)),
)


def _register_configs():
    exp_store = store(group="experiment", package="_global_")
    exp_store(xor_config, name="xor")
    exp_store(xor_onehot_config, name="xor-onehot")
    exp_store(ep_xor_config, name="ep-xor")
    exp_store(ep_xor_onehot_config, name="ep-xor-onehot")
    exp_store(ep_xor_dummy_config, name="ep-xor-dummy")
    exp_store(ep_mnist_config, name="ep-mnist")
