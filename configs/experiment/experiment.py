from hydra_zen import make_config, store

xor_config = make_config(
    hydra_defaults=[
        "_self_",
        {"override /data": "xor"},
        {"override /model": "xor"},
    ],
    trainer=dict(max_epochs=500),
    callbacks=dict(early_stopping=None),
    model=dict(optimizer=dict(lr=0.02, momentum=0.9)),
)

xor_onehot_config = make_config(
    bases=(xor_config,),
    hydra_defaults=[
        "_self_",
        {"override /model": "xor-onehot"},
    ],
)


def _register_configs():
    exp_store = store(group="experiment", package="_global_")
    exp_store(xor_config, name="xor")
    exp_store(xor_onehot_config, name="xor-onehot")
