from torch import nn, optim
from hydra_zen import builds, make_config

from src.models.classifier_module import ClassifierLitModule
from src.models.components.simple_dense_net import SimpleDenseNet

# Build the model configuration
mnist_model = builds(
    ClassifierLitModule,
    net=builds(
        SimpleDenseNet,
        cfg=[784, 128, 256, 64, 10],
        batch_norm=True,
        bias=True,
    ),
    optimizer=builds(
        optim.Adam,
        lr=0.001,
        weight_decay=0.0,
        populate_full_signature=True,
    ),
    criterion=builds(nn.CrossEntropyLoss),
    compile=False,
    populate_full_signature=True,
)

def _register_configs():
    """Register configs to hydra-zen store."""
    return mnist_model
