from hydra_zen import builds
from torchvision import transforms

from src.data.mnist_datamodule import MNISTDataModule

mnist_transforms = builds(
    transforms.Compose,
    builds_bases=(transforms.ToTensor,),
)

mnist_datamodule = builds(
    MNISTDataModule,
    data_dir="${paths.data_dir}/MNIST",
    train_val_test_split=[55_000, 5_000, 10_000],
    batch_size=64,
    num_workers=0,
    pin_memory=False,
    transforms=mnist_transforms,
    populate_full_signature=True,
)

def _register_configs():
    """Register configs to hydra-zen store."""
    return mnist_datamodule
