# ruff: noqa: F722
from numpy import ndarray
from torch import Tensor
from jaxtyping import Float

BatchedInput = Float[ndarray | Tensor, "batch in"]
BatchedOutput = Float[ndarray | Tensor, "batch out"]
NpBatchedVecPair = tuple[Float[ndarray, "batch in"], Float[ndarray, "batch out"]]
TorchBatchedVecPair = tuple[Float[Tensor, "batch in"], Float[Tensor, "batch out"]]

StateVec = Float[ndarray, "out*in"]
NpDeviceArray = Float[ndarray, "out*in"]
TorchDeviceArray = Float[Tensor, "out*in"]
DeviceArray = NpDeviceArray | TorchDeviceArray
