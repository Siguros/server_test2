import torch
import lightning as L
import aihwkit
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

class SimpleModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(28*28, 10)
    
    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

def test_environment():
    print("Testing PyTorch...")
    x = torch.randn(2, 3)
    print(f"PyTorch tensor shape: {x.shape}")
    
    print("\nTesting Lightning...")
    model = SimpleModel()
    print(f"Lightning model: {model}")
    
    print("\nTesting aihwkit...")
    print(f"aihwkit version: {aihwkit.__version__}")
    
    print("\nTesting CUDA availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    test_environment()
