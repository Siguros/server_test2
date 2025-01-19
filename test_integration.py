"""
aihwkit과 ml repository의 통합 테스트
"""
import torch
import pytorch_lightning as pl
from torch import nn
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice

class AnalogMLP(pl.LightningModule):
    """아날로그 MLP 모델"""
    
    def __init__(self, input_size=784, hidden_sizes=[256, 128], output_size=10):
        super().__init__()
        self.save_hyperparameters()
        
        # RPU 설정
        rpu_config = SingleRPUConfig(device=ConstantStepDevice())
        
        # 아날로그 레이어
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(sizes)-1):
            layers.append(AnalogLinear(
                sizes[i], 
                sizes[i+1],
                bias=True,
                rpu_config=rpu_config
            ))
            if i < len(sizes)-2:
                layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        
    def configure_optimizers(self):
        from aihwkit.optim import AnalogSGD
        optimizer = AnalogSGD(self.parameters(), lr=0.05)
        optimizer.regroup_param_groups(self)
        return optimizer

def test_cuda_availability():
    """CUDA 가용성 테스트"""
    print("=== CUDA 가용성 테스트 ===")
    print(f"PyTorch CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"사용 가능한 GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

def test_aihwkit_integration():
    """aihwkit 통합 테스트"""
    print("\n=== aihwkit 통합 테스트 ===")
    try:
        from aihwkit.simulator.rpu_base import cuda
        print(f"aihwkit CUDA 컴파일됨: {cuda.is_compiled()}")
        
        # 모델 생성
        model = AnalogMLP()
        print("\n모델 구조:")
        print(model)
        
        # CUDA로 모델 이동
        if torch.cuda.is_available():
            model = model.cuda()
            print("\nCUDA로 모델 이동 성공")
        
        print("\naihwkit 통합 테스트 성공")
    except Exception as e:
        print(f"\naihwkit 통합 테스트 실패: {str(e)}")

def test_lightning_integration():
    """PyTorch Lightning 통합 테스트"""
    print("\n=== PyTorch Lightning 통합 테스트 ===")
    try:
        import pytorch_lightning as pl
        print(f"PyTorch Lightning 버전: {pl.__version__}")
        
        # trainer 생성
        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            max_epochs=1
        )
        print("\nTrainer 생성 성공")
        print("\nPyTorch Lightning 통합 테스트 성공")
    except Exception as e:
        print(f"\nPyTorch Lightning 통합 테스트 실패: {str(e)}")

if __name__ == "__main__":
    test_cuda_availability()
    test_aihwkit_integration()
    test_lightning_integration()
